import random
import types
import torch
import numpy as np
import ot as pot
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from tqdm import tqdm
import math
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

QM9_ATOM_ENCODING = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
GEOM_ATOM_ENCODING = {
        'H': 0,
        'B': 1,
        'C': 2,
        'N': 3,
        'O': 4,
        'F': 5,
        'Al': 6,
        'Si': 7,
        'P': 8,
        'S': 9,
        'Cl': 10,
        'As': 11,
        'Br': 12,
        'I': 13,
        'Hg': 14,
        'Bi': 15
        }

def gradient_clipping(model, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    if len(gradnorm_queue) == 0:
        max_grad_norm = float('inf')  # No clipping if queue is empty
    elif len(gradnorm_queue) == 1:
        max_grad_norm = 2.0 * gradnorm_queue.mean()  # More conservative with single sample
    else:
        max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Handle DDP wrapper - get underlying module's parameters
    if hasattr(model, 'module'):
        params = model.module.parameters()
    else:
        params = model.parameters()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        params, max_norm=max_grad_norm, norm_type=2.0)
    
    # Convert to float for comparison and storage
    grad_norm_float = float(grad_norm)
    
    if grad_norm_float > max_grad_norm:
        gradnorm_queue.add(max_grad_norm)
        #tqdm.write(f'\nClipped gradient with value {grad_norm_float:.1f} '
        #      f'while allowed {max_grad_norm:.1f}')
    else:
        gradnorm_queue.add(grad_norm_float)
        
    return grad_norm



#Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        if len(self.items) == 0:
            return 0.0
        return np.mean(self.items)

    def std(self):
        if len(self.items) <= 1:
            return 0.0
        return np.std(self.items)
    
def get_fully_connected_edges_batch(n_nodes: torch.Tensor):
    """
    Creates a batch of fully-connected edge_index tensors.
    
    Args:
        n_nodes (torch.Tensor): A 1D tensor where n_nodes[i] is the number of nodes in graph i.
    """
    batch_size = len(n_nodes)
    
    offsets = torch.cat([
        torch.tensor([0], device=n_nodes.device), 
        torch.cumsum(n_nodes, dim=0)[:-1]
    ])

    rows, cols = [], []
    for i in range(batch_size):
        num_nodes_in_graph = n_nodes[i]
        offset = offsets[i]
        local_edges = torch.combinations(torch.arange(num_nodes_in_graph), r=2)
        
        local_edges = (torch.cat([local_edges, local_edges.flip(1)], dim=0) + offset).T

        rows.append(local_edges[0])
        cols.append(local_edges[1])

    # Concatenate all edge indices from all graphs
    all_rows = torch.cat(rows, dim=0)
    all_cols = torch.cat(cols, dim=0)
    
    return torch.stack([all_rows, all_cols], dim=0)

fully_conneted_dict = {}

def get_fully_connected_edges(n_nodes):
    if n_nodes in fully_conneted_dict:
        return fully_conneted_dict[n_nodes]
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    fully_conneted_dict[n_nodes] = torch.tensor(edges)
    return fully_conneted_dict[n_nodes]


class CosineScheduler:
    def __init__(self, max_steps, initial_step_size, min_step_size=0.0):
        self.max_steps = max_steps
        self.initial_step_size = initial_step_size
        self.min_step_size = min_step_size

    def get_step_size(self, current_step):
        if current_step < 0:
            raise ValueError("Current step cannot be negative.")
        if current_step > self.max_steps:
            # You might want to keep it at min_step_size or raise an error
            # depending on your specific sampling needs.
            return self.min_step_size

        # Calculate the progress ratio
        progress = current_step / self.max_steps
        
        # Apply the cosine formula
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Scale the step size
        step_size = self.min_step_size + (self.initial_step_size - self.min_step_size) * cosine_decay
        
        return step_size


class ExponentialScheduler:
    def __init__(self, max_steps, initial_step_size, min_step_size=0.0):
        self.max_steps = max_steps
        self.initial_step_size = initial_step_size
        self.min_step_size = min_step_size

    def get_step_size(self, current_step):
        if current_step < 0:
            raise ValueError("Current step cannot be negative.")
        if current_step > self.max_steps:
            return self.min_step_size

        # Calculate the exponential decay factor
        decay_factor = (self.min_step_size / self.initial_step_size) ** (current_step / self.max_steps)
        
        # Calculate the step size
        step_size = self.initial_step_size * decay_factor
        
        return step_size

# code adapted from Equivariant Flow Matching (Klein et al. 2023)
def superpose_points(points, reference):
    """
    Rotates 'points' to optimally match 'reference' (Kabsch algorithm).
    Assumes points are already centered.
    """
    # Compute covariance matrix
    cov = torch.matmul(points.transpose(-2, -1), reference)
    
    # SVD
    # robust=True helps prevent crashes if gradients explode slightly
    try:
        U, S, V = torch.linalg.svd(cov)
    except RuntimeError:
        # Fallback for rare stability issues
        return points 

    # Rotation matrix R = U * V^T
    # Note: In some implementations, V is V.T already. torch.linalg.svd returns Vh (V conjugate transpose).
    # Check your torch version. For torch > 1.8, it returns U, S, Vh.
    R = torch.matmul(U, V) 

    # Handling reflection case (det(R) = -1)
    # In molecular physics, we usually want proper rotation (det=1), 
    # but for general point clouds, reflection might be allowed. 
    # If you strictly forbid reflection (chirality), uncomment below:
    # d = torch.det(R).sign()
    # diag = torch.ones(points.shape[-1], device=points.device)
    # diag[-1] = d
    # R = torch.matmul(U * diag, V)

    return torch.matmul(points, R)

def get_ot_aligned_random_samples(random_molecule, data_molecule):
    '''
    Prepares aligned (x0, x1) pairs for Flow Matching training.
    
    1. x0 is reordered batch-wise to match the closest x1.
    2. x0 particles are permuted to match x1 particles.
    3. x0 is rotated/translated to superpose onto x1.
    
    This minimizes ||x1 - x0||^2, creating the "straightest" flow paths.
    '''
    # Detach from graph to ensure we don't backprop through the matching process
    # (The matching is a fixed target generator, not a differentiable parameter)
    x0 = random_molecule.detach().clone()
    x1 = data_molecule.detach().clone()
    
    batchsize, n_particles, n_dim = x0.shape

    # Center inputs for rotation-invariant matching
    x0_mean = x0.mean(dim=1, keepdim=True)
    x1_mean = x1.mean(dim=1, keepdim=True)
    x0_c = x0 - x0_mean
    x1_c = x1 - x1_mean

    # --- 1. Compute Invariant Cost Matrix ---
    # We compute the cost between every pair in the batch
    # optimizing over Permutation AND Rotation.
    cost_matrix = torch.zeros(batchsize, batchsize, device=x0.device)
    
    # Note: This double loop is the bottleneck. 
    # For very large batches, consider approximating with just particle-permutation 
    # or using Sinkhorn without rotation first.
    for i in range(batchsize):
        # Expand x0[i] to compare against all x1
        p1 = x0_c[i] # (N, 3)
        
        for j in range(batchsize):
            p2 = x1_c[j] # (N, 3)
            
            # Quick particle matching (Procrustes-invariant approx)
            # We just match particles based on distance to allow rotation check
            dists = torch.cdist(p1, p2)
            r_idx, c_idx = linear_sum_assignment(dists.cpu().numpy()**2)
            
            # Align p2 to p1 temporarily to measure fit
            p2_perm = p2[c_idx]
            p2_aligned = superpose_points(p2_perm, p1)
            
            # Squared Euclidean Distance
            cost = (p1 - p2_aligned).pow(2).sum()
            cost_matrix[i, j] = cost

    # --- 2. Batch Assignment (Minibatch OT) ---
    row_idx, col_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
    
    # Map x0 indices to match x1 (sorted by col_idx)
    # if col_idx is [0, 1, 2], row_idx tells us which x0 matches them.
    sorter = np.argsort(col_idx)
    best_x0_indices = row_idx[sorter]
    
    x0_ordered = x0[best_x0_indices]

    # --- 3. Final Fine-Grained Alignment ---
    # Now we have the correct pairs, we perform the final exact transformation
    # to create the training targets.
    final_x0 = torch.zeros_like(x0)
    
    for k in range(batchsize):
        src = x0_ordered[k]
        tgt = x1[k]
        
        # Center
        src_c = src - src.mean(0)
        tgt_c = tgt - tgt.mean(0)
        
        # Particle Permutation (Hungarian)
        d = torch.cdist(src_c, tgt_c)
        r_p, c_p = linear_sum_assignment(d.cpu().numpy()**2)
        
        # Permute src particles to match tgt order
        # We sort by c_p so src rows align with tgt rows 0,1,2...
        p_sort = np.argsort(c_p)
        src_perm = src[r_p[p_sort]]
        
        # Rotation (Kabsch)
        # Rotate src_perm (centered) to match tgt (centered)
        src_rotated = superpose_points(src_perm - src_perm.mean(0), tgt_c)
        
        # Add target mean (so x0 moves to x1's location)
        # In FM, we want x0 and x1 to be in the same spatial frame.
        final_x0[k] = src_rotated + tgt.mean(0)

    return final_x0

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np
from copy import deepcopy

def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


# code from Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation
def get_assignments(src, dst):
    distance_mtx = cdist(src, dst, metric="euclidean")
    _, dest_ind = linear_sum_assignment(distance_mtx, maximize=False)
    distances = distance_mtx[range(len(dest_ind)), dest_ind]
    return distances, dest_ind


def icp(A, B, max_iterations=100, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        R: final Rotation matrix for A
        rotated: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    src = np.copy(A)
    dst = np.copy(B)

    prev_error = 0

    for i in range(max_iterations):
        # get assignments
        distances, indices = get_assignments(src, dst)

        # compute the transformation between the current source and nearest destination points
        _, R, _ = best_fit_transform(src, dst[indices, :])

        # rotate and update the current source
        src = np.dot(R, src.T).T

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    if i > max_iterations - 1:
        print("out of iteration")

    # calculate final transformation
    _, R, _ = best_fit_transform(A, src)
    A_rotated = np.dot(R, A.T).T
    return R, A_rotated, indices

def _rearange_z_optimal_rotation_first_3d(x: np.ndarray, z: np.ndarray, node_mask: np.ndarray):
    """
    x:  [b, n, 3+5]
    z:  [b, n, 3+5]
    node_mask: [b, n, 1]
    """
    ret_z = deepcopy(z)
    length = node_mask.squeeze().sum(axis=-1).astype(np.int32)  # [b]

    for _idx, l in enumerate(length):
        _, z_rotated, _ = icp(z[_idx, :l, :3], x[_idx, :l, :3])
        ret_z[_idx, :l, :3] = z_rotated
    return ret_z

def _rearange_z_first3d(x: np.ndarray, z: np.ndarray, node_mask: np.ndarray):
    """
    x:  [b, n, 3+5]
    z:  [b, n, 3+5]
    node_mask: [b, n, 1]
    """
    ret_z = deepcopy(z)
    length = node_mask.squeeze().sum(axis=-1).astype(np.int32)  # [b]
    distance_matrices = np.sqrt(
        np.sum(
            (
                np.expand_dims(x[:, :, :3], axis=2)
                - np.expand_dims(z[:, :, :3], axis=1)
            )
            ** 2,
            axis=-1,
        )
    )  # [b, n, n]
    for _idx, l in enumerate(length):
        _, col_ind = linear_sum_assignment(
            distance_matrices[_idx, :l, :l], maximize=False
        )
        ret_z[_idx, :l, :] = z[_idx, col_ind, :]
    return ret_z


def get_ot_aligned_random_samples_2(random_molecule_pos, random_molecule_x, data_molecule_pos, data_molecule_x):
    
    _z = torch.cat([random_molecule_pos, random_molecule_x], dim=-1)

    xh = torch.cat([data_molecule_pos, data_molecule_x], dim=-1)
    node_mask = torch.ones((random_molecule_pos.shape[0], random_molecule_pos.shape[1], 1), dtype=torch.float32)
    
    _z = _rearange_z_optimal_rotation_first_3d(
        xh.detach().cpu().numpy(),
        _z.detach().cpu().numpy(),
        node_mask.detach().cpu().numpy(),
    )
    _z = _rearange_z_first3d(
        xh.detach().cpu().numpy(),
        _z,
        node_mask.detach().cpu().numpy(),
    )
    z = torch.tensor(
        _z,
        dtype=xh.dtype,
        device=xh.device,
    )
    return z[:, :, :3], z[:, :, 3:]


def _rearange_z_optimal_rotation_first_3d_list(data_pos, rnd_pos):
    
    rnd_pos = deepcopy(rnd_pos)
    length = len(data_pos)  # [b]
    rnd_pos_rotated = []
    for idx in range(length):
        _, rnd_pos_rotated_single, _ = icp(rnd_pos[idx], data_pos[idx])
        rnd_pos_rotated.append(rnd_pos_rotated_single)
    return rnd_pos_rotated

def _rearange_z_first3d_list(data_pos, rnd_pos):
    
    ret_z = deepcopy(rnd_pos)
    length = len(data_pos)  # [b]
    #distance_matrices = np.sqrt(
    #    np.sum(
    #        (
    #            np.expand_dims(x[:, :, :3], axis=2)
    #            - np.expand_dims(z[:, :, :3], axis=1)
    #        )
    #        ** 2,
    #        axis=-1,
    #    )
    #)  # [b, n, n]
    distance_matrices = []
    for idx in range(length):
        dists = cdist(data_pos[idx], rnd_pos[idx], metric="euclidean")
        distance_matrices.append(dists)

    for _idx in range(length):
        _, col_ind = linear_sum_assignment(
            distance_matrices[_idx], maximize=False
        )
        #ret_z[_idx, :l, :] = z[_idx, col_ind, :]
        ret_z[_idx] = rnd_pos[_idx][col_ind]
    return ret_z


def get_ot_aligned_random_samples_2_list(random_molecule_pos, data_molecule_pos):
    
    device = random_molecule_pos[0].device

    #convert to numpy
    random_molecule_pos = [pos.cpu().numpy() for pos in random_molecule_pos]
    data_molecule_pos = [pos.cpu().numpy() for pos in data_molecule_pos]
    
    _random_molecule_pos = _rearange_z_optimal_rotation_first_3d_list(
        data_molecule_pos,
        random_molecule_pos)
    

    _random_molecule_pos = _rearange_z_first3d_list(
        data_molecule_pos,
        _random_molecule_pos,
        )
    
    # to torch
    _random_molecule_pos = [torch.tensor(pos, dtype=torch.float32, device=device) for pos in _random_molecule_pos]
    return _random_molecule_pos


class FMCollater:
    def __init__(self, generate_random_function, with_data_stds=False, ot=True):
        self.generate_random = generate_random_function
        self.with_data_stds=with_data_stds
        self.ot = ot


    def __call__(self, list_of_data):
        batch = Batch.from_data_list(list_of_data)
        
        n_atom_list = torch.bincount(batch.batch).tolist()
        if self.with_data_stds:
            pca = PCA(n_components=3)
            data_stds_list = []
            rotation_list = []
            for mol in batch.to_data_list():
                pca.fit(mol.pos)
                data_stds_list.append(pca.explained_variance_**0.5)
                rotation_list.append(pca.components_)
        else: 
            data_stds_list = None
            rotation_list = None

        #generate random samples
        random_molecules_list = self.generate_random(batch_size=batch.batch_size, n_atom_list=n_atom_list, data_stds_list=data_stds_list)

        if self.with_data_stds:
            # apply rotation to random samples to be aligned with noise variance axis of data
            for i in range(batch.batch_size):
                random_molecules_list[i].pos = torch.matmul(random_molecules_list[i].pos, torch.tensor(rotation_list[i], device=random_molecules_list[i].pos.device).T)

        random_molecules_batch = Batch.from_data_list(random_molecules_list)

        # ot-align random positions to data positions
        #random_positions_batch = [mol.pos for mol in random_molecules_list]
        #random_positions_batch = torch.stack(random_positions_batch, dim=0)
        random_positions_list = [mol.pos for mol in random_molecules_list]

        #data_positions_batch = data.pos.reshape(data.batch_size, -1, 3)
        data_positions_list = [mol.pos for mol in batch.to_data_list()]
        #noise_positions = get_ot_aligned_random_samples(random_positions_batch, data_positions_batch)
        if self.ot:
            random_positions_list = get_ot_aligned_random_samples_2_list(
                random_positions_list, 
                data_positions_list, 
            )

        random_molecules_batch.pos = torch.cat(random_positions_list, dim=0)

        batch.random_molecule_batch = random_molecules_batch
        
        return batch

class RandomMoleculeGenerator:
    def __init__(self, config):
        """
        Args:
            config: configuration dictionary
        """
        self.config = config
        self.atomtype_distribution = None
        self.position_distribution_std = None
        self.atom_encoding = QM9_ATOM_ENCODING if config["dataset_name"] == "qm9" else GEOM_ATOM_ENCODING
        self.size_prior_per_num_atoms = config.get("size_prior_per_num_atoms", False)
        self.size_prior_per_atom = config.get("size_prior_per_atom", False)
        self.size_prior_scaled_gaussian = config.get("size_prior_scaled_gaussian", False)
        self.std_factor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32) # default to no noise, can be set to dataset stds in set_position_distribution
        self.place_valid_distances = False
        self.min_distance = 0.7
        self.uniform_pos_prior = False

        assert sum([self.size_prior_per_num_atoms, self.size_prior_per_atom, self.size_prior_scaled_gaussian]) <= 1, "Only one of size_prior_per_num_atoms, size_prior_per_atom, size_prior_scaled_gaussian can be True"
        
    def set_atomtype_distribution(self, input, type="categorical"):
        """
        Set the atom type distribution for the sampler.
        Args:
            atomtype_distribution (dict): A dictionary with atom types as keys and their probabilities as values.
        """
        if len(input.shape) == 1:
            #assume input is a vector of probabilities
            probs = input
            
            if type == "categorical":
                assert np.isclose(sum(probs).item(),1.0), "Probabilities must sum to 1"
                self.atomtype_distribution = torch.distributions.Categorical(probs=probs)
            elif type == "dirichlet":
                self.atomtype_distribution = torch.distributions.Dirichlet(probs)
            else:
                raise ValueError("type must be either 'categorical' or 'dirichlet'")
            print(f"Atom type distribution set to {probs}")
        elif len(input.shape) == 3:
            #assume input is a list of atom types
            self.atomtype_distribution = input
            print(f"Atom types are samples from tensor of shape {self.atomtype_distribution.shape}")

    def set_position_distribution(self, dataset):
        """
        Set the position distribution for the sampler.
        Args:
            pos (torch.Tensor): A tensor of shape [num_atoms, 3] representing the positions of the atoms.
        """

        #if isinstance(dataset, dict):
        #    print(f"Position distribution set to size dependent prior list")
        #    self.position_distribution_std = dataset
        #    return
        if self.size_prior_per_num_atoms:
            self.position_distribution_std = dataset.get_size_priors_per_num_atoms()
            print(f"Position distribution set to standard dev shape {self.position_distribution_std}")
        elif self.size_prior_per_atom:
            self.position_distribution_std = dataset.get_size_prior_per_atom()
            print(f"Position distribution set to per data sample standard dev")
        elif self.size_prior_scaled_gaussian:
            self.position_distribution_std = dataset.get_size_priors_scaled_gaussian_fkt()
            print(f"Position distribution set to scaled gaussian standard dev")
        else:
            self.position_distribution_std = dataset.get_size_prior()
            print(f"Position distribution set to standard dev {self.position_distribution_std}")

    def set_num_atoms_distribution(self, dataset, max_n_atoms=None):
        """
        Set the number of atoms distribution for the sampler.
        Args:
            num_atoms_distribution (torch.Tensor): A tensor representing the distribution of the number of atoms.
        """
        slices = dataset.slices["x"]
        slices_offset = slices[1:] - slices[:-1]
        n_atoms = torch.tensor(np.bincount(slices_offset))
        self.num_atoms_distribution = n_atoms/n_atoms.sum()
        self.num_atoms_distribution = self.num_atoms_distribution.to("cpu")

        # clip num_atoms distr
        if max_n_atoms is not None and len(self.num_atoms_distribution) > max_n_atoms:
            print("Clipping number of atoms distribution to max_n_atoms = 70")
            self.num_atoms_distribution = self.num_atoms_distribution[:max_n_atoms]
            self.num_atoms_distribution = self.num_atoms_distribution / self.num_atoms_distribution.sum()
        print(f"Number of atoms distribution set to {self.num_atoms_distribution}")

    def generate_random_molecule(self, batch_size=1, num_atoms=None, n_atom_list=None, data_stds_list=None):
        """
        Generate a naive random molecule with a specified number of atoms and without bonds.


            Returns:
                Instance of pytorch geometric data object representing the molecule. 
        """

        #backwards compatibility
        if not hasattr(self, 'num_atoms_distribution'):
            num_atoms = self.config["num_atoms"]
        
        if n_atom_list is not None:
            assert len(n_atom_list) == batch_size, "n_atom_list must have length equal to batch_size"
            # to tensor if not already
            if not isinstance(n_atom_list, torch.Tensor):
                n_atom_list = torch.tensor(n_atom_list, dtype=torch.long)
            num_atoms = n_atom_list
        elif num_atoms is None:
            num_atoms = torch.multinomial(self.num_atoms_distribution, num_samples=batch_size, replacement=True)
        else:
            num_atoms = torch.tensor([num_atoms] * batch_size, dtype=torch.long)

        data_list = []
        for i in range(batch_size):
            num_atoms_i = num_atoms[i].item()

            #get dataset molecule distribution
            if self.atomtype_distribution is None:
                #get one-hot list
                atoms = [random.choice(list(self.atom_encoding.values())) for i in range(num_atoms_i)]
            elif isinstance(self.atomtype_distribution, torch.distributions.Categorical): 
                #sample from distribution
                atoms = self.atomtype_distribution.sample((num_atoms_i,))
                #one-hot encode atoms
                atoms_one_hot = torch.nn.functional.one_hot(atoms, num_classes=5).float()
            elif isinstance(self.atomtype_distribution, torch.distributions.Dirichlet):
                #sample from distribution
                atoms_one_hot = self.atomtype_distribution.sample((num_atoms_i,))
            elif isinstance(self.atomtype_distribution, torch.Tensor):
                #select random atom types from the tensor
                atoms_one_hot = self.atomtype_distribution[torch.randint(0, self.atomtype_distribution.shape[0], (1,))][0]
            else:
                raise ValueError("atomtype_distribution must be a torch.distributions.Categorical or a list of atom types")

            feats = atoms_one_hot

            if data_stds_list is not None:
                std = data_stds_list[i]
            else:
                if isinstance(self.position_distribution_std, types.LambdaType):
                    std = self.position_distribution_std(num_atoms_i)
                # get std if self.standard_deviation is a dict
                elif isinstance(self.position_distribution_std, dict):
                    stds = self.position_distribution_std[num_atoms_i]

                    if isinstance(stds, list):
                        # randomly sample from list of stds
                        std = torch.tensor(random.choice(stds))
                    else:
                        std = stds
                else:
                    std = self.position_distribution_std

            if self.place_valid_distances:
                positions = []
                for i in range (0, num_atoms_i):
                    valid = False    
                    max_attempts = 100
                    attempts = 0
                    while not valid:
                        new_pos = torch.randn((1, 3)) * std * self.std_factor
                        if i == 0:
                            valid = True
                        else:
                            dists = torch.norm(torch.stack(positions) - new_pos, dim=1)
                            if torch.all(dists > self.min_distance): 
                                valid = True
                        attempts += 1
                        if attempts > max_attempts:
                            print(f"Could not place atom {i} with valid distances after {max_attempts} attempts within a std of {std * self.std_factor}, placing it anyway")
                            valid = True

                    positions.append(new_pos)
                
                positions = torch.cat(positions, dim=0)
            else:
                if self.uniform_pos_prior:
                    positions = (torch.rand((num_atoms_i, 3)) - 0.5) * 2 * std * self.std_factor
                else:
                    #random positions
                    positions = torch.randn((num_atoms_i, 3)) * std * self.std_factor
                
            #subtract mean 
            positions = positions - torch.mean(positions, dim=0, keepdim=True)

            #make a graph with all atoms connected
            edges = get_fully_connected_edges(num_atoms_i)
            edge_attr = torch.ones(edges[0].shape[0], 1)
            
            #create graph
            data_list.append(Data(x=feats, pos=positions, edge_index=edges, edge_attr=edge_attr))

        return data_list

    def __call__(self, *args, **kwds):
        return self.generate_random_molecule(*args, **kwds)


class ShapeEnergy(torch.nn.Module):
    """
    Composable shape-guiding energy that promotes specific molecular geometries.
    
    Computes PCA eigenvalues (l1 >= l2 >= l3) of atom positions per molecule
    and returns an energy that penalizes deviation from the target shape.
    
    Modes:
        'linear':    E = l2 + l3        (push all variance into first axis)
        'planar':    E = l3             (push all variance into first two axes)
        'spherical': E = (l1 - l3)     (promote equal spread in all directions)
        'disk':      E = l3 + (l1 - l2)        (promote variance in two axes, penalize third)
    
    Usage:
        shape_energy = ShapeEnergy(mode='linear')
        # Add to composite list in your model
        model.composites.append(shape_energy)
        model.composite_weights.append(lambda_shape)
    """
    
    def __init__(self, mode='linear', normalize_by_atoms=False, eps=1e-6):
        super().__init__()
        self.mode = mode
        self.normalize_by_atoms = normalize_by_atoms
        self.eps = eps
        
        assert mode in ('linear', 'planar', 'spherical', 'disk'), \
            f"Unknown shape mode '{mode}'. Choose from 'linear', 'planar', 'spherical', 'disk'."
    
    def forward(self, graph):
        """
        Args:
            graph: PyG-style batch with
                - graph.pos:   (total_atoms, 3) atom coordinates
                - graph.batch: (total_atoms,) molecule index per atom
        
        Returns:
            energy: (num_molecules,) per-molecule shape energy
        """
        pos = graph.pos          # (total_atoms, 3)
        batch = graph.batch      # (total_atoms,)
        
        # Compute per-molecule centroids
        centroids = scatter(pos, batch, dim=0, reduce='mean')  # (num_molecules, 3)
        
        # Center atom positions
        centered = pos - centroids[batch]  # (total_atoms, 3)
        
        # Compute covariance matrix per molecule
        # cov_i = (1/N_i) * sum_j (x_j - mu_i)(x_j - mu_i)^T
        # We build this via scatter of outer products
        outer = centered.unsqueeze(-1) * centered.unsqueeze(-2)  # (total_atoms, 3, 3)
        cov_sum = scatter(outer, batch, dim=0, reduce='sum')     # (num_molecules, 3, 3)
        
        counts = scatter(torch.ones(pos.shape[0], device=pos.device), 
                        batch, dim=0, reduce='sum')              # (num_molecules,)
        counts = counts.clamp(min=1).view(-1, 1, 1)
        cov = cov_sum / counts                                   # (num_molecules, 3, 3)
        
        # Symmetrize for numerical stability
        cov = (cov + cov.transpose(-1, -2)) / 2
        
        # Eigendecomposition (eigenvalues returned in ascending order)
        eigenvalues = torch.linalg.eigvalsh(cov)  # (num_molecules, 3), ascending: l3, l2, l1
        
        l3 = eigenvalues[:, 0]  # smallest
        l2 = eigenvalues[:, 1]  # middle  
        l1 = eigenvalues[:, 2]  # largest
        
        # Compute shape energy based on mode
        if self.mode == 'linear':
            energy = l2 + l3
        elif self.mode == 'planar':
            energy = l3
        elif self.mode == 'spherical':
            energy = l1 - l3
        elif self.mode == 'disk':
            energy = l3 + (l1 - l2)

        
        # Optionally normalize by atom count to make energy scale-invariant
        # across different molecule sizes
        if self.normalize_by_atoms:
            counts_flat = counts.view(-1)
            energy = energy / (counts_flat + self.eps)
        
        return energy
