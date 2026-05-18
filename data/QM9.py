import os
import os.path as osp
import sys
import tarfile
from typing import Callable, List, Optional
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor as tensor
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_bz2,
)
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, scatter

MIN_N_ATOMS = 3
MAX_N_ATOMS = 29

# Check if a string can be converted to an int, without throwing an error.
def is_int(str):
    try:
        int(str)
        return True
    except:
        return False
    
class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    Code adapted from torch_geometric.datasets.QM9
    """  # noqa: E501

    raw_url = ('https://springernature.figshare.com/ndownloader/files/3195389',
               'https://springernature.figshare.com/ndownloader/files/3195404')
    atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    atom_colors = ["white", "black", "blue", "red", "green"]
    atom_decoder = list(atom_encoder.keys())
    atom_radius_pm = [
        37,   # H
        77,   # C
        75,   # N
        73,   # O
        71,   # F
    ]
    charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    top_bond_sym = ['C1H', 'C1C', 'C1O', 'N1C', 'N1H', 'C2O', 'O1H', 'C2C']
    top_angle_sym = ['C1C-C1H', 'C1C-C1C', 'C1C-C1O', 'C1C-C1N', 'C1N-N1C', 'C1O-O1C', 'O1C-C1H', 'C2C-C1C']
    top_dihedral_sym = ['H1C-C1C-C1C', 'C1C-C1C-C1C', 'H1C-C1C-C1H', 'H1C-C1C-C1O', 'C1C-C1C-C1O', 'C1N-N1C-C1C',
                         'H1C-C1N-N1C', 'H1C-C1C-C1N']
    def __init__(   
        self,
        root: str,
        split,
        filter_n_atoms=None,
        transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        
        # add required file path to self.processed_paths
        # If path is not present, it will be created
        self.processed_folder = osp.join(root, 'processed')

        if filter_n_atoms is not None:
            n_atoms, n_atoms_str = self.parse_n_atoms(filter_n_atoms)
            self.n_atoms = n_atoms
            self.processed_folder = osp.join(self.processed_folder, n_atoms_str)
        else:
            self.n_atoms = None

        if split == 'full':
            file_path = self.processed_folder + '/full.pt' 
        elif split == 'train':
            file_path = self.processed_folder + '/train.pt'
        elif split == 'test':
            file_path = self.processed_folder + '/test.pt'
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.required_files = [file_path]
        super().__init__(root, transform, force_reload=force_reload)
        print("Loading QM9 dataset from:", self.processed_folder)
        return self.load(file_path)
    
    def parse_n_atoms(self, n_atoms):
        """n_atoms is either an int, or a tuple marking the range of atoms. Returns string 'startn_endn' """
        if isinstance(n_atoms, int):
            if n_atoms < MIN_N_ATOMS:
                n_atoms = MIN_N_ATOMS
            elif n_atoms > MAX_N_ATOMS:
                n_atoms = MAX_N_ATOMS
            return (n_atoms, n_atoms), f"{n_atoms}_{n_atoms}"
        elif isinstance(n_atoms, tuple) and len(n_atoms) == 2:
            n_atoms_min = max(n_atoms[0], MIN_N_ATOMS)
            n_atoms_max = min(n_atoms[1], MAX_N_ATOMS)
            return (n_atoms_min, n_atoms_max), f"{n_atoms_min}_{n_atoms_max}"
        else:
            raise ValueError(f"Invalid n_atoms: {n_atoms}. Must be int or tuple of length 2.")
        
    def process_xyz_files(self, data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
        """
        Take a set of datafiles and apply a predefined data processing script to each
        one. Data can be stored in a directory, tarfile, or zipfile. An optional
        file extension can be added.

        Parameters
        ----------
        data : str
            Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
        process_file_fn : callable
            Function to process files. Can be defined externally.
            Must input a file, and output a dictionary of properties, each of which
            is a torch.tensor. Dictionary must contain at least three properties:
            {'num_elements', 'charges', 'positions'}
        file_ext : str, optional
            Optionally add a file extension if multiple types of files exist.
        file_idx_list : ?????, optional
            Optionally add a file filter to check a file index is in a
            predefined list, for example, when constructing a train/valid/test split.
        stack : bool, optional
            ?????
        """
        print('Processing data file: {}'.format(data))
        if tarfile.is_tarfile(data):
            tardata = tarfile.open(data, 'r')
            files = tardata.getmembers()

            readfile = lambda data_pt: tardata.extractfile(data_pt)

        elif os.path.isdir(data):
            files = os.listdir(data)
            files = [os.path.join(data, file) for file in files]

            readfile = lambda data_pt: open(data_pt, 'r')

        else:
            raise ValueError('Can only read from directory or tarball archive!')

        # Use only files that end with specified extension.
        if file_ext is not None:
            files = [file for file in files if file.endswith(file_ext)]

        # Use only files that match desired filter.
        if file_idx_list is not None:
            files = [file for idx, file in enumerate(files) if idx in file_idx_list]

        # Now loop over files using readfile function defined above
        # Process each file accordingly using process_file_fn

        molecules = []

        for file in files:
            with readfile(file) as openfile:
                molecules.append(process_file_fn(openfile))

        # Check that all molecules have the same set of items in their dictionary:
        props = molecules[0].keys()
        assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

        # Convert list-of-dicts to dict-of-lists
        molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

        # If stacking is desireable, pad and then stack.
        if stack:
            molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

        return molecules
    
    def gen_splits_gdb9(self):
        """
        Generate GDB9 training/validation/test splits used.

        First, use the file 'uncharacterized.txt' in the GDB9 figshare to find a
        list of excluded molecules.

        Second, create a list of molecule ids, and remove the excluded molecule
        indices.

        Third, assign 100k molecules to the training set, 10% to the test set,
        and the remaining to the validation set.

        Finally, generate torch.tensors which give the molecule ids for each
        set.
        """
        gdb9_txt_excluded = self.raw_dir + '/uncharacterized.txt'

        # First get list of excluded indices
        excluded_strings = []
        with open(gdb9_txt_excluded) as f:
            lines = f.readlines()
            excluded_strings = [line.split()[0]
                                for line in lines if len(line.split()) > 0]

        excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

        assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
            len(excluded_idxs))

        # Now, create a list of indices
        Ngdb9 = 133885
        Nexcluded = 3054

        included_idxs = np.array(
            sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

        # Now generate random permutations to assign molecules to training/validation/test sets.
        Nmols = Ngdb9 - Nexcluded

        Ntrain = 100000
        Ntest = int(0.1*Nmols)
        Nvalid = Nmols - (Ntrain + Ntest)

        # Generate random permutation
        np.random.seed(0)
        data_perm = np.random.permutation(Nmols)

        # Now use the permutations to generate the indices of the dataset splits.
        # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

        train, valid, test, extra = np.split(
            data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

        assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
            len(train), len(valid), len(test), len(extra))

        train = included_idxs[train]
        valid = included_idxs[valid]
        test = included_idxs[test]

        splits = {'train': train, 'val': valid, 'test': test}

        return splits
    
    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        return ['dsgdb9nsd.xyz.tar.bz2', 'uncharacterized.txt']
    
    @property
    def processed_file_names(self) -> List[str]:
        #return os.listdir(self.processed_dir)
        return self.required_files

    def download(self) -> None:
        file_path = download_url(self.raw_url[0], self.raw_dir)
        # rename file to expected name
        os.rename(file_path, osp.join(self.raw_dir, self.raw_file_names[0]))
        
        file_path = download_url(self.raw_url[1], self.raw_dir)
        os.rename(file_path, osp.join(self.raw_dir, self.raw_file_names[1]))


    def get_fully_connected_edges(self, n_nodes):
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)

        edges = [rows, cols]
        return torch.tensor(edges)
    
    def process_xyz_gdb9(self, datafile):
        """
        Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

        Parameters
        ----------
        datafile : python file object
            File object containing the molecular data in the MD17 dataset.

        Returns
        -------
        molecule : dict
            Dictionary containing the molecular properties of the associated file object.

        Notes
        -----
        TODO : Replace breakpoint with a more informative failure?
        """
        xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

        num_atoms = int(xyz_lines[0])
        mol_props = xyz_lines[1].split()
        mol_xyz = xyz_lines[2:num_atoms+2]
        mol_freq = xyz_lines[num_atoms+2]

        atom_charges, atom_positions = [], []
        for line in mol_xyz:
            atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
            atom_charges.append(self.charge_dict[atom])
            atom_positions.append([float(posx), float(posy), float(posz)])

        prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        prop_strings = prop_strings[1:]
        mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
        mol_props = dict(zip(prop_strings, mol_props))
        mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

        molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
        molecule.update(mol_props)
        molecule = {key: torch.tensor(val) for key, val in molecule.items()}

        return molecule
    
    def process(self) -> None:

        # gen splits
        splits_dict = self.gen_splits_gdb9()
        
        # Process GDB9 dataset, and return dictionary of splits
        gdb9_data = {}
        for split, split_idx in splits_dict.items():
            gdb9_data[split] = self.process_xyz_files(self.raw_paths[0], self.process_xyz_gdb9, file_idx_list=split_idx, stack=True)

        
        splits = splits_dict.keys()
        inv_charge_dict = {v: k for k, v in self.charge_dict.items()}

        # iter dict
        data_lists_dict = {"full": []}
        for split, split_data in gdb9_data.items():
            num_molecules = split_data['index'].shape[0]
            data_lists_dict[split] = []
            for i in tqdm(range(num_molecules), desc=f"Processing {split} split"):
                idx = split_data['index'][i].item()
                num_atoms = split_data['num_atoms'][i].item()
                pos = split_data['positions'][i][:num_atoms]
                charges = split_data['charges'][i][:num_atoms]
                # use inverse charge dictionary to get atom types
                atom_types = [inv_charge_dict[charge.item()] for charge in charges]
                encoded_atom_types = [self.atom_encoder[atom] for atom in atom_types]
                x = one_hot(torch.tensor(encoded_atom_types), num_classes=len(self.atom_encoder))
                edge_index = self.get_fully_connected_edges(num_atoms)
                edge_attr = None
                data = Data(
                    x=x,
                    pos=pos,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    idx=idx,
                )

                data_lists_dict[split].append(data)
                data_lists_dict["full"].append(data)


        # Save full dataset
        self.save(data_lists_dict["full"], self.processed_folder + '/full.pt')

        for split in splits:
            self.save(data_lists_dict[split], self.processed_folder + f'/{split}.pt')

    def get_size_prior_per_atom(self):
        '''
        Get size priors for each molecule size in the dataset. 
        Size priors are calculated as the l1, l2, l3 of a molecule in the dataset.
        '''
        # load size priors from file
        store_path = osp.join(self.root, "size_prior_list.pkl")
        if osp.exists(store_path):
            with open(store_path, "rb") as f:
                size_prior_list = pickle.load(f)
            print(f"Loaded size priors from {store_path}")
        else:            
            print(f"Size priors not found at {store_path}, calculating size priors...")
            size_prior_list = self.calc_size_prior_per_atom()
        return size_prior_list


    # calc l1, l2, l3 for each molecule in qm9 and store in dict with molecule size as key
    def calc_size_prior_per_atom(self):
        pca = PCA(n_components=3)
        size_prior_list = {}
        
        for i in tqdm(range(len(self)), desc="Calculating size priors.."):
            data = self[i] 
            num_atoms = data.x.shape[0]
            pos = data.pos.cpu().numpy()
            pca.fit(pos)
            l1, l2, l3 = pca.explained_variance_**0.5
            if num_atoms not in size_prior_list:
                size_prior_list[num_atoms] = []
            size_prior_list[num_atoms].append((l1, l2, l3))

        store_path = osp.join(self.root, "size_prior_list.pkl")
        with open(store_path, "wb") as f:
            pickle.dump(size_prior_list, f)
        print(f"Saved size priors to {store_path}")

        return size_prior_list

    def get_dataset_infos(self):
        dataset_info = {
            'atom_colors': self.atom_colors,
            'atom_radius_pm': self.atom_radius_pm,
            'atom_decoder': self.atom_decoder,
            'name': 'qm9',
            "root_path": self.root,
            "top_bond_sym": self.top_bond_sym,
            "top_angle_sym": self.top_angle_sym,
            "top_dihedral_sym": self.top_dihedral_sym,
        }
        return dataset_info

    @property
    def dataset_infos(self):
        return self.get_dataset_infos()
    
    def get_size_priors(self):
        #return {3: tensor([0.2560, 0.8875, 0.0050]), 4: tensor([0.7071, 0.7723, 0.1017]), 5: tensor([1.1660, 0.6305, 0.1621]), 6: tensor([1.2879, 0.8310, 0.1852]), 7: tensor([0.9704, 1.2780, 0.3599]), 8: tensor([1.0666, 1.4254, 0.2626]), 9: tensor([1.1413, 1.5060, 0.3143]), 10: tensor([1.2436, 1.5530, 0.3263]), 11: tensor([1.2450, 1.6068, 0.4736]), 12: tensor([1.2741, 1.6337, 0.5939]), 13: tensor([1.2386, 1.6360, 0.7857]), 14: tensor([1.2408, 1.6124, 0.9493]), 15: tensor([1.2545, 1.5961, 1.0600]), 16: tensor([1.2666, 1.5793, 1.1444]), 17: tensor([1.2932, 1.5605, 1.1982]), 18: tensor([1.3055, 1.5707, 1.2363]), 19: tensor([1.3198, 1.5834, 1.2752]), 20: tensor([1.3362, 1.6101, 1.3110]), 21: tensor([1.3458, 1.6303, 1.3217]), 22: tensor([1.3657, 1.6827, 1.3790]), 23: tensor([1.3838, 1.6947, 1.3667]), 24: tensor([1.4356, 1.7812, 1.4169]), 25: tensor([1.4280, 1.8044, 1.3880]), 26: tensor([1.4658, 1.8542, 1.4564]), 27: tensor([1.4860, 1.9313, 1.3915]), 29: tensor([1.5536, 2.0404, 1.3657])}
        return {3: tensor([1.109, 0.001, 0.001]), 4: tensor([1.206, 0.505, 0.1  ]), 5: tensor([1.225, 0.668, 0.443]), 6: tensor([1.434, 0.683, 0.35 ]), 7: tensor([1.634, 0.804, 0.31 ]), 8: tensor([1.703, 0.899, 0.293]), 9: tensor([1.71 , 1.038, 0.267]), 10: tensor([1.813, 1.108, 0.311]), 11: tensor([1.876, 1.158, 0.361]), 12: tensor([1.929, 1.203, 0.431]), 13: tensor([1.938, 1.22 , 0.525]), 14: tensor([1.929, 1.232, 0.624]), 15: tensor([1.929, 1.236, 0.707]), 16: tensor([1.924, 1.242, 0.771]), 17: tensor([1.916, 1.244, 0.829]), 18: tensor([1.927, 1.249, 0.862]), 19: tensor([1.951, 1.256, 0.894]), 20: tensor([1.993, 1.267, 0.904]), 21: tensor([1.997, 1.28 , 0.937]), 22: tensor([2.107, 1.282, 0.928]), 23: tensor([2.084, 1.301, 0.96 ]), 24: tensor([2.247, 1.288, 0.946]), 25: tensor([2.209, 1.315, 0.971]), 26: tensor([2.391, 1.288, 0.934]), 27: tensor([2.301, 1.337, 0.992]), 29: tensor([2.447, 1.348, 1.005])}
    
    def get_size_prior(self):
        return tensor([1.963, 1.254, 0.834])
        #return tensor([1.3066, 1.6045, 1.1928])
    
    def get_size_priors_scaled_gaussian_fkt(self):
        return lambda n_atoms: (0.2938 * np.log(n_atoms +1 ) + -0.0479) * self.get_size_prior()
    
if __name__ == '__main__':
    dataset = QM9(root='/home/griesbchr/Repositories/toy_ebm/data/qm9', split='train')

    std_per_molecule= []
    mol_sizes = []
    slices = dataset.slices["x"]
    for i in tqdm(range(len(slices)-1)):
        pos_mol = dataset.pos[slices[i]:slices[i+1]]
        pos_mol_zeromean = pos_mol - pos_mol.mean(dim=0, keepdim=True)
        std_per_molecule.append(pos_mol_zeromean.std(dim=0))
        mol_sizes.append(pos_mol.shape[0])

    # average std over all molecules
    position_distribution_std = torch.mean(torch.stack(std_per_molecule), dim=0)
    position_distribution_std = position_distribution_std.to("cpu")

    print(f"Avergaed position distribution over all molecule sizes: {position_distribution_std}")
    
    std_per_mol = {size: [] for size in set(mol_sizes)}

    for size, std_mol in zip(mol_sizes, std_per_molecule):
        std_per_mol[size].append(std_mol)

    # average std over molecules of same size
    for size in std_per_mol:
        std_per_mol[size] = torch.mean(torch.stack(std_per_mol[size]), dim=0).to("cpu")

    print(f"Averaged position distribution per molecule size: {std_per_mol}")
