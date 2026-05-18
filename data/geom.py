import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import pickle
import gc

import torch
from torch import Tensor as tensor
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, scatter
import build_geom_dataset


class GEOM(InMemoryDataset):
    r"""
    Execute instructions from preprocess_geom_edm.py
    Code adapted from torch_geometric.datasets.QM9
    """  # noqa: E501
    atom_encoder = {
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
    atom_colors = [
        'white',       # 0: H
        'beige',       # 1: B
        'black',       # 2: C
        'blue',        # 3: N
        'red',         # 4: O
        'green',       # 5: F
        'pink',        # 6: Al (Other)
        'pink',        # 7: Si (Other)
        'orange',      # 8: P
        'yellow',      # 9: S
        'green',       # 10: Cl
        'pink',        # 11: As (Other)
        'darkred',     # 12: Br
        'darkviolet',  # 13: I
        'beige',       # 14: Hg (Transition Metal)
        'pink'         # 15: Bi (Other)
    ]
    atomic_nb = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]
    atomic_number_tensor = torch.tensor(atomic_nb, dtype=torch.float)
    atom_decoder = list(atom_encoder.keys())

    atom_radius_pm = [
        37,   # H
        82,   # B
        77,   # C
        75,   # N
        73,   # O
        71,   # F
        118,  # Al
        111,  # Si
        106,  # P
        102,  # S
        99,   # Cl
        119,  # As
        114,  # Br
        133,  # I
        132,  # Hg
        148   # Bi
    ]

    top_bond_sym = ['C1H', 'C12C', 'C1C', 'C1N', 'C12N', 'C1O', 'C2O', 'H1N']
    top_angle_sym = ['C12C-C12C', 'C1C-C1H', 'C12C-C1H', 'N1C-C1C', 'C1C-C1C', 'C1C-C12C', 'N1C-C1H', 'C1N-N1C']
    top_dihedral_sym = ['C12C-C12C-C1H', 'C12C-C12C-C12C', 'H1C-C1C-C1C', 'H1C-C1C-C1H', 'C1N-N1C-C1H', 'C1N-N1C-C1C',
                         'H1C-C1C-C12C', 'N1C-C1C-C1H']
    def __init__(   
        self,
        root: str,
        split,
        filter_n_atoms=None,
        transform: Optional[Callable] = None,
        force_reload: bool = False,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        
        # add required file path to self.processed_paths
        # If path is not present, it will be created
        self.processed_folder = osp.join(root, 'processed')
        self.min_n_atoms = 3
        self.max_n_atoms = 181

        if filter_n_atoms is not None:
            n_atoms, n_atoms_str = self.parse_n_atoms(filter_n_atoms)
            self.n_atoms = n_atoms
            self.processed_folder = osp.join(self.processed_folder, n_atoms_str)
        else:
            self.n_atoms = None

        if split == 'train':
            file_path = self.processed_folder + '/train.pt'
        elif split == 'val':
            file_path = self.processed_folder + '/val.pt'
        elif split == 'test':
            file_path = self.processed_folder + '/test.pt'
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.required_files = [file_path]
        super().__init__(root, transform, force_reload=force_reload, pre_filter=pre_filter)
        print("Loading GEOM dataset", split,"split from:", self.processed_folder)
        return self.load(file_path)
    
    def parse_n_atoms(self, n_atoms):
        """n_atoms is either an int, or a tuple marking the range of atoms. Returns string 'startn_endn' """
        if isinstance(n_atoms, int):
            if n_atoms < self.min_n_atoms:
                n_atoms = self.min_n_atoms
            elif n_atoms > self.max_n_atoms:
                n_atoms = self.max_n_atoms
            return (n_atoms, n_atoms), f"{n_atoms}_{n_atoms}"
        elif isinstance(n_atoms, tuple) and len(n_atoms) == 2:
            n_atoms_min = max(n_atoms[0], self.min_n_atoms)
            n_atoms_max = min(n_atoms[1], self.max_n_atoms)
            return (n_atoms_min, n_atoms_max), f"{n_atoms_min}_{n_atoms_max}"
        else:
            raise ValueError(f"Invalid n_atoms: {n_atoms}. Must be int or tuple of length 2.")

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        return ['geom_drugs_30.npy', 'geom_drugs_energies.npy', 'geom_drugs_n_30.npy']
    
    @property
    def processed_file_names(self) -> List[str]:
        #return os.listdir(self.processed_dir)
        return self.required_files

    def get_fully_connected_edges(self, n_nodes):
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)

        edges = [rows, cols]
        return torch.tensor(edges)

    def process(self) -> None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit.Chem.rdchem import HybridizationType
        RDLogger.DisableLog('rdApp.*')  # type: ignore

        splits = ["train", "val", "test"]
        split_indices = [0, 1, 2]

        print("Loading splits")
        data_tuple, energies_tuple = build_geom_dataset.load_split_data(self.raw_paths[0],
                                                        self.raw_paths[1],
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=None)
        print("Finished loading splits")
        
        data_list = []
        idx = 0
        for split_idx, conformer_list in zip(split_indices, data_tuple):
            split = splits[split_idx]
            energies = energies_tuple[split_idx]

            # iterate conformers
            for i, conf in enumerate(tqdm(conformer_list, desc=f"Processing {split} split")):
                
                pos = torch.from_numpy(conf[:, -3:]).float()
                atom_types = torch.from_numpy(conf[:, 0].astype(int))[:, None]
                x = (atom_types == self.atomic_number_tensor).float()
                energy = torch.tensor(energies[i], dtype=torch.float32)
                N = pos.size(0)
                if self.n_atoms is not None:
                    if not (self.n_atoms[0] <= N <= self.n_atoms[1]):
                        continue

                data = Data(
                    x=x,
                    pos=pos,
                    idx=idx,
                    energy=energy
                )
                idx += 1
                data_list.append(data)
            # this one needs about 80GB of RAM for the training split
            self.save(data_list, self.processed_folder + f'/{split}.pt')
            print("Done processing split:", split)
            data_list = []


    def get_dataset_infos(self):
        dataset_info = {
            'atom_colors': self.atom_colors,
            'atom_radius_pm': self.atom_radius_pm,
            'atom_decoder': self.atom_decoder,
            'name': 'geom',
            "root_path": self.root,
            "top_bond_sym": self.top_bond_sym,
            "top_angle_sym": self.top_angle_sym,
            "top_dihedral_sym": self.top_dihedral_sym
        }
        return dataset_info
        
    @property
    def dataset_infos(self):
        return self.get_dataset_infos()
    

    def get_size_prior(self):
        return tensor([3.508, 1.994, 1.271])
    
    def get_size_priors_per_num_atoms(self):
        return {3: tensor([0.772, 0.329, 0.001]), 4: tensor([1.049, 0.657, 0.106]), 5: tensor([1.019, 0.771, 0.47 ]), 6: tensor([1.436, 1.155, 0.001   ]), 7: tensor([1.339, 0.804, 0.675]), 8: tensor([1.416, 0.945, 0.568]), 9: tensor([1.472, 0.982, 0.512]), 10: tensor([1.604, 0.962, 0.62 ]), 11: tensor([1.608, 1.011, 0.592]), 12: tensor([1.72 , 1.059, 0.662]), 13: tensor([1.734, 1.089, 0.664]), 14: tensor([1.767, 1.167, 0.694]), 15: tensor([1.923, 1.191, 0.706]), 16: tensor([2.025, 1.224, 0.722]), 17: tensor([2.125, 1.244, 0.725]), 18: tensor([2.325, 1.323, 0.677]), 19: tensor([2.353, 1.343, 0.718]), 20: tensor([2.343, 1.359, 0.771]), 21: tensor([2.619, 1.399, 0.726]), 22: tensor([2.665, 1.422, 0.766]), 23: tensor([2.766, 1.455, 0.783]), 24: tensor([2.827, 1.465, 0.823]), 25: tensor([2.936, 1.489, 0.841]), 26: tensor([2.996, 1.516, 0.872]), 27: tensor([3.059, 1.551, 0.89 ]), 28: tensor([3.101, 1.571, 0.929]), 29: tensor([3.153, 1.603, 0.941]), 30: tensor([3.191, 1.622, 0.97 ]), 31: tensor([3.211, 1.66 , 0.996]), 32: tensor([3.257, 1.678, 1.016]), 33: tensor([3.284, 1.71 , 1.036]), 34: tensor([3.303, 1.738, 1.06 ]), 35: tensor([3.343, 1.768, 1.073]), 36: tensor([3.368, 1.791, 1.095]), 37: tensor([3.389, 1.817, 1.112]), 38: tensor([3.406, 1.845, 1.128]), 39: tensor([3.423, 1.865, 1.15 ]), 40: tensor([3.439, 1.892, 1.169]), 41: tensor([3.463, 1.906, 1.184]), 42: tensor([3.474, 1.929, 1.203]), 43: tensor([3.493, 1.947, 1.218]), 44: tensor([3.51 , 1.966, 1.235]), 45: tensor([3.527, 1.989, 1.248]), 46: tensor([3.545, 2.005, 1.265]), 47: tensor([3.564, 2.016, 1.279]), 48: tensor([3.567, 2.032, 1.297]), 49: tensor([3.568, 2.049, 1.31 ]), 50: tensor([3.571, 2.065, 1.329]), 51: tensor([3.585, 2.082, 1.343]), 52: tensor([3.593, 2.092, 1.352]), 53: tensor([3.588, 2.109, 1.37 ]), 54: tensor([3.604, 2.122, 1.383]), 55: tensor([3.608, 2.136, 1.394]), 56: tensor([3.619, 2.145, 1.413]), 57: tensor([3.633, 2.157, 1.419]), 58: tensor([3.625, 2.176, 1.434]), 59: tensor([3.653, 2.187, 1.446]), 60: tensor([3.676, 2.191, 1.456]), 61: tensor([3.64 , 2.209, 1.477]), 62: tensor([3.675, 2.22 , 1.482]), 63: tensor([3.698, 2.24 , 1.493]), 64: tensor([3.687, 2.256, 1.511]), 65: tensor([3.652, 2.265, 1.531]), 66: tensor([3.739, 2.288, 1.528]), 67: tensor([3.708, 2.294, 1.549]), 68: tensor([3.699, 2.32 , 1.561]), 69: tensor([3.711, 2.335, 1.578]), 70: tensor([3.688, 2.358, 1.596]), 71: tensor([3.666, 2.375, 1.631]), 72: tensor([3.805, 2.345, 1.6  ]), 73: tensor([3.728, 2.377, 1.61 ]), 74: tensor([3.75, 2.39, 1.64]), 75: tensor([3.783, 2.452, 1.617]), 76: tensor([3.822, 2.414, 1.621]), 77: tensor([3.794, 2.429, 1.649]), 78: tensor([3.903, 2.392, 1.633]), 79: tensor([3.818, 2.402, 1.665]), 80: tensor([3.951, 2.407, 1.674]), 81: tensor([3.918, 2.461, 1.618]), 82: tensor([3.892, 2.453, 1.639]), 83: tensor([3.924, 2.437, 1.699]), 84: tensor([3.938, 2.509, 1.66 ]), 85: tensor([3.964, 2.52 , 1.677]), 86: tensor([4.001, 2.444, 1.723]), 87: tensor([3.903, 2.519, 1.716]), 88: tensor([4.054, 2.606, 1.738]), 89: tensor([3.959, 2.536, 1.82 ]), 90: tensor([3.65 , 2.458, 1.849]), 91: tensor([3.817, 2.463, 1.837]), 92: tensor([3.765, 2.424, 1.913]), 93: tensor([3.905, 2.735, 1.749]), 94: tensor([4.246, 2.603, 1.768]), 95: tensor([4.501, 2.696, 1.806]), 96: tensor([6.393, 2.393, 1.719]), 97: tensor([3.712, 2.625, 1.945]), 98: tensor([5.014, 2.458, 1.803]), 99: tensor([4.446, 2.617, 1.848]), 100: tensor([4.308, 2.536, 1.823]), 101: tensor([4.155, 2.586, 1.899]), 102: tensor([5.279, 2.287, 1.883]), 103: tensor([3.595, 2.609, 2.067]), 104: tensor([4.863, 2.6  , 1.847]), 105: tensor([3.937, 2.54 , 2.008]), 106: tensor([4.802, 2.563, 1.973]), 107: tensor([4.259, 2.539, 1.962]), 108: tensor([3.904, 2.678, 2.131]), 109: tensor([3.361, 2.773, 2.134]), 110: tensor([3.672, 2.79 , 2.066]), 111: tensor([3.911, 2.663, 2.037]), 112: tensor([3.851, 2.525, 2.086]), 113: tensor([4.146, 2.826, 2.073]), 114: tensor([3.776, 2.725, 2.254]), 115: tensor([3.877, 2.766, 2.105]), 116: tensor([4.096, 2.588, 2.059]), 117: tensor([4.186, 2.563, 2.196]), 118: tensor([4.256, 2.842, 1.926]), 119: tensor([6.261, 2.697, 1.699]), 120: tensor([4.997, 2.825, 1.796]), 121: tensor([3.793, 2.752, 2.051]), 122: tensor([3.719, 3.048, 1.801]), 123: tensor([4.471, 2.846, 2.004]), 124: tensor([3.745, 2.855, 2.091]), 125: tensor([4.117, 2.852, 2.057]), 126: tensor([3.774, 2.893, 2.066]), 127: tensor([4.714, 2.62 , 2.074]), 128: tensor([4.524, 2.763, 2.086]), 129: tensor([4.119, 2.852, 2.353]), 130: tensor([4.472, 2.67 , 2.02 ]), 131: tensor([5.119, 2.419, 2.163]), 132: tensor([4.712, 2.857, 2.28 ]), 133: tensor([3.891, 2.951, 2.239]), 134: tensor([4.416, 2.821, 2.071]), 135: tensor([4.13 , 2.862, 2.033]), 136: tensor([4.101, 2.889, 2.208]), 137: tensor([3.653, 2.966, 2.389]), 138: tensor([4.786, 2.815, 2.032]), 139: tensor([4.334, 2.884, 2.283]), 140: tensor([4.229, 2.978, 2.18 ]), 141: tensor([4.474, 2.847, 2.236]), 142: tensor([4.406, 2.734, 2.127]), 143: tensor([4.164, 2.941, 2.32 ]), 144: tensor([4.498, 2.79 , 2.272]), 145: tensor([4.402, 3.049, 1.837]), 146: tensor([4.342, 2.952, 2.138]), 147: tensor([4.231, 2.991, 2.23 ]), 148: tensor([5.159, 2.585, 2.171]), 149: tensor([4.165, 3.279, 2.137]), 151: tensor([3.453, 3.143, 2.482]), 152: tensor([3.692, 3.115, 2.48 ]), 155: tensor([6.445, 2.973, 2.145]), 159: tensor([4.231, 3.195, 2.242]), 160: tensor([3.96 , 3.449, 2.273]), 165: tensor([6.908, 2.446, 1.845]), 171: tensor([3.934, 3.38 , 2.5  ]), 175: tensor([4.501, 3.161, 2.154]), 176: tensor([4.367, 3.078, 2.668]), 181: tensor([5.116, 3.063, 2.209])}
    
    def get_size_priors_scaled_gaussian_fkt(self):
        return lambda n_atoms: (0.3549*np.log(n_atoms +1 ) + -0.4985)*self.get_size_prior()

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
        
        for i in range(len(self)):
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
    
if __name__ == '__main__':
    dataset = GEOM(root='/path/to/data/geom', split='train')

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
