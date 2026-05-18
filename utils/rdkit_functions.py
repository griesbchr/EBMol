### code from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/rdkit_functions.py

import io

from rdkit import Chem
import numpy as np
from rdkit.Geometry import Point3D
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from utils.bond_analyze import get_bond_order, geom_predictor, generate_ob_molecule
import torch
#from configs.datasets_config import get_dataset_info
import pickle
import os
from rdkit import RDLogger
from openbabel import openbabel
from torch_geometric.data import Data, Batch
RDLogger.DisableLog('rdApp.*')


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


bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_smiles_list=None, dataset_infos=None):
        self.atom_decoder = dataset_infos['atom_decoder']
        self.dataset_infos = dataset_infos
        self.dataset_smiles_list = dataset_smiles_list

    def compute_validity(self, generated, open_babel=None, with_conformer=False, limit_bonds_to_one=False, verbose=False, multiprocessing=False, catch_errors=True):
        """ generated: list of couples (positions, atom_types)"""
        if open_babel is None:
            if len(self.atom_decoder) == 16:
                open_babel = True
            elif len(self.atom_decoder) == 5:
                open_babel = False
            else:
                raise NotImplementedError("Atom type dimension is "+str(len(self.atom_decoder))+" but only 5 (QM9) or 17 (GEOM) are supported.")
        
        if multiprocessing:
            # Bind the worker to our new byte-processing method
            worker_func = partial(
                self._process_single_graph_bytes,
                open_babel=open_babel,
                with_conformer=with_conformer,
                limit_bonds_to_one=limit_bonds_to_one
            )
            
            valid = []

            # Helper generator to serialize tensors to bytes on-the-fly
            def yield_as_bytes(graphs):
                for g in graphs:
                    buffer = io.BytesIO()
                    torch.save(g, buffer)
                    yield buffer.getvalue()

            # Wrap your dataset in the byte-converter
            byte_generator = yield_as_bytes(generated)

            # Initialize the Pool 
            with mp.Pool(mp.cpu_count()) as pool:
                # We feed the byte generator to the pool instead of the raw tensors
                results_iterable = pool.imap_unordered(worker_func, byte_generator, chunksize=10)
                
                if verbose:
                    # We still use len(generated) so tqdm knows exactly how many items there are
                    results_iterable = tqdm(results_iterable, total=len(generated), desc="Computing validity")
                    
                for smiles in results_iterable:
                    if smiles is not None:
                        valid.append(smiles)
        else: 
            valid = []
            for graph in tqdm(generated, desc="Computing validity"):
                smiles = self.process_single_graph(
                    graph,
                    open_babel=open_babel,
                    with_conformer=with_conformer,
                    limit_bonds_to_one=limit_bonds_to_one,
                    catch_errors=catch_errors
                )
                if smiles is not None:
                    valid.append(smiles)

        return valid, len(valid) / len(generated), 0

    def _process_single_graph_bytes(self, graph_bytes, open_babel, with_conformer, limit_bonds_to_one):
        """Worker method to deserialize bytes back to tensors and process a single graph."""
        try:
            buffer = io.BytesIO(graph_bytes)
            graph = torch.load(buffer)
            return self.process_single_graph(graph, open_babel, with_conformer, limit_bonds_to_one)
        except Exception as e:
            print(f"Error in worker process: {e}")
            # Return None if deserialization or molecule generation fails
            return None
    
    def process_single_graph(self, graph, open_babel, with_conformer, limit_bonds_to_one, catch_errors=True):
        """Original method for processing a single graph without multiprocessing."""
        try:
            mol = self.build_molecule(
                *graph, 
                open_babel=open_babel, 
                with_conformer=with_conformer, 
                limit_bonds_to_one=limit_bonds_to_one
            )
            smiles = self.mol2smiles(mol, catch_errors=catch_errors)
            
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                return self.mol2smiles(largest_mol, catch_errors=catch_errors)
                
            return None
        except Exception as e:
            # Return None if molecule generation fails or throws an RDKit error
            print(f"Error processing graph: {e}")
            return None
        
    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated, verbose=True, open_babel=None, with_conformer=False, limit_bonds_to_one=False, multiprocessing=False, catch_errors=True):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, avg_num_mol_parts = self.compute_validity(generated, open_babel=open_babel, with_conformer=with_conformer, limit_bonds_to_one=limit_bonds_to_one, multiprocessing=multiprocessing, catch_errors=catch_errors)
        if verbose:
            print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
            # molecule parts
            print(f"Average number of molecular parts per molecule: {avg_num_mol_parts:.2f}")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            if verbose:
                print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")
                print(f"Valid and unique molecules: {len(unique) / len(generated)*100 :.2f}%")
            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                if verbose:
                    print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        return [validity, uniqueness, novelty], unique, avg_num_mol_parts


    def mol2smiles(self, mol, catch_errors=True):
        try:
            result = Chem.SanitizeMol(mol, catchErrors=catch_errors)
        except Exception as e:
            return None
        if result == 0:
            return Chem.MolToSmiles(mol,
                                    kekuleSmiles=True,
                                    isomericSmiles=True,
                                    )
        else:
            return None

    #def mol2smiles(self, mol):
    #    try:
    #        Chem.SanitizeMol(mol)
    #    except ValueError:
    #        return None
    #    return Chem.MolToSmiles(mol)


    def build_molecule(self, positions, atom_types, open_babel=False, with_conformer=False, limit_bonds_to_one=False):
        if open_babel:
            X, A, E = self.build_xae_molecule_openbabel(positions, atom_types)
        else:
            X, A, E = self.build_xae_molecule(positions, atom_types, limit_bonds_to_one=limit_bonds_to_one)
        mol = Chem.RWMol()
        for atom in X:
            a = Chem.Atom(self.atom_decoder[atom.item()])
            mol.AddAtom(a)

        all_bonds = torch.nonzero(A)
        for bond in all_bonds:
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
        if not with_conformer:
            return mol
        
        conf = Chem.Conformer(mol.GetNumAtoms())
        
        # Extract coordinates to a standard Python list (assuming positions is a tensor)
        pos_list = positions.cpu().tolist()
        
        # Set the 3D position for each atom
        for i in range(mol.GetNumAtoms()):
            x, y, z = pos_list[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
            
        # Add the conformer to the RDKit molecule
        mol.AddConformer(conf)
        
        # Force RDKit to perceive wedge/dash stereocenters from the 3D coordinates
        Chem.AssignStereochemistryFrom3D(mol)

        return mol


    def build_xae_molecule(self, positions, atom_types, limit_bonds_to_one=False):
        """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
            args:
            positions: N x 3  (already masked to keep final number nodes)
            atom_types: N
            returns:
            X: N         (int)
            A: N x N     (bool)                  (binary adjacency matrix)
            E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
        """
        n = positions.shape[0]
        X = atom_types
        A = torch.zeros((n, n), dtype=torch.bool)
        E = torch.zeros((n, n), dtype=torch.int)


        pos = positions.unsqueeze(0)
        dists = torch.cdist(pos, pos, p=2).squeeze(0)
        for i in range(n):
            for j in range(i):
                pair = sorted([atom_types[i], atom_types[j]])
                if self.dataset_infos['name'] == 'qm9':
                    order = get_bond_order(self.atom_decoder[pair[0]], self.atom_decoder[pair[1]], dists[i, j])
                elif self.dataset_infos['name'] == 'geom':
                    order = geom_predictor((self.atom_decoder[pair[0]], self.atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=limit_bonds_to_one)
                # TODO: a batched version of get_bond_order to avoid the for loop
                if order > 0:
                    # Warning: the graph should be DIRECTED
                    A[i, j] = 1
                    E[i, j] = order
        return X, A, E



    def build_xae_molecule_openbabel(self, positions, atom_types):
        """ 
        Returns a triplet (X, A, E) using Open Babel for bond perception.
        """
        n = positions.shape[0]
        X = atom_types
        
        # Initialize output tensors
        A = torch.zeros((n, n), dtype=torch.bool)
        E = torch.zeros((n, n), dtype=torch.int)

        mol = generate_ob_molecule(positions, atom_types, self.atom_decoder)

        # 3. Extract Bonds back into A and E tensors
        # Iterate over all bonds detected by Open Babel
        for bond in openbabel.OBMolBondIter(mol):
            # Get 0-based indices of the two atoms
            idx1 = bond.GetBeginAtom().GetIndex()
            idx2 = bond.GetEndAtom().GetIndex()
            
            order = bond.GetBondOrder()
            
            # Match your original logic: Fill lower triangle (row > col)
            # This ensures compatibility with your build_molecule logic
            row, col = max(idx1, idx2), min(idx1, idx2)
            
            A[row, col] = True
            E[row, col] = int(order)

        return X, A, E

def save_molecules_to_xyz(molecules, path, dataset_infos, save_atom_probabilities=True):
    """ molecules: list of pairs (positions, atom_types)"""
    if isinstance(molecules, Data) and not isinstance(molecules, Batch):
        molecules = [molecules]
    if isinstance(molecules, Batch):
        molecules = molecules.to_data_list()

    molecules = [(mol.pos, mol.x) for mol in molecules]
    # to cpu tensors
    molecules = [(pos.cpu(), atom_types.cpu()) for pos, atom_types in molecules]

    # convert one_hot to atom types
    if not save_atom_probabilities:
        molecules = [(pos, torch.argmax(atom_types, dim=-1)) for pos, atom_types in molecules]
    with open(path, 'w') as f:
        for pos, atom_types in molecules:
            f.write(f"{len(pos)}\n\n")  # number of atoms and a blank line
            for i in range(len(pos)):
                write_string = ""
                if not save_atom_probabilities:
                    atom_symbol = dataset_infos["atom_decoder"][atom_types[i].item()]
                    write_string += f"{atom_symbol} "
                else:
                    # save atom type probabilities
                    atom_type_probs = atom_types[i].tolist()
                    write_string += " ".join([f"{prob:.4f}" for prob in atom_type_probs]) + " "
                x, y, z = pos[i].tolist()
                write_string += f"{x:.4f} {y:.4f} {z:.4f}\n"
                f.write(write_string)
    print(f"Saved {len(molecules)} molecules to {path}")

def save_molecules_to_sdf(molecules, path, dataset_infos, open_babel=True, with_conformer=True, limit_bonds_to_one=False):
    """ molecules: list of pairs (positions, atom_types)"""
    if isinstance(molecules, Data) and not isinstance(molecules, Batch):
        molecules = [molecules]
    if isinstance(molecules, Batch):
        molecules = molecules.to_data_list()

    molecules = [(mol.pos, mol.x) for mol in molecules]
    # to cpu tensors
    molecules = [(pos.cpu(), atom_types.cpu()) for pos, atom_types in molecules]

    # convert one_hot to atom types
    molecules = [(pos, torch.argmax(atom_types, dim=-1)) for pos, atom_types in molecules]

    # dtype to int
    molecules = [(pos, atom_types.int()) for pos, atom_types in molecules]
    
    with Chem.SDWriter(path) as writer:
        for pos, atom_types in molecules:
            metric = BasicMolecularMetrics(dataset_infos=dataset_infos)
            mol = metric.build_molecule(pos, atom_types, open_babel=open_babel, with_conformer=with_conformer, limit_bonds_to_one=limit_bonds_to_one)
            writer.write(mol)
    print(f"Saved {len(molecules)} molecules to {path}")

def load_molecules_from_xyz(path, dataset_infos, type="data_list"):
    molecules = []
    with open(path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            num_atoms = int(lines[i].strip())
            i += 2  # skip blank line
            pos = []
            atom_types = []
            for _ in range(num_atoms):
                parts = lines[i].strip().split()
                if len(parts) == 4: # no probabilities, just atom types
                    atom_symbol = parts[0]
                    x, y, z = map(float, parts[1:4])
                    pos.append([x, y, z])
                    atom_type = torch.tensor(dataset_infos["atom_decoder"].index(atom_symbol))
                    atom_type_one_hot = torch.nn.functional.one_hot(atom_type, num_classes=len(dataset_infos["atom_decoder"])).float()
                    atom_types.append(atom_type_one_hot)
                else: # atom type probabilities
                    atom_type_probs = list(map(float, parts[:-3]))
                    x, y, z = map(float, parts[-3:])
                    pos.append([x, y, z])
                    atom_types.append(torch.tensor(atom_type_probs))
                i += 1
            molecules.append((torch.tensor(pos), torch.stack(atom_types)))

    if type == "data_list":
        data_list = []
        for pos, atom_types in molecules:
            data = Data(pos=pos, 
                        x=atom_types.float(),
                        edge_index=get_fully_connected_edges(pos.shape[0]), 
                        edge_attr=None)
            data_list.append(data)
        return data_list
    elif type == "tuple_list":
        return molecules
    else:
        raise ValueError("Invalid type argument: choose 'data_list' or 'tuple_list'")

def xyz_to_sdf(xyz_path, sdf_path, dataset_infos, open_babel=True, with_conformer=True, limit_bonds_to_one=False):
    molecules = load_molecules_from_xyz(xyz_path, dataset_infos, type="tuple_list")

    with Chem.SDWriter(sdf_path) as writer:
        for data in molecules:
            metric = BasicMolecularMetrics(dataset_infos=dataset_infos)
            mol = metric.build_molecule(data[0], data[1], open_babel=open_babel, with_conformer=with_conformer, limit_bonds_to_one=limit_bonds_to_one)
            writer.write(mol)
    
def load_sdf_as_rdkit_mols(sdf_path):
    """ Returns a list of RDKit molecule objects from an SDF file. """
    mols = []
    with Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False) as supplier:
        for mol in supplier:
            if mol is not None:
                mols.append(mol)
    return mols

def samples_to_rdkit_mols(samples, dataset_infos, open_babel=True, with_conformer=True, limit_bonds_to_one=False, verbose=False):
    """ Convert a list of samples (positions, atom_types) to RDKit molecule objects. """
    rdkit_mols = []
    metric = BasicMolecularMetrics(dataset_infos=dataset_infos)
    for graph in tqdm(samples, desc="Converting samples to RDKit molecules", disable=not verbose):
        mol = metric.build_molecule(
            graph.pos.cpu(), 
            torch.argmax(graph.x.cpu(), dim=-1), 
            open_babel=open_babel, 
            with_conformer=with_conformer, 
            limit_bonds_to_one=limit_bonds_to_one
        )
        rdkit_mols.append(mol)
    return rdkit_mols

def rdkit_mols_to_data_list(rdkit_mols, dataset_infos):
    """ Convert a list of RDKit molecule objects to a list of Data samples"""
    data_list = []
    for mol in tqdm(rdkit_mols, desc="Converting RDKit molecules to Data samples"):
        pos = []
        atom_types = []
        for atom in mol.GetAtoms():
            pos.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
            atom_types.append(dataset_infos["atom_decoder"].index(atom.GetSymbol()))
        pos = torch.tensor([[p.x, p.y, p.z] for p in pos], dtype=torch.float)
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        data = Data(pos=pos, 
                    x=torch.nn.functional.one_hot(atom_types, num_classes=len(dataset_infos["atom_decoder"])).float(),
                    edge_index=get_fully_connected_edges(pos.shape[0]), 
                    edge_attr=None)
        data_list.append(data)
    return data_list


if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)
