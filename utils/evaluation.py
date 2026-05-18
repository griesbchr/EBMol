import numpy as np
import random
import os
from torch_geometric.data import Batch, Data
import utils.rdkit_functions as rdkit_functions
import utils.bond_analyze as bond_analyze
from openbabel import openbabel

from vendi_score import vendi, data_utils, molecule_utils

def compute_vendi_score(samples, dataset_infos, fp_method="morgan", n_samples=None):

    if n_samples is None:
        n_samples = len(samples)
    
    n_samples  = min(n_samples, len(samples))

    random_inds = random.sample(range(len(samples)), n_samples)
    sel_samples = [samples[i] for i in random_inds]
    rdkit_samples = rdkit_functions.samples_to_rdkit_mols(sel_samples, dataset_infos=dataset_infos, open_babel=True, with_conformer=True)

    fps_samples = molecule_utils.get_tanimoto_K(rdkit_samples, fp=fp_method)
    
    return vendi.score_K(fps_samples), n_samples
def check_stability(positions, atom_type, debug=False, atom_decoder=None, single_bond=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            order = bond_analyze.geom_predictor((atom1, atom2), dist, limit_bonds_to_one=single_bond)

            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for idx, atom_type_i, nr_bonds_i in zip(range(len(atom_type)), atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds for atom idx %d" % (atom_decoder[atom_type_i], nr_bonds_i, idx))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)


# from GEOM-Drugs Revisited: Toward More  Chemically Accurate Benchmarks for 3D  Molecule Generation
geom_drugs_h_tuple_valencies = {
    "Br": {
        0: [(0, 1)],
        1: [(0, 2)],
    },
    "C": {
        0: [(0, 4), (2, 2), (2, 1), (3, 0)],
        -1: [(0, 3), (2, 1), (3, 0)],
        1: [(0, 3), (2, 1), (3, 0)],
    },
    "N": {
        0: [(0, 3), (2, 0), (2, 1), (3, 0)],
        1: [(0, 4), (2, 0), (2, 1), (2, 2), (3, 0)],
        -1: [(0, 2), (2, 0)],
        -2: [(0, 1)],
    },
    "H": {
        0: [(0, 1)],
    },
    "S": {
        0: [(0, 2), (0, 3), (0, 6), (2, 0)],
        1: [(0, 3), (2, 0), (2, 1), (3, 0)],
        2: [(0, 4), (2, 1), (2, 2)],
        3: [(0, 2), (0, 5)],
        -1: [(0, 1)],
    },
    "O": {
        0: [(0, 2), (2, 0)],
        -1: [(0, 1)],
        1: [(0, 3)],
    },
    "F": {
        0: [(0, 1)],
    },
    "Cl": {
        0: [(0, 1)],
        1: [(0, 2)],
    },
    "P": {
        0: [(0, 3), (0, 5)],
        1: [(0, 4)],
    },
    "I": {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [(0, 3)],
    },
    "Si": {
        0: [(0, 4)],
        1: [(0, 5)],
    },
    "B": {
        -1: [(0, 4)],
        0: [(0, 3)],
    },
    "Bi": {
        0: [(0, 3)],
        2: [(0, 5)],
    }
}
def _is_valid_valence_tuple(combo, allowed, charge, element_symbol=None):
    if isinstance(allowed, dict):
        # Fallback: If charge not found, assume 0 or check if list exists
        if charge not in allowed:
            return False
        return _is_valid_valence_tuple(combo, allowed[charge], charge, element_symbol)
    elif isinstance(allowed, (list, set, tuple)):
        return combo in allowed
    return False

def check_stability_openbabel(positions, atom_type, debug=False, atom_decoder=None):
    """
    Checks stability using Open Babel perception instead of manual distance loops.
    """
    if not debug:
        openbabel.obErrorLog.SetOutputLevel(0)
    else:
        openbabel.obErrorLog.SetOutputLevel(1)

    # Standardize input to numpy if it's a tensor
    if hasattr(positions, 'cpu'):
        positions = positions.cpu().numpy()
    if hasattr(atom_type, 'cpu'):
        atom_type = atom_type.cpu().numpy()

    mol = bond_analyze.generate_ob_molecule(positions, atom_type, atom_decoder)

    nr_stable_atoms = 0
    invalid_idxs = []
    #Iterate Atoms and Check Tuple (Aromatic, Regular)
    for atom in openbabel.OBMolAtomIter(mol):
        idx = atom.GetIndex()
        element_symbol = atom_decoder[atom_type[idx]]
        charge = atom.GetFormalCharge()
        
        # Calculate the Tuple: (Num Aromatic Bonds, Sum of Regular Bond Orders)
        n_aromatic_bonds = 0
        regular_valency = 0
        
        for bond in openbabel.OBAtomBondIter(atom):
            if bond.IsAromatic():
                n_aromatic_bonds += 1
            else:
                # GetBondOrder returns 1, 2, 3
                regular_valency += bond.GetBondOrder()
        
        combo = (n_aromatic_bonds, regular_valency)
        
        # Get allowed configs for this element
        allowed_configs = geom_drugs_h_tuple_valencies.get(element_symbol, {})
        
        # Validate
        is_stable = _is_valid_valence_tuple(combo, allowed_configs, charge, element_symbol)

        if not is_stable:
            if debug: 
                print(f"[OB] FAIL {element_symbol} (idx {idx}): Charge {charge}, Tuple {combo} not in allowed.")
            invalid_idxs.append(idx)
        nr_stable_atoms += int(is_stable)

    n_atoms = len(atom_type)
    molecule_stable = (nr_stable_atoms == n_atoms)

    return molecule_stable, nr_stable_atoms, n_atoms, invalid_idxs

def compute_stability_batch(batch, debug=False, atom_decoder=None, verbose=False, open_babel=True, single_bond=False, midi=False):

    #convert to list if its only a data object
    if isinstance(batch, Data) and not isinstance(batch, Batch):
        batch = [batch]

    if isinstance(batch, Batch):
        batch = batch.to_data_list()

    # make sure only open babel or midi is used
    if open_babel and midi:
        raise ValueError("Only one of open_babel or midi can be used at the same time")

    #single bond can only be used without openbabel or midi
    if single_bond and (open_babel or midi):
        raise ValueError("single_bond can only be used when open_babel and midi are both False")

    
    results = []

    stability_results = []
    for i in range(len(batch)):
        position = batch[i].pos.cpu().detach().numpy()
        atom_type = batch[i].x.argmax(dim=1).cpu().detach().numpy()
        if open_babel:
            molecule_stable, nr_stable_bonds, num_atoms, invalid_idxs = check_stability_openbabel(position, atom_type, debug=debug, atom_decoder=atom_decoder)
        else:
            molecule_stable, nr_stable_bonds, num_atoms = check_stability(position, atom_type, debug=debug, atom_decoder=atom_decoder, single_bond=single_bond)
            invalid_idxs = None
        stability_results.append((molecule_stable, nr_stable_bonds, num_atoms))

        results.append({
            'molecule_stable': molecule_stable,
            'nr_stable_bonds': nr_stable_bonds,
            'num_atoms': num_atoms,
            'percentage_stable_bonds': nr_stable_bonds / num_atoms if num_atoms > 0 else 0,
        })

    if debug or verbose:
        print("Percentage of stable molecules: ",
            round(sum([result[0] for result in stability_results]) / len(batch) * 100, 2))
        print('Percentage of stable bonds: ',
            round(sum([result[1] for result in stability_results]) / sum([result[2] for result in stability_results]) * 100, 2))
    
    batch_results = {
        'percentage_stable_molecules': sum([result[0] for result in stability_results]) / len(batch) * 100,
        'percentage_stable_bonds': sum([result[1] for result in stability_results]) / sum([result[2] for result in stability_results]) * 100,
        "invalid_atom_indices": invalid_idxs,
    }

    return results, batch_results


def compute_rdkit_metrics_batch(batch, dataset, debug=False, verbose=False, open_babel=None, with_conformer=False, limit_bonds_to_one=False, catch_errors=True):
    """
    Computes RDKit metrics for a batch of molecules.
    :param batch: Batch of molecules
    :param debug: If True, prints debug information
    :return: List of RDKit metrics for each molecule in the batch
    """

    if isinstance(batch, Data) and not isinstance(batch, Batch):
        batch = [batch]
    if isinstance(batch, Batch):
        batch = batch.to_data_list()

    # Calculate basic molecular metrics
    # first set dataset smiles empty to compute dataset smiles 
    #basic_metrics = rdkit_functions.BasicMolecularMetrics([], atom_decoder= dataset.atom_decoder)

    # dataset to position, atom type list
    #dataset_graph = [(dataset[i].pos.cpu().detach(), dataset[i].x.argmax(dim=1).cpu().detach()) for i in range(len(dataset))]
    #dataset_smiles = basic_metrics.compute_validity(dataset_graph, open_babel=open_babel, with_conformer=with_conformer)[0]

    #basic_metrics.dataset_smiles_list = dataset_smiles#

    # load dataset smiles from txt
    #
    path = dataset.root + '/train_smiles_openbabel_{}_conformer_{}_limit_{}.txt'.format(open_babel, with_conformer, limit_bonds_to_one)
    
    # check if file exists
    if os.path.exists(path):
        with open(path, "r") as f:
            dataset_smiles = [line.strip() for line in f.readlines()]
    else:
        dataset_smiles = None
        print("WARNING: Dataset smiles file not found at {}. Please generate it using utils/generate_geom_smiles.py".format(path))
        print("Continuing without Novelty metric")

    basic_metrics = rdkit_functions.BasicMolecularMetrics(dataset_smiles, dataset_infos=dataset.get_dataset_infos())

    eval_molecules = batch.copy()
    positions = [data.pos.cpu().detach() for data in eval_molecules]
    atom_types = [data.x.argmax(dim=1).cpu().detach() for data in eval_molecules]
    generated = [(positions[i], atom_types[i]) for i in range(len(positions))]
    results = basic_metrics.evaluate(generated,verbose=verbose, open_babel=open_babel, with_conformer=with_conformer, limit_bonds_to_one=limit_bonds_to_one, catch_errors=catch_errors)

    return results
                        