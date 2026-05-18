import msgpack
import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse


def extract_conformers(args):
    drugs_file = os.path.join(args.data_dir, args.data_file)
    save_file = f"geom_drugs_{'no_h_' if args.remove_h else ''}{args.conformations}"
    smiles_list_file = 'geom_drugs_smiles.txt'
    number_atoms_file = f"geom_drugs_n_{'no_h_' if args.remove_h else ''}{args.conformations}"

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))

    all_smiles = []
    all_number_atoms = []
    dataset_conformers = []
    energies = []
    mol_id = 0
    for i, drugs_1k in enumerate(unpacker):
        print(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            all_smiles.append(smiles)
            conformers = all_info['conformers']
            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer['totalenergy'])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:args.conformations]
            for id in lowest_energies:
                conformer = conformers[id]
                coords = np.array(conformer['xyz']).astype(float)  # n x 4
                if args.remove_h:
                    mask = coords[:, 0] != 1.0
                    coords = coords[mask]
                n = coords.shape[0]
                all_number_atoms.append(n)
                mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                id_coords = np.hstack((mol_id_arr, coords))

                energies.append(conformer['totalenergy'])
                dataset_conformers.append(id_coords)    
                mol_id += 1

    print("Total number of conformers saved", mol_id)
    all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)

    print("Total number of atoms in the dataset", dataset.shape[0])
    print("Average number of atoms per molecule", dataset.shape[0] / mol_id)

    # Save conformations
    np.save(os.path.join(args.data_dir, save_file), dataset)
    # Save SMILES
    with open(os.path.join(args.data_dir, smiles_list_file), 'w') as f:
        for s in all_smiles:
            f.write(s)
            f.write('\n')

    # save energies
    np.save(os.path.join(args.data_dir, "geom_drugs_energies.npy"), np.array(energies))

    # Save number of atoms per conformation
    np.save(os.path.join(args.data_dir, number_atoms_file), all_number_atoms)
    print("Dataset processed.")


def load_split_data(conformation_file,
                    energies_file,
                    val_proportion=0.1,
                    test_proportion=0.1,
                    filter_size=None):
    from pathlib import Path
    path = Path(conformation_file)
    base_path = path.parent.absolute()

    # base_path = os.path.dirname(conformation_file)
    all_data = np.load(conformation_file)  # 2d array: num_atoms x 5

    energies = np.load(energies_file)  # 1d array: num_conformations


    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)

    # Filter based on molecule size.
    if filter_size is not None:
        # Keep only molecules <= filter_size
        data_list = [
            molecule for molecule in data_list
            if molecule.shape[0] <= filter_size
        ]

        assert len(data_list) > 0, 'No molecules left after filter.'

    # CAREFUL! Only for first time run:
    # perm = np.random.permutation(len(data_list)).astype('int32')
    # print('Warning, currently taking a random permutation for '
    #       'train/val/test partitions, this needs to be fixed for'
    #       'reproducibility.')
    # assert not os.path.exists(os.path.join(base_path, 'geom_permutation.npy'))
    # np.save(os.path.join(base_path, 'geom_permutation.npy'), perm)
    # del perm

    perm = np.load(os.path.join('./data/geom/geom_permutation.npy'))

    # print(perm.shape)
    print("datalen", len(data_list), "perm", perm.shape)
    N = len(data_list)
    
    # data_list = [data_list[i] for i in perm]
    data_list = [data_list[i%N] for i in perm]

    energies = energies[perm]

    num_mol = len(data_list)
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)
    val_data, test_data, train_data = data_list[:val_index],data_list[val_index:test_index], data_list[test_index:]
    val_energies, test_energies, train_energies = energies[:val_index], energies[val_index:test_index], energies[test_index:]
    # np.split(data_list,
                                            #    [val_index, test_index])

    data_tuple = (train_data, val_data, test_data)
    energies_tuple = (train_energies, val_energies, test_energies)
    return data_tuple, energies_tuple


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conformations",
        type=int,
        default=30,
        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--remove_h",
                        action='store_true',
                        help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/path/to/data/geom/raw/')
    parser.add_argument("--data_file", type=str, default="drugs_crude.msgpack")
    args = parser.parse_args()
    extract_conformers(args)
