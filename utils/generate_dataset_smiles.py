import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from data.QM9 import QM9
from data.geom import GEOM
from utils import rdkit_functions

dataset_name = "geom"   #qm9, geom

import os
root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(root)
data_path = root + "/data/" + dataset_name
filter_n_atoms = None

def transform_zero_center(data):
    data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
    return data

def transform_add_fully_connected_edges(data):
    from torch_geometric.utils import dense_to_sparse
    num_nodes = data.pos.size(0)
    adj = torch.ones((num_nodes, num_nodes), device=data.pos.device) - torch.eye(num_nodes, device=data.pos.device)
    edge_index, _ = dense_to_sparse(adj)
    data.edge_index = edge_index
    return data

if dataset_name == "geom":
    transform = Compose([transform_zero_center, transform_add_fully_connected_edges])

    qm9 = GEOM(data_path, split="test", filter_n_atoms=filter_n_atoms, transform=transform)
    qm9.shuffle()
    qm9_train = GEOM(data_path, split="train", filter_n_atoms=filter_n_atoms, transform=transform)
elif dataset_name == "qm9":

    def transform_zero_center(data):
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        return data

    qm9 = QM9(data_path, split="test", filter_n_atoms=filter_n_atoms, transform=transform_zero_center)
    qm9.data.cuda()
    qm9.print_summary()
    qm9.shuffle()
    qm9_train = QM9(data_path, split="train", filter_n_atoms=filter_n_atoms, transform=transform_zero_center)
else:
    raise ValueError("Invalid dataset name: {}".format(dataset_name))

print("Converting dataset to position and atom type list...")
# implement multiprocessing for this step
underlying_data = getattr(qm9_train, '_data', None) or qm9_train.data

# 2. Perform operations ONCE on the entire dataset
all_pos = underlying_data.pos.cpu().detach()
all_x_argmax = underlying_data.x.argmax(dim=1).cpu().detach()

# 3. Use PyG's pre-calculated slices to split the tensors back into individual graphs
# 'x' and 'pos' are both node-level attributes, so they share the same slice indices
node_slices = qm9_train.slices['x'] 

# 4. Extract slices into your final list
dataset_graph = [
    (
        all_pos[node_slices[i] : node_slices[i+1]], 
        all_x_argmax[node_slices[i] : node_slices[i+1]]
    )
    for i in range(len(qm9_train))
]


# store smiles in qm9_train
open_babel_args = [True]           # geom: True, qm9: False
with_conformer_args = [True]       # geom: True, qm9: False
limit_bonds_to_one = [False]         # geom: False, qm9: False 
#iter with zip
for open_babel, with_conformer, limit in zip(open_babel_args, with_conformer_args, limit_bonds_to_one):
    print("Processing open_babel: {}, with_conformer: {}, limit_bonds_to_one: {}".format(open_babel, with_conformer, limit))
    basic_metrics = rdkit_functions.BasicMolecularMetrics([], dataset_infos=qm9_train.get_dataset_infos())
    # dataset to position, atom type list
    dataset_smiles = basic_metrics.compute_validity(dataset_graph, open_babel=open_babel, with_conformer=with_conformer, limit_bonds_to_one=limit, verbose=True, multiprocessing=False)[0]
    path = qm9_train.root
    # store as txt
    with open(path + "/train_smiles_openbabel_{}_conformer_{}_limit_{}.txt".format(open_babel, with_conformer, limit), "w") as f:
        for smiles in dataset_smiles:
            f.write(smiles + "\n")

print("Done!")