device_id = 0
import torch

torch.cuda.set_device(device_id)
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

import numpy as np
from functools import partial
import time

from utils import rdkit_functions
from utils import evaluation
from data.QM9 import QM9
from data.geom import GEOM

from ebmol.ebm_utils import get_fully_connected_edges, RandomMoleculeGenerator, ShapeEnergy
from ebmol.deep_ebm import EBM
from ebmol.energy_functions import EGNN_EnergyModel as EnergyModel
from ebmol.samplers.parallel_tempering import ParallelTemperingSampler
from ebmol.samplers.mla import MLASampler 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
root =  os.path.dirname(os.path.abspath(__file__))
experiments_folder = root + "/experiments"
samples_path = root + "/samples/"

data_path = root + "/data/geom"


exp_name = "geom_ls_msp_eregatom"

latest = True
epoch = None
iteration = None

load_samples_name = ""  #eg. geom_ls_msp_eregatom_n10000_ws16_rlx200

# ============ Loading dataset ============

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

transform = Compose([transform_zero_center, transform_add_fully_connected_edges])
filter_n_atoms = None
dataset_test = GEOM(data_path, split="test", filter_n_atoms=filter_n_atoms, transform=transform)
dataset_test.shuffle()
is_rescaled = False
dataset_train = GEOM(data_path, split="train", filter_n_atoms=filter_n_atoms, transform=transform)

n_samples = 50
dataset_iter = iter(dataset_test)
qm9_data_list = [next(dataset_iter) for i in range(n_samples)]


# ============ Loading Model ============
def load_config(exp_name, epoch, iteration, latest=False):
    experiment_name = exp_name
    if latest:
        #find latest epoch in experiments folder
        import os
        model_files = os.listdir(experiments_folder + "/" + experiment_name)
        model_epochs = {}
        for file in model_files:
            if file.startswith("model_") and file.endswith(".pth"):
                parts = file.split("_")
                epoch_part = int(parts[1])
                iter_part =  int(parts[2].split(".")[0])
                if epoch_part not in model_epochs or iter_part > model_epochs[epoch_part]:
                    model_epochs[epoch_part] = iter_part

        latest_epoch = max(model_epochs.keys())
        latest_iter = model_epochs[latest_epoch]        
        checkpoint_path = experiments_folder + "/" + experiment_name + "/model_" + str(latest_epoch) +"_"+str(latest_iter)+ ".pth"
        print("Loading model {} from checkpoint at epoch {} and iter {}".format(exp_name, latest_epoch, latest_iter))

    else:
        assert epoch is not None, "Epoch must be specified if latest is False"
        assert iteration is not None, "Iteration must be specified if latest is False"
        checkpoint_path = experiments_folder + "/" + experiment_name + "/model_" + str(epoch) +"_"+str(iteration)+ ".pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    #load config
    config.verbose = True
    return config, state_dict, checkpoint


config, state_dict, checkpoint = load_config(exp_name, epoch, iteration, latest=latest)
rnd_mol_gen = RandomMoleculeGenerator(config)


energy_fkt = EnergyModel(config, device=device)
# check if keys start with "energy_model." and remove if yes
if all(key.startswith("energy_model.") for key in state_dict.keys()):
    state_dict = {key.replace("energy_model.", ""): value for key, value in state_dict.items()}
energy_fkt.load_state_dict(state_dict)
energy_fkt.to(device)

ebm = EBM(energy_fkt, rnd_mol_gen, config)
ebm.to(device)

eval_ema =  True

if eval_ema and energy_fkt.ema_model is not None:
    print("Using EMA model for evaluation")
    energy_fkt.egnn = energy_fkt.ema_model

# Setup atom type distribution
ebm.generate_random.set_atomtype_distribution(torch.tensor([1/config.n_feat]*config.n_feat, dtype=torch.float32), type="dirichlet")

# Setup prior distribution of atom positions
ebm.generate_random.set_position_distribution(dataset_train)
ebm.generate_random.set_num_atoms_distribution(dataset_train)

ebm.to(device)
ebm.device = device


if load_samples_name is None or load_samples_name == "":
    # ============ Sample molecules ============
    num_samples = 50
    save_appendix = ""
    save_samples = True

    shape_steering = False

    # for sdf conversion
    open_babel = True
    with_conformer = True 
    limit_bonds_to_one = False


    if shape_steering:
        shape_energy = ShapeEnergy(mode='linear')
        shape_energy.to(device)
        ebm.energy_model.composites = [shape_energy]
        ebm.energy_model.composite_weights = [0]
        ebm.energy_model.composition = True

    sampler_config = {
    "sigma_scale_pos": 0.2,
    "sigma_scale_cat": 0.4,
    "phi": 0.0,
    "step_size_pos": 0.05,
    "step_size_x": 0.05,
    "min_cat_x": 0.0005,

    "temp_scales": [1.0, 0.8, 0.5, 0.35, 0.25, 0.2, 0.15, 0.125, 0.1, 0.075, 0.05],       #11 levels
    "replicas_per_temp": 8,  #num parallel chains per temperature
    "withdraw_steps": 16,     #extract sample after every n swaps, has to be >= 2 
    "burn_in_steps": 32,        # 2 * withdraw_steps 
    "swap_steps": 10, #num steps between swaps

    "composition": False,  # whether to use composition of energies or just the EBM energy for sampling
    "composite_weights" : [50],  

    # relaxation sampler parameters
    "relaxation_steps": 200, 
    "rlx_temp": 0.0,
    "rlx_min_cat_x": 0.0,
    "rlx_step_size_pos": 0.01,  
    "rlx_step_size_x": 0.01,    
    "rlx_phi": 0.00,

    "energy_type": "max_node",   # max_node, avg_node
    "replace_runaways": True,
    "runaway_threshold": 10_000,  #threshold for replacing runaway samples

    "reseed": True, # replace sampled data with noise after sampling

    "persistent": True,
    "draw_from_last": True,
    }

    n_atom_list = None

    sampler = MLASampler(energy_fkt, rnd_mol_gen, config=sampler_config)
    pt_sampler = ParallelTemperingSampler(sampler, config=sampler_config)

    start_time = time.time()
    samples_batch, infos = pt_sampler._sample(batch_size=num_samples, verbose=True, n_atom_list=n_atom_list, samples=None)
    runtime = time.time() - start_time


    samples = samples_batch.to_data_list()
    samples_list = samples
    # mean energies and sr per level
    swap_rates_dict = infos['swap_rates_log']
    mean_energies_dict = infos['avg_energy_log']
    mean_energies = []
    for i in range(len(pt_sampler.temp_scales)):
        mean_energy = np.array(mean_energies_dict[i]).mean()
        mean_energies.append(mean_energy)
        print("Level {}: Energy: {:.2f}".format(i, mean_energy))

    for i in range(len(pt_sampler.temp_scales)-1):
        swap_rate = np.array(swap_rates_dict[i]).mean()
        print("Lvl {}-{}: Swap Rate: {:.2f}, Energy diff: {:.2f}".format(i, i+1, swap_rate, mean_energies[i+1] - mean_energies[i]))

    print("Sampled {} samples at {:.2f} seconds/sample".format(len(samples), runtime/len(samples)))


    if save_samples:
        save_name = exp_name + "_n" + str(len(samples)) + "_ws" + str(sampler_config["withdraw_steps"]) + "_rlx" + str(sampler_config["relaxation_steps"])
        save_path = samples_path + save_name + save_appendix + ".xyz"
        rdkit_functions.save_molecules_to_xyz(samples, save_path, dataset_infos=dataset_train.get_dataset_infos())

        # save to sdf
        save_path = samples_path + save_name + save_appendix + ".sdf"
        rdkit_functions.save_molecules_to_sdf(samples, save_path, dataset_infos=dataset_train.get_dataset_infos(), open_babel=open_babel, with_conformer=with_conformer, limit_bonds_to_one=limit_bonds_to_one)

    print("Mean max atom feature of sampled molecules:", round(samples_batch.x.max(dim=-1).values.mean().item(), 2))
    print("Std max atom feature of sampled molecules:", round(samples_batch.x.max(dim=-1).values.std().item(), 2))

    print("")
    print("Average steps per sample: {:.2f}".format(sampler_config["swap_steps"] * len(sampler_config["temp_scales"]) * sampler_config["withdraw_steps"] + sampler_config["relaxation_steps"]))
else:
    # ============ Load samples ============
    load_path = samples_path + load_samples_name + ".xyz"
    samples = rdkit_functions.load_molecules_from_xyz(load_path, dataset_infos=dataset_train.get_dataset_infos())
    samples_batch = Batch.from_data_list(samples)
    print("Loaded {} samples from {}".format(len(samples), load_path))

# ============ Evaluate samples ============
results = evaluation.compute_stability_batch(samples_batch, atom_decoder=dataset_test.atom_decoder, verbose=True, open_babel=True, debug=False)
_ = evaluation.compute_rdkit_metrics_batch(samples_batch, dataset=dataset_train, verbose=True, open_babel=True, with_conformer=True, limit_bonds_to_one=False)
vendi_score, n_samples_vendi = evaluation.compute_vendi_score(samples, dataset_train.get_dataset_infos(), n_samples=1000)
print("Vendi Score ({} samples): {:.2f}".format(n_samples_vendi, vendi_score))