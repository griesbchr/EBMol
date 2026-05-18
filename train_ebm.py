from json.tool import main
import math

import torch
import os
import time
import json

from data.QM9 import QM9
from data.geom import GEOM
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from ebmol.energy_functions import EGNN_EnergyModel
from ebmol.deep_ebm import EBM
from ebmol.ebm_utils import ExponentialScheduler, FMCollater, RandomMoleculeGenerator

import argparse

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='Train EBM')
    parser.add_argument('--dataset_name', type=str, default="geom", help='Dataset name (qm9 or geom)')
    return parser.parse_args()


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

def main(rank, world_size, args, config, distributed_training=False):
    if distributed_training:
        setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')

    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
            print(f"Setting {k} to {v}")

    run_name = f"{time.strftime('%m%d_%H%M')}_{config.dataset_name}_{config.exp_appendix}"
    config.run_name = run_name

    #if hasattr(config, 'use_ema') and config.use_ema and rank != 0:
    #    config.use_ema = False

    if config.exp_appendix == "debug":
        config.enable_comet = False

    if config.loss_type == "flow_matching":
        config.persistent = False   # never use pcd with flow matching

    #create folder 
    os.makedirs(f"{config.root}/experiments/{run_name}", exist_ok=True)
    config.exp_path = f"{config.root}/experiments/{run_name}"

    # save config to path
    config_path = os.path.join(config.exp_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(vars(config), f, indent=4)

    #set up logger
    logger = None
    experiment = None

    if config.dataset_name == "qm9":
        dataset_path = f"{config.root}/data/qm9"
        train_dataset = QM9(dataset_path, split='train', filter_n_atoms=config.num_atoms, transform=transform_zero_center, force_reload=False)
        test_dataset = QM9(dataset_path, split='test', filter_n_atoms=config.num_atoms, transform=transform_zero_center, force_reload=False)
    elif "geom" in config.dataset_name:
        if "small" in config.dataset_name:
            if logger: logger.info("Using GEOM small dataset")
            dataset_path = f"{config.root}/data/geom_small"
        else:
            dataset_path = f"{config.root}/data/geom"
        transform = Compose([transform_zero_center, transform_add_fully_connected_edges])
        train_dataset = GEOM(dataset_path, split='train', filter_n_atoms=config.num_atoms, transform=transform, force_reload=False)
        test_dataset = GEOM(dataset_path, split='val', filter_n_atoms=config.num_atoms, transform=transform, force_reload=False)
    
   
    len_train_dataloader = math.ceil(len(train_dataset) / config.batch_size)
    if distributed_training:
        len_train_dataloader = math.ceil(len_train_dataloader / world_size)

    sampler = DistributedSampler(test_dataset) if distributed_training else None
    shuffle = False if distributed_training else True
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=config.n_eval_samples, 
                                 shuffle=shuffle, sampler=sampler, 
                                 pin_memory=True, num_workers=4, 
                                 persistent_workers=True)
    
    
    rnd_mol_gen = RandomMoleculeGenerator(config)
    rnd_mol_gen.set_atomtype_distribution(torch.tensor([1.0/config.n_feat]*config.n_feat, dtype=torch.float32), type="dirichlet")

    #set num_atoms distribution to same as dataset distribution
    rnd_mol_gen.set_num_atoms_distribution(train_dataset)

    # Setup prior distribution of atom positions
    rnd_mol_gen.set_position_distribution(train_dataset)

    if config.cktpt_path is not None and config.cktpt_path != "":
        checkpoint = torch.load(config.cktpt_path, weights_only=False)
        ckpt_config = checkpoint['config']
        state_dict = checkpoint['state_dict']

        # Setup energy function and model
        energy_model = EGNN_EnergyModel(config=ckpt_config,device=device)
        energy_model.load_state_dict(state_dict)
        energy_model.to(device)

        if distributed_training:
            energy_model = DDP(energy_model, device_ids=[rank], find_unused_parameters=False)
        
        if ckpt_config.n_layers != config.n_layers:
            if logger: logger.warning(f"Checkpoint n_layers {ckpt_config.n_layers} does not match config n_layers {config.n_layers}. Using checkpoint value.")

        ebm = EBM(energy_model, 
                rnd_mol_gen,
                config,
                device=device,
                experiment=experiment)

        # Load EMA if it exists in checkpoint
        if 'ema_state_dict' in checkpoint and config.use_ema:
            ebm.energy_model_module.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            if logger: logger.info("Loaded EMA weights from checkpoint")

        #setup optimizer from state_dict
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(energy_model.parameters(), lr=config.lr, weight_decay=0.05)
            optimizer.load_state_dict(checkpoint['optimizer'])
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(energy_model.parameters(), lr=config.lr)
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            raise ValueError(f"Unknown optimizer {config.optimizer}")

        #set lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.lr

        #setup scheduler from state_dict
        n_steps = len(train_dataset) * config.epochs
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=config.min_lr)

        if config.load_lr_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        start_buffer = checkpoint.get('buffer', None)

        # filter buffer for num_atoms
        if ckpt_config.num_atoms != config.num_atoms:
            initial_len = len(start_buffer) if start_buffer is not None else 0
            if start_buffer is not None:
                start_buffer = [data for data in start_buffer if config.num_atoms[0] <= data.x.shape[0] <= config.num_atoms[1]]
                if logger: logger.info(f"Filtered buffer from {initial_len} to {len(start_buffer)} samples based on num_atoms range {config.num_atoms}")

        if logger: logger.info(f"Loaded checkpoint from {config.cktpt_path} at epoch {start_epoch}, iteration {start_iteration}")

    else:
        # Setup EBM
        energy_model = EGNN_EnergyModel(config=config,device=device)
        energy_model.to(device)
        if distributed_training:
            energy_model = DDP(energy_model, device_ids=[rank], find_unused_parameters=False)
        torch.cuda.synchronize()

        ebm = EBM(energy_model=energy_model,
                rnd_mol_gen=rnd_mol_gen,
                config=config,
                device=device,
                experiment=experiment)
        
        # Set up learning and sampling parameters
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(energy_model.parameters(), lr=config.lr,weight_decay=0.05)
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(energy_model.parameters(), lr=config.lr)
        else:
            raise ValueError(f"Unknown optimizer {config.optimizer}")

        # LR scheduler
        n_steps = len_train_dataloader * config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=config.min_lr)

        start_epoch = 0
        start_iteration = 0
        start_buffer = None

    if hasattr(config, "step_size_pos_max") and hasattr(config, "step_size_pos_min"):
        stepsize_scheduler = ExponentialScheduler(max_steps=config.epochs * len_train_dataloader,
                                            initial_step_size=config.step_size_pos_max,
                                            min_step_size=config.step_size_pos_min)
    else:
        stepsize_scheduler = None



    # Setup OT calculations for flow matching in dataloader for speedup
    if config.loss_type == "flow_matching" or config.loss_type == "energy_matching" or config.loss_type == "restoring_field_matching":
        with_data_stds = config.get('size_prior_per_atom', False)
        collate_fn = FMCollater(generate_random_function=rnd_mol_gen.generate_random_molecule, with_data_stds=with_data_stds, ot=config.get('flow_matching_ot', True))
        # The pytorch geometric DataLoader does not support collate_fn

        if distributed_training:
            sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=True)
            train_dataloader = TorchDataLoader(train_dataset, batch_size=config.batch_size, 
                                        pin_memory=True, num_workers=4, persistent_workers=True,
                                        collate_fn=collate_fn, sampler=sampler, drop_last=True)
        else:
            train_dataloader = TorchDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                        pin_memory=True, num_workers=4, persistent_workers=True,
                                        collate_fn=collate_fn)
    else:
        if distributed_training:
            sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=True)
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                            pin_memory=True, num_workers=4, persistent_workers=True,
                                            sampler=sampler, drop_last=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                            pin_memory=True, num_workers=4, persistent_workers=True)

    # Train the model
    ebm.train(optimizer, train_dataloader, scheduler, config.epochs, config, logger=logger, experiment=None, start_epoch=start_epoch, start_iteration=start_iteration, start_buffer=start_buffer, stepsize_scheduler=stepsize_scheduler, rank=rank, eval_dataloader=test_dataloader)

    if distributed_training:
        cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    args = parse_args()
    if args.dataset_name  == "qm9":
        from configs.qm9_config import config as config
    elif "geom" in args.dataset_name:
        from configs.geom_config import config as config
    else:
        raise ValueError(f"Unknown dataset name {args.dataset_name}")
    config.world_size = world_size
    if config.distributed_training or world_size > 1:
        mp.spawn(main, args=(world_size, args, config, True), nprocs=world_size, join=True)
    else:
        main(0, world_size, args, config, False)