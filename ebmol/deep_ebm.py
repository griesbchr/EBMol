from math import dist

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
from functools import partial
from torch.nn.parallel.distributed import DistributedDataParallel
from torch_geometric.data import Data, Batch
import torch.distributed as dist

from ebmol.ebm_utils import gradient_clipping, Queue
from ebmol.samplers.mla import MLASampler
from ebmol.samplers.parallel_tempering import ParallelTemperingSampler
import utils.evaluation as evaluation


class EBM(nn.Module):
    def __init__(self, energy_model, rnd_mol_gen, config, device="cpu", experiment=None):
        super(EBM, self).__init__()
        if isinstance(energy_model, DistributedDataParallel):
            self.energy_model = energy_model          # The Wrapper (Use for forward/backward)
            self.energy_model_module = energy_model.module # The Inner Model (Use for EMA/Attribute access)
            self.distributed_training = True
        else:
            self.energy_model = energy_model
            self.energy_model_module = energy_model
            self.distributed_training = False
        
        self.config = config

        self.composition = False
        self.composites = []
        self.composite_weights = []

        self.generate_random = rnd_mol_gen
        self.device = device
   
        if self.config["loss_type"] == "flow_matching":
            self.loss_function = self.flow_matching_loss
        elif self.config["loss_type"] == "restoring_field_matching":
            self.loss_function = partial(self.flow_matching_loss, extrapolation=True)
        else:
            raise ValueError(f"Loss type {self.config['loss_type']} not supported")
        
        self.setup_sampler(config, self.energy_model_module, experiment)

        self.to(device)
        
    def setup_sampler(self, config, energy_model, experiment=None):
        #setup sampler
        self.sampler = MLASampler(energy_model, 
                                    self.generate_random, 
                                    step_size_pos=config.step_size_pos, 
                                    step_size_x=config.step_size_x, 
                                    phi=config.phi,
                                    batch_size=config.batch_size,
                                    verbose=config.verbose,
                                    experiment=experiment,
                                    config=config)
    
        if config.parallel_tempering:
            self.sampler = ParallelTemperingSampler(self.sampler, config)

    #returns unnormalized log probability of EBM
    def forward(self, graph):
        return self.energy_model(graph)
    
    def get_energy(self, graph):
        return -self.energy_model(graph)
   

    def interpolate_data(self, data1, data2, t_pos, t_cat):
        """
        Interpolate between two data objects.
        Args:
            data1 (Data): First data object.
            data2 (Data): Second data object.
            t (float): Interpolation factor between 0 and 1.
        Returns:
            Data: Interpolated data object.
        """
        t_pos_batched = t_pos[data1.batch].unsqueeze(-1)
        t_cat_batched = t_cat[data1.batch].unsqueeze(-1)
        interp_x = data1.x + t_cat_batched * (data2.x - data1.x)
        interp_pos = data1.pos + t_pos_batched * (data2.pos - data1.pos)

        interp_data = data1.clone()
        interp_data.x = interp_x
        interp_data.pos = interp_pos
        return interp_data
        
    def flow_matching_loss(self, data, **kwargs):

        extrapolation = kwargs.get('extrapolation', False)
   
        random_molecules_batch = data.random_molecule_batch

        #interpolate between random and data
        if extrapolation:
            t_pos = torch.rand(data.batch_size, device=data.pos.device) * 2  
            t_cat = 1-torch.abs(t_pos - 1)  
            interp_data = self.interpolate_data(random_molecules_batch, data, t_pos, t_cat)   #extrapolation t_pos
        else:
            t_pos = torch.rand(data.batch_size, device=data.pos.device)  
            t_cat = t_pos  #use same t for categorical and positional features
            interp_data = self.interpolate_data(random_molecules_batch, data, t_pos, t_cat)


        #compute gradients from energy 
        self.energy_model.train()
        interp_data.pos.requires_grad_(True)
        interp_data.x.requires_grad_(True)
        inputs = [interp_data.pos, interp_data.x]
        energy = self.energy_model(interp_data)

        grads = torch.autograd.grad(
            outputs=energy.squeeze(-1), 
            inputs=inputs,
            grad_outputs=torch.ones_like(energy.squeeze(-1)),
            create_graph=True)
        
        vel_pos = -grads[0].contiguous()  
        vel_x = -grads[1].contiguous()    

        # compute velocity targe and calculate loss        
        target_v_pos = (data.pos - random_molecules_batch.pos)
        target_v_x = (data.x - random_molecules_batch.x)
        
        if extrapolation:
            if self.config.get('loss_smoothing'):
                # smooth scaling
                gamma = self.config.get('sharpness_gamma')
                time_diff = (1.0 - t_pos[data.batch])
                scale_pos = torch.tanh(gamma * time_diff).unsqueeze(-1)
                target_v_pos = target_v_pos * scale_pos

                scale_cat = torch.tanh(gamma * (1.0 - t_cat[data.batch])).unsqueeze(-1)
                target_v_x = target_v_x * scale_cat
            else:
                # invert target pos velocities for t_pos > 1
                mask = (t_pos[data.batch] > 1.0).unsqueeze(-1)
                target_v_pos = torch.where(mask, -target_v_pos, target_v_pos)
            
        loss_pos = torch.mean((vel_pos - target_v_pos) ** 2)      #-grads because we use energy, not log prob
        loss_x = torch.mean((vel_x - target_v_x) ** 2)
        
        if self.config.get("energy_regularization") and self.config.get('energy_regularization_weight', 0.0) > 0:
            with torch.no_grad():
                clean_graph = data.clone()  # your x_0 before interpolation
            energy_at_data, node_energies_at_data, _ = self.energy_model(clean_graph, return_node_energy=True)
            energy_reg_loss = self.config.energy_regularization_weight * torch.mean(node_energies_at_data ** 2)
        else:
            energy_reg_loss = torch.tensor(0.0, device=loss_pos.device)

        loss = loss_pos + loss_x + energy_reg_loss

        log_dict = {'loss': loss.item(),
                    'loss_pos': loss_pos.item(),
                    'loss_x': loss_x.item(),
                    'energy_reg_loss': energy_reg_loss.item()
                    }
        
        return loss, log_dict


    def train(self, optimizer, dataloader, scheduler, epochs, config, logger=None, experiment=None, start_epoch=0, start_iteration=0, start_buffer=None, stepsize_scheduler=None, rank=0, eval_dataloader=None):
        if eval_dataloader is None:
            eval_dataloader = dataloader
        
        if rank == 0: loss_list = []

        #loss_queue = deque(maxlen=100)  # Automatically drops oldest when full
        gradnorm_queue = Queue(max_len=100)
        gradnorm_queue.add(3000)  # Add large value that will be flushed.
        
        iteration = 0
        for epoch in tqdm(range(start_epoch, epochs+1), total=epochs, initial=start_epoch):       #+1 to stop at epoch n not n-1
            if logger: logger.info(f"### {config.run_name} Epoch: {epoch}###")
            if hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            if experiment: experiment.set_epoch(epoch)
            
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for i, data in pbar:
                
                data = data.to(self.device)
                if hasattr(data, 'random_molecule_batch'):
                    data.random_molecule_batch = data.random_molecule_batch.to(self.device)
                # Skip iterations if resuming in the middle of an epoch
                if epoch == start_epoch and i < start_iteration:
                    pbar.update(1) # Manually update tqdm progress for skipped iteration
                    continue
                
                if experiment: experiment.set_step(i + epoch * len(dataloader))  # Set step for comet_ml experiment
                        
                optimizer.zero_grad()

                loss, log_dict = self.loss_function(data, pbar=None)

                # Backprop
                loss.backward()
                
                grad_norm = gradient_clipping(self.energy_model, gradnorm_queue)
                optimizer.step()
                
                #del loss, data
                scheduler.step()

                if((iteration % self.config.eval_interval) == 0 and i > 0):# or i == len(dataloader) - 1:       
                    torch.cuda.empty_cache()
                    self.eval(eval_dataloader, epoch, i ,config, logger, experiment, optimizer, scheduler, rank)
                    torch.cuda.empty_cache()

                if config.use_ema:
                    self.energy_model_module.update_ema()

                if rank == 0:
                    #log_dict['grad_norm'] = grad_norm.item()
                    loss_list.append(log_dict)

                    if (iteration % self.config.log_interval) == 0:
                        # print name of experiment
                        print(f"******** {config.run_name} ********")
                        keys = loss_list[0].keys()
                        avg_loss = {k: np.mean([d[k] for d in loss_list]) for k in keys}
                        avg_loss_string = ' | '.join([f"{k}: {v:.3f}" for k, v in avg_loss.items()])
                        msg = f"Ep {epoch}, Iter{i}: {avg_loss_string}"
                        if logger: logger.info(msg)
                        print(msg)

                        loss_list = []

                    if((iteration % self.config.save_interval) == 0 and i > 0):# or i == len(dataloader) - 1:       
                        self.save(eval_dataloader, epoch, i ,config, logger, experiment, optimizer, scheduler)

                iteration = iteration + 1

        # always save at the end of training
        self.save_and_eval(eval_dataloader, epoch, i ,config, logger, experiment, optimizer, scheduler, rank)
        return

    def save(self, eval_dataloader, epoch, i, config, logger, experiment, optimizer, scheduler):
        state_dict = self.energy_model_module.state_dict()
        save_dict = {"state_dict": state_dict,
            "config": config,
            "epoch": epoch,
            "iteration": i,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()}
        
        # Save EMA weights if using EMA
        if config.use_ema and hasattr(self.energy_model_module, 'ema_model') and self.energy_model_module.ema_model is not None:
            save_dict["ema_state_dict"] = self.energy_model_module.ema_model.state_dict()
        
        # Save buffer if exists
        if hasattr(self, "sampler") and hasattr(self.sampler, "buffer"):
            save_dict["buffer"] = self.sampler.buffer
            
        torch.save(save_dict, f"{config.exp_path}/model_{epoch}_{i}.pth")
        print('Model saved')

    def save_and_eval(self, eval_dataloader, epoch, i, config, logger, experiment, optimizer, scheduler, rank=0):
        if rank == 0:
            self.save(eval_dataloader, epoch, i ,config, logger, experiment, optimizer, scheduler)
        self.eval(eval_dataloader, epoch, i ,config, logger, experiment, optimizer, scheduler, rank)
    
    def eval(self, eval_dataloader, epoch, i, config, logger, experiment, optimizer, scheduler, rank=0):
        energy_model_inner = self.energy_model_module 
        ema_model = energy_model_inner.ema_model if hasattr(energy_model_inner, 'ema_model') else None

        if config.eval_ema and config.use_ema and ema_model is not None:
            if logger: logger.info("Evaluating with EMA parameters")
            # store current model parameters
            current_state_dict = copy.deepcopy(energy_model_inner.egnn.state_dict())
            # load ema parameters
            energy_model_inner.egnn.load_state_dict(ema_model.state_dict())

        # generate samples using multiple gpus if distributed training
        if dist.is_initialized():
            samples_list = []
            # Calculate local batch size
            world_size = dist.get_world_size()
            n_samples_per_gpu = config.n_eval_samples//world_size
            
            print(f"Sampling eval batch with {n_samples_per_gpu} samples on rank {rank}")
            samples_batch_, infos = self.sampler._sample(batch_size=n_samples_per_gpu, sample_ts=None, verbose=True)
            samples_batch_ = samples_batch_.detach()
            print(f"Rank {rank} finished sampling.")
            local_samples = samples_batch_.to_data_list()
            local_samples = [s.cpu() for s in local_samples]

            # gather samples from all ranks
            all_samples_list = [None for _ in range(world_size)]
            dist.all_gather_object(all_samples_list, local_samples)
            
            samples_list = [item for sublist in all_samples_list for item in sublist]
        else:
            samples_list = []
            n_eval_samples = config.n_eval_samples
            print(f"Sampling eval batch with {n_eval_samples} samples on rank {rank}")
            samples_batch_, infos = self.sampler._sample(batch_size=n_eval_samples, sample_ts=None, verbose=True)
            samples_batch_ = samples_batch_.detach()
            print(f"Rank {rank} finished sampling.")
            samples_list.extend(samples_batch_.to_data_list())

        if rank == 0:
            try:
                self.evaluate_epoch(eval_dataloader, samples_list, epoch, i, config, logger, experiment)
            except Exception as e:
                print(f"Error occurred during evaluation: {e}")
                import traceback
                traceback.print_exc()
            
        # Restore original weights if EMA was used
        if config.eval_ema and config.use_ema and ema_model is not None:
            energy_model_inner.egnn.load_state_dict(current_state_dict)
        
        if dist.is_initialized():
            dist.barrier()

    def evaluate_epoch(self, dataloader, samples_list, epoch, iteration, config, logger=None, experiment=None):
        samples_batch = Batch.from_data_list(samples_list).to(self.device)
        samples_list = [sample.to(self.device) for sample in samples_list]



        # compute rdkit metrics for the batch
        [validity, uniqueness, novelty], _, _ = evaluation.compute_rdkit_metrics_batch(samples_batch, 
                                                                                 dataloader.dataset, 
                                                                                 open_babel=config.eval_open_babel,
                                                                                 with_conformer=config.eval_with_conformer,
                                                                                 limit_bonds_to_one=config.eval_limit_bonds_to_one)
        msg = f"Eval Ep {epoch}, Iter {iteration}: Validity: {validity*100:.3f}, Uniqueness: {uniqueness*100:.3f}, Novelty: {novelty*100:.3f}"

        _, batch_results = evaluation.compute_stability_batch(samples_batch, 
                                                        atom_decoder=dataloader.dataset.atom_decoder,
                                                        open_babel=config.eval_stability_open_babel,
                                                        verbose=False)
        percentage_stable_bonds = batch_results['percentage_stable_bonds']
        percentage_stable_molecules = batch_results['percentage_stable_molecules']
        msg += f", Stable Bonds: {percentage_stable_bonds:.3f}, Stable Molecules: {percentage_stable_molecules:.3f}"
        print(msg)