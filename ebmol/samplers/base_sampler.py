import torch
from torch.nn.functional import one_hot 
from tqdm import tqdm
import numpy as np
import random

from torch_geometric.data import Data, Batch

# define generic sampler that handles torch_geometric batching
class MCMCSampler:
    def __init__(self, energy_model, generate_random, batch_size, verbose, experiment, config=None):
        self.energy_model = energy_model
        self.sample_ts = None# sample_ts
        self.sample_ts_step = None# sample_ts_step
        self.generate_random = generate_random
        self.batch_size = batch_size
        self.persistent = None #persistent
        self.max_buffer_size = None# buffer_size
        self.verbose = verbose
        self.sample_buffer_percentage = None, #sample_buffer_percentage
        self.experiment = experiment
        self.buffer = None
        self.config = config

    #init buffer with one generated batch   
    def init_buffer(self):
        print("Initializing buffer with random samples...")
        self.persistent = False
        
        # if verbose exists set to false
        if hasattr(self.energy_model, 'verbose'):
            self.energy_model.verbose = False
        # generate a batch of random samples
        batch_list = self.generate_random(batch_size=self.max_buffer_size)
        if len(batch_list) > self.max_buffer_size:
            batch_list = batch_list[:self.max_buffer_size]

        self.buffer = batch_list
        self.persistent = True

        if hasattr(self.energy_model, 'verbose'):
            self.energy_model.verbose = True

    def _step(self, data):
        """
        Perform a single step of the sampling process.

        This method should be implemented by subclasses to define the specific
        behavior of the sampling step.

        Args:
            data: type pytorch_geometric.data.Data or torch_geometric.data.Batch
        Returns:
            tuple: A tuple containing the updated data and a dictionary of additional information.
        """
        raise NotImplementedError
    
    def set_step_parameters(self):
        if hasattr(self.config, 'noise_level_pos_step'):
            self.noise_level_pos = self.config.noise_level_pos_step
        if hasattr(self.config, 'noise_level_x_step'):
            self.noise_level_x = self.config.noise_level_x_step
        if hasattr(self.config, 'phi_step'):
            self.phi = self.config.phi_step
        if hasattr(self.config, 'step_size_pos_step'):
            self.step_size_pos = self.config.step_size_pos_step
        if hasattr(self.config, 'step_size_x_step'):
            self.step_size_x = self.config.step_size_x_step
        if hasattr(self.config, 'annealing_factor_step'):
            self.annealing_factor = self.config.annealing_factor_step
        if hasattr(self.config, 'sigma_scale_step'):
            self.sigma_scale = self.config.sigma_scale_step
        if hasattr(self.config, 'temperature_step'):
            self.temperature = self.config.temperature_step
        return
    
    def reset_step_parameters(self):
        if hasattr(self.config, 'annealing_factor_step'):
            self.annealing_factor = self.config.annealing_factor
        if hasattr(self.config, 'noise_level_pos_step'):
            self.noise_level_pos = self.config.noise_level_pos
        if hasattr(self.config, 'noise_level_x_step'):
            self.noise_level_x = self.config.noise_level_x
        if hasattr(self.config, 'phi_step'):
            self.phi = self.config.phi
        if hasattr(self.config, 'step_size_pos_step'):
            self.step_size_pos = self.config.step_size_pos
        if hasattr(self.config, 'step_size_x_step'):
            self.step_size_x = self.config.step_size_x
        if hasattr(self.config, 'sigma_scale_step'):
            self.sigma_scale = self.config.sigma_scale
        if hasattr(self.config, 'temperature_step'):
            self.temperature = self.config.temperature
        return
    
    def _sample(self, batch_size, sample_ts=None, sample_ts_step=None, cat_map_estimate=False, pbar=None):
        """
        Generates batches of samples with a given batch_size using a MCMC method.

        Args:
            n_samples (int): Number of samples to generate.
            sample_ts (optional): Time steps for sampling. Defaults to None.
            pbar (optional): Progress bar object for displaying progress. Defaults to None.

        Returns:
            list: A list of generated samples.

        Notes:
            - The method evaluates the energy model and generates samples either from a buffer or by initializing random samples.
            - It calculates the average change in log probability and the acceptance ratio over the batch.
            - If verbose mode is enabled, it prints the formatted average delta log probability per sample and the acceptance ratio.
        """
        self.energy_model.eval()
        #overwrite sample_ts if provided
        sample_ts = self.sample_ts if sample_ts is None else sample_ts
        sample_ts_step = self.sample_ts_step if sample_ts_step is None else self.sample_ts

        #set sample_ts_step if not provided
        sample_ts_step = sample_ts if sample_ts_step is None else sample_ts_step

        if self.persistent and not self.buffer: self.init_buffer()

        # sample percentage of batch from buffer, generate the rest
        if self.persistent:
            assert batch_size > 1, "Batch size must be greater than 1 for buffer sampling"
            n_gen_samples = int(self.batch_size * (1-self.sample_buffer_percentage))
            n_buffer_samples = self.batch_size - n_gen_samples
            if n_buffer_samples > len(self.buffer):
                n_buffer_samples = len(self.buffer)

            # get buffer samples
            sample_ids = random.sample(range(len(self.buffer)), n_buffer_samples)
            # sample buffer samples
            buffer_samples = [self.buffer[i] for i in sample_ids]
            # remove drawn samples from buffer
            self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if i not in sample_ids]

            if n_gen_samples > 0:

                burnin_data_list = self.generate_random(batch_size=n_gen_samples)

                burnin_samples_batch = Batch.from_data_list(burnin_data_list).to(self.energy_model.device)
                burnin_samples_batch, burnin_batch_infos = self._step(burnin_samples_batch, n_steps=sample_ts)
                stepped_burnin_samples = burnin_samples_batch.to_data_list()
            else:
                burnin_batch_infos = {
                    "mean_delta_logp": 0.0,
                    "accepts": np.zeros(n_gen_samples),
                }   

            # construct data_batch from buffer and random samples
            data_batch_list = buffer_samples + stepped_burnin_samples
            data_batch = Batch.from_data_list(data_batch_list)
        else:
            # whole batch are random samples
            random_data_list = self.generate_random(batch_size=batch_size)
            batch = Batch.from_data_list(random_data_list).to(self.energy_model.device)

            # burnin batch
            data_batch, burnin_batch_infos = self._step(batch, n_steps=sample_ts)


        # always step the batch
        self.set_step_parameters()
        data_batch, step_batch_infos = self._step(data_batch, sample_ts_step)
        self.reset_step_parameters()


        #update buffer if persistent
        if self.persistent:
            data_batch_list = data_batch.to_data_list()
            self.buffer = data_batch_list + self.buffer
            # remove oldest samples if buffer is full
            if len(self.buffer) > self.max_buffer_size:
                self.buffer = self.buffer[:self.max_buffer_size]

        if self.verbose: 
            #print formated with 3 commas after decimal
            msg = f'burnin dlogp: {burnin_batch_infos["mean_delta_logp"]:.3f}'
            if pbar: 
                pbar.set_postfix({"burnin dlogp": f'{burnin_batch_infos["mean_delta_logp"]:.3f}', "acc": f'{np.mean(burnin_batch_infos["accepts"]):.3f}'})
            else:
                tqdm.write(msg)
            if self.experiment:
                self.experiment.log_metric("sample ts", sample_ts)
                self.experiment.log_metric("step ts", sample_ts_step)
                self.experiment.log_metric("mean_logp", burnin_batch_infos["mean_logp"])
                self.experiment.log_metric("burnin dlogp", burnin_batch_infos["mean_delta_logp"])
                self.experiment.log_metric("burnin acc ratio", np.mean(burnin_batch_infos["accepts"]))
                self.experiment.log_metric("burnin cat acc ratio", np.mean(burnin_batch_infos["acceptance_rate_categorical"]))
                self.experiment.log_metric("burnin nan_reject", int(burnin_batch_infos["nan_reject"]))
                self.experiment.log_metric("step nan_reject", int(step_batch_infos["nan_reject"]))
                if self.persistent:
                    self.experiment.log_metric("step dlogp", step_batch_infos["mean_delta_logp"])
                    self.experiment.log_metric("step acc ratio", np.mean(step_batch_infos["accepts"]))
                    self.experiment.log_metric("step cat acc ratio", np.mean(step_batch_infos["acceptance_rate_categorical"]))
                if self.predict_atom_types:
                    self.experiment.log_metric("burnin norm_grad_cat", burnin_batch_infos["norm_grad_cat_last"])
                    self.experiment.log_metric("step norm_grad_cat", step_batch_infos["norm_grad_cat_last"])
                    self.experiment.log_metric("burnin atoms_changed", burnin_batch_infos['atoms_changed'])
                if self.persistent and self.predict_atom_types:
                    self.experiment.log_metric("step atoms_changed", step_batch_infos['atoms_changed'])
                    self.experiment.log_metric("step norm_grad_pos", step_batch_infos["norm_grad_pos_last"])
                    self.experiment.log_metric("burnin norm_grad_pos", burnin_batch_infos["norm_grad_pos_last"])


        # convert to one-hot using map estimation
        #if cat_map_estimate:
        #    data_batch.x = one_hot(torch.argmax(data_batch.x , dim=1), data_batch.x.shape[1]).float()

        return data_batch
    
    '''
    sample function that handles torch_geometric batching
    '''
    def sample(self, n_samples=None, sample_ts=None, pbar=None):
        assert self.buffer is not None, "Buffer sampling not implemented yet. Please use _sampling method instead."
        if n_samples is None or n_samples == 1:
            return self._sample(batch_size=1, sample_ts=sample_ts, pbar=pbar)[0]
        else:
            batch_list = self._sample(n_batches=n_samples//self.batch_size, batch_size=self.batch_size, sample_ts=sample_ts, pbar=pbar)
            sample_list = []
            for batch in batch_list:
                sample_list += batch.to_data_list()
            
            return sample_list#[:n_samples]        
    
        return self._minimize(data=data, n_steps=n_steps, update_mask=update_mask, verbose=verbose)

    def to_one_hot(self, x):
        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)
        x_one_hot[:, range(self.dim), x[0, :]] = 1.
        return x_one_hot
    
    def to_categorical(self, x):
        return torch.argmax(x, dim=2)
    
    def get_grad(self, data, gradient_mask=None):

        grads_all, logp = self.energy_model.get_grad(data, gradient_mask=gradient_mask)

        return grads_all, logp