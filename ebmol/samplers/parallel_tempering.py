import torch
from tqdm import tqdm
from torch_geometric.data import Batch


class ParallelTemperingSampler:

    def __init__(self, sampler, config):
        self.sampler = sampler
        self.generate_random = sampler.generate_random
        self.temp_scales = torch.tensor(config.get("temp_scales"))
        
        self.relaxation_steps = config.get("relaxation_steps")
        self.swap_steps = config.get("swap_steps")
        self.withdraw_steps = config.get("withdraw_steps")     #extract sample after every n swaps
        self.burn_in_steps = config.get("burn_in_steps")
        
        self.replace_runaways = config.get("replace_runaways")
        self.runaway_threshold = config.get("runaway_threshold", None)
        
        self.reseed = config.get("reseed")
        self.diverged_counter = 0
        # Fresh-noise transport: reseeded replicas are frozen and pushed upward
        # through temperature/noise levels until they reach highest_noise_level_idx.
        self.enable_fresh_noise_transport = bool(config.get("enable_fresh_noise_transport", False))
        self.highest_noise_level_idx = int(config.get("highest_noise_level_idx", 0))
        if self.enable_fresh_noise_transport:
            assert 0 <= self.highest_noise_level_idx < len(self.temp_scales), "highest_noise_level_idx is out of bounds for temp_scales."
        self.fresh_noise_flags = None
        # is enable_fresh_noise_transport is true, reseed also has to be true
        assert (not self.enable_fresh_noise_transport) or self.reseed, "enable_fresh_noise_transport requires reseed to be True."

        # Optional diversity injection: perturb only a small random fraction of
        # harvested low-temperature replicas to improve cross-level mixing.
        self.inject_diversity_noise = config.get("inject_diversity_noise", False)
        self.diversity_noise_fraction = float(config.get("diversity_noise_fraction", 0.0))
        self.diversity_noise_pos_std = float(config.get("diversity_noise_pos_std", 0.0))
        self.diversity_noise_x_std = float(config.get("diversity_noise_x_std", 0.0))
        self.diversity_noise_min_count = int(config.get("diversity_noise_min_count", 1))
        assert 0.0 <= self.diversity_noise_fraction <= 1.0, "diversity_noise_fraction must be in [0, 1]."
        
        self.check_before_harvest_func = config.get("check_before_harvest_func", None)   # function that takes data list and returns boolean mask of which samples to keep
        self.output_filter_func = config.get("output_filter_func", None)
        self.rlx_temp = config.get("rlx_temp", 0.0)
        self.energy_type = config.get("energy_type", "max_node")
        
        self.batch_size = config.get("replicas_per_temp", 16)
        self.persistent = config.get("persistent", True)
        self.config = config
        self.replicas = None
        self.burned_in = False

        self.energy_composition = config.get("composition", False)
        if self.energy_composition:
            assert self.sampler.energy_model.composites is not None and len(self.sampler.energy_model.composites) > 0, "Energy composition is enabled but no composites found in energy model."
            assert len(config.get("composite_weights", [])) == len(self.sampler.energy_model.composites), "Length of composite_weights must match number of composites in energy model."
            self.sampler.energy_model.composite_weights = config.get("composite_weights", [1.0])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def relax_samples(self, samples, gradient_mask=None):
        if self.relaxation_steps > 0:
            # set phi to zero during relaxation
            self.set_relaxation_parameters()
            if gradient_mask is not None:
                gradient_mask = gradient_mask[:samples.pos.shape[0]]
            relaxed_samples, relaxation_infos = self.sampler._step(samples, self.relaxation_steps, temp=self.rlx_temp, gradient_mask=gradient_mask)
            self.reset_relaxation_parameters()
            return relaxed_samples, relaxation_infos
        else:
            return samples, None
            
    def set_relaxation_parameters(self):
        self.old_phi = self.sampler.phi
        self.old_min_cat_x = self.sampler.min_cat_x
        self.old_step_size_pos = self.sampler.step_size_pos
        self.old_step_size_x = self.sampler.step_size_x

        self.sampler.step_size_pos = self.config.get("rlx_step_size_pos")
        self.sampler.step_size_x = self.config.get("rlx_step_size_x")
        self.sampler.phi = self.config.get("rlx_phi")
        self.sampler.min_cat_x = self.config.get("rlx_min_cat_x")
    
    def reset_relaxation_parameters(self):
        self.sampler.phi = self.old_phi
        self.sampler.min_cat_x = self.old_min_cat_x
        self.sampler.step_size_pos = self.old_step_size_pos
        self.sampler.step_size_x = self.old_step_size_x

    def calc_energy(self, data_batch):
        with torch.no_grad():
            if self.energy_type == "max_node":
                _, node_energies, _ = self.sampler.energy_model(data_batch, return_node_energy=True)
                energies = torch.scatter_reduce(torch.zeros(data_batch.batch_size, device=data_batch.x.device), 0, data_batch.batch, node_energies.squeeze(), reduce='max', include_self=False)  # dims [batch_size, out_node_nf=1]
            elif self.energy_type == "avg_node":
                _, node_energies, _ = self.sampler.energy_model(data_batch, return_node_energy=True)
                energies = torch.scatter_reduce(torch.zeros(data_batch.batch_size, device=data_batch.x.device), 0, data_batch.batch, node_energies.squeeze(), reduce='mean', include_self=False)  # dims [batch_size, out_node_nf=1]
            elif self.energy_type == "energy":
                energies = self.sampler.energy_model(data_batch).squeeze(-1)
            else:
                raise ValueError(f"Unknown energy_type {self.energy_type} in ParallelTemperingSampler.")
        return energies

    def calc_n_steps(self, n_samples):
        # calculate number of steps to run sampler for
        n_withdraws = n_samples // self.batch_size
        n_steps = n_withdraws * self.withdraw_steps * self.swap_steps
        if self.replicas is None:
            n_steps += self.burn_in_steps * self.swap_steps
        return n_steps

    def _inject_diversity_noise(self, replica_indices):
        """Inject small Gaussian noise into a random subset of selected replicas."""
        if self.diversity_noise_fraction <= 0.0:
            return 0
        if len(replica_indices) == 0:
            return 0

        n_candidates = int(replica_indices.numel())
        n_to_perturb = int(round(self.diversity_noise_fraction * n_candidates))
        if self.diversity_noise_fraction > 0.0:
            n_to_perturb = max(self.diversity_noise_min_count, n_to_perturb)
        n_to_perturb = min(n_candidates, n_to_perturb)
        if n_to_perturb <= 0:
            return 0

        perm = torch.randperm(n_candidates, device=replica_indices.device)
        chosen_replica_indices = replica_indices[perm[:n_to_perturb]]
        atom_mask = torch.isin(self.replicas.batch, chosen_replica_indices)

        if self.diversity_noise_pos_std > 0.0:
            pos_noise = torch.randn_like(self.replicas.pos[atom_mask]) * self.diversity_noise_pos_std
            self.replicas.pos[atom_mask] = self.replicas.pos[atom_mask] + pos_noise

        if self.diversity_noise_x_std > 0.0:
            x_noise = torch.randn_like(self.replicas.x[atom_mask]) * self.diversity_noise_x_std
            x_noisy = self.replicas.x[atom_mask] + x_noise
            # Keep categorical node features normalized after perturbation.
            self.replicas.x[atom_mask] = torch.softmax(x_noisy, dim=-1)

        return n_to_perturb
    
    def _sample(self, batch_size, samples=None, verbose=False, pbar=None, sample_ts=None, sample_ts_step=None, n_steps=None, n_atom_list=None, gradient_mask=None):
        assert self.withdraw_steps >= 2, "if 1, then reseeded samples stay in lowest temp level."

        n_steps = self.calc_n_steps(batch_size)
        self.diverged_counter = 0

        self.temp_scales = self.temp_scales.to(self.device)

        self.temp_levels = torch.arange(len(self.temp_scales), device=self.device)
        num_replicas = len(self.temp_scales) * self.batch_size
        if samples is None:
            self.replicas = self.sampler.generate_random(batch_size=num_replicas, n_atom_list=n_atom_list)
            self.replicas = Batch.from_data_list(self.replicas).to(self.device)
        else:
            assert samples.batch_size == num_replicas, f"Number of provided samples {samples.batch_size} does not match number of replicas {num_replicas}."
            self.replicas = samples

        self.replica_temp_levels = self.temp_levels.repeat_interleave(self.batch_size)
        if (self.fresh_noise_flags is None) or (self.fresh_noise_flags.numel() != num_replicas) or (samples is None):
            self.fresh_noise_flags = torch.zeros(num_replicas, dtype=torch.bool, device=self.device)
        else:
            self.fresh_noise_flags = self.fresh_noise_flags.to(self.device)
        sampled_data = []
        
         
        temp_levels_log = [self.replica_temp_levels.cpu().numpy()]
        energy_log = []
        swap_rates_log = {t_lvl.item():[] for t_lvl in self.temp_levels}
        avg_energy_log = {t_lvl.item():[] for t_lvl in self.temp_levels}
        num_replicas_per_level_log = {t_lvl.item(): [self.batch_size] for t_lvl in self.temp_levels}
        check_before_harvest_acceptance_log = []
        relaxation_infos = None

        swap_rates = {i: 0.0 for i in range(len(self.temp_levels)-1)}
        num_replacements = 0
        diversity_injections_log = []

        if verbose:
            pb = tqdm(range(n_steps // self.swap_steps), desc="PT Sampling")
        else:
            pb = range(n_steps // self.swap_steps)

        for step in pb:
            # Clear transport flag only when flagged replicas reach highest-noise level.
            if self.enable_fresh_noise_transport and self.fresh_noise_flags.any():
                reached_highest_noise = self.fresh_noise_flags & (
                    self.replica_temp_levels == self.temp_levels[self.highest_noise_level_idx]
                )
                self.fresh_noise_flags[reached_highest_noise] = False

            # Keep flagged replicas unchanged during sampler dynamics.
            frozen_atom_mask = None
            frozen_pos = None
            frozen_x = None
            if self.enable_fresh_noise_transport and self.fresh_noise_flags.any():
                flagged_replica_indices = self.fresh_noise_flags.nonzero(as_tuple=False).flatten()
                frozen_atom_mask = torch.isin(self.replicas.batch, flagged_replica_indices)
                if frozen_atom_mask.any():
                    frozen_pos = self.replicas.pos[frozen_atom_mask].clone()
                    frozen_x = self.replicas.x[frozen_atom_mask].clone()

            temps = self.temp_scales[self.replica_temp_levels]
            if self.energy_composition:
                self.sampler.energy_model.composition = True
            self.replicas = self.sampler._step(self.replicas, self.swap_steps, temp=temps, gradient_mask=gradient_mask)[0] 
            if self.energy_composition:
                self.sampler.energy_model.composition = False

            if frozen_atom_mask is not None and frozen_atom_mask.any():
                self.replicas.pos[frozen_atom_mask] = frozen_pos
                self.replicas.x[frozen_atom_mask] = frozen_x

            energies = self.calc_energy(self.replicas)

            if self.replace_runaways:
                replace_mask = energies > self.runaway_threshold
                self.diverged_counter += replace_mask.sum().item()
                if self.enable_fresh_noise_transport and self.fresh_noise_flags.any():
                    replace_mask = replace_mask & (~self.fresh_noise_flags)
                if verbose: num_replacements += replace_mask.sum().item()
            else:
                replace_mask = torch.zeros_like(energies, dtype=torch.bool)

            # Extract samples from lowest temperature replicas
            burnin_done =  self.burned_in or step > self.burn_in_steps
            if (step + 1) % self.withdraw_steps == 0 and burnin_done:

                min_temp_mask = (self.replica_temp_levels == self.temp_levels[-1])
                min_temp_inds = min_temp_mask.nonzero().flatten()

                if not min_temp_mask.any(): continue
                
                samples = Batch.from_data_list(self.replicas[min_temp_inds])

                samples, relaxation_infos = self.relax_samples(samples, gradient_mask=gradient_mask)

                # apply check_before_harvest_func if provided
                if self.check_before_harvest_func is not None:
                    check_mask = self.check_before_harvest_func(samples)
                    min_temp_inds = min_temp_inds[check_mask]
                    check_before_harvest_acceptance_log.append(check_mask.float().mean().item())
                    
                    # only output samples that passed the check
                    samples_list = samples[check_mask]
                    if len(samples_list) <= 0: continue
                    samples = Batch.from_data_list(samples_list)
                
                # only reseed samples that were accepted, leave others unchanged
                if self.reseed:
                    replace_mask[min_temp_inds] = True
                    if self.enable_fresh_noise_transport:
                        self.fresh_noise_flags[min_temp_inds] = True

                # Inject controlled noise into a small subset of harvested
                # low-temperature chains to encourage level mixing.
                if self.inject_diversity_noise:
                    n_injected = self._inject_diversity_noise(min_temp_inds)
                    diversity_injections_log.append(n_injected)

                # store samples that passed the check
                sampled_data.append(samples)#.detach().cpu())

                self.burned_in = True

            # replace replicas with extremely low energy with random samples with same number of atoms
            if replace_mask.any():
                    num_replace = replace_mask.sum().item()
                    replace_indices = replace_mask.nonzero().flatten()
                    num_atoms = self.replicas.batch.bincount()[replace_indices]
                    new_replicas = self.sampler.generate_random(batch_size=num_replace, n_atom_list=num_atoms)

                    # overwrite pos and x of the replicas to be replaced in the batched data
                    new_replicas = Batch.from_data_list(new_replicas).to(self.device)
                    mask = torch.isin(self.replicas.batch, replace_indices)
                    self.replicas.pos[mask] = new_replicas.pos
                    self.replicas.x[mask] = new_replicas.x

                    # recalculate energies for all replicas
                    new_energies = self.calc_energy(new_replicas)
                    energies[replace_indices] = new_energies


            # Attempt swaps between adjacent replicas with DEO strategy
            parity = 1 - (step % 2)  # even or odd
            for t_idx in range(parity, len(self.temp_levels) - 1, 2):
                upper_idxs = (self.replica_temp_levels == self.temp_levels[t_idx]).nonzero().flatten()
                lower_idxs = (self.replica_temp_levels == self.temp_levels[t_idx + 1]).nonzero().flatten()

                upper_energy = energies[upper_idxs]
                lower_energy = energies[lower_idxs]
                beta_upper = 1.0 / self.temp_scales[t_idx]
                beta_lower = 1.0 / self.temp_scales[t_idx + 1]
                delta = (beta_upper - beta_lower) * (upper_energy - lower_energy)
                acceptance_prob = torch.exp(delta)
                acceptance_prob = torch.clamp(acceptance_prob, max=1.0)

                random_vals = torch.rand_like(acceptance_prob, device=self.device)
                swap_mask = random_vals < acceptance_prob

                if self.enable_fresh_noise_transport:
                    upper_flagged = self.fresh_noise_flags[upper_idxs]
                    lower_flagged = self.fresh_noise_flags[lower_idxs]
                    # Always move flagged replicas upward and never downward.
                    swap_mask = swap_mask | lower_flagged
                    swap_mask = swap_mask & (~upper_flagged)

                # Swap configurationsq by swapping temp values
                upper_swap_idxs = upper_idxs[swap_mask]
                lower_swap_idxs = lower_idxs[swap_mask]

                self.replica_temp_levels[upper_swap_idxs] = self.temp_levels[t_idx + 1]
                self.replica_temp_levels[lower_swap_idxs] = self.temp_levels[t_idx]

                mean_sr = swap_mask.float().mean().item()
                swap_rates[t_idx] = mean_sr
                swap_rates_log[t_idx].append(mean_sr)

            # update logs 
            # calc average energy per level
            avg_energy = []
            for t in self.temp_levels:
                level_mask = (self.replica_temp_levels == t)
                if level_mask.sum() > 0:
                    mean_energy = energies[level_mask].mean().item()
                    avg_energy.append(mean_energy)
                else:
                    avg_energy.append(0.0)
                avg_energy_log[t.item()].append(mean_energy)
                num_replicas_per_level_log[t.item()].append(level_mask.sum().item())

            # update progress bar and logs
            if (step + 1) % self.withdraw_steps == 0 and verbose:
                    #avg_energy_str = ','.join([f'T{t}:{avg_energy[t]:.1f}' for t in range(len(self.temp_levels))])
                    energy_last_step_str = ','.join([f'T{t}:{avg_energy_log[t][-2]:.1f}' for t in range(len(self.temp_levels))])
                    swap_rates_str = ','.join([f'T{t}-{t+1}:{swap_rates.get(t, 0.0):.1f}' for t in range(len(self.temp_levels)-1)])
                    replacements_per_step = round(num_replacements / self.withdraw_steps, 1)
                    replacements_per_step_str = f'{replacements_per_step:.1f}/{self.batch_size * len(self.temp_levels)}'
                    pb.set_postfix({'Rnawys/stp': replacements_per_step_str, 'E/lvl': energy_last_step_str, '\tSR/lvl': swap_rates_str})
                    num_replacements = 0


            #map temp levels to indices
            temp_levels_log.append(self.replica_temp_levels.cpu().numpy())
            energy_log.append(energies.cpu().numpy())


        if self.output_filter_func is not None:
            filtered_samples = []
            # concatenate all sampled data
            n_rejected = 0
            n_total = 0
            for data_batch in sampled_data:
                filter_mask = self.output_filter_func(data_batch.cuda())
                assert sum(filter_mask) > 0, "All samples were filtered out by output_filter_func."
                filtered_samples.append(Batch.from_data_list(data_batch[filter_mask]).cpu())
                n_rejected += (~filter_mask).sum().item()
                n_total += len(data_batch)
            sampled_data = filtered_samples
            tqdm.write(f"Filtered out {100 * n_rejected / n_total:.2f}% with output_filter_func.")
        
        # convert list of batches to single batch
        data_list = []
        for data_batch in sampled_data:
            data_list.extend(data_batch.to_data_list())
        sampled_data = Batch.from_data_list(data_list)

        if not self.persistent:
            self.replicas = None
            self.burned_in = False
            self.replica_temp_levels = None
            self.fresh_noise_flags = None

        infos = {'temp_lvl_log': temp_levels_log, 
                 'energies_log': energy_log, 
                 'swap_rates_log': swap_rates_log, 
                 'avg_energy_log': avg_energy_log, 
                 'num_replicas_per_level_log': num_replicas_per_level_log,
                 'diversity_injections_log': diversity_injections_log,
                 'check_before_harvest_acceptance_log': check_before_harvest_acceptance_log,
                 'relaxation_infos': relaxation_infos,
                 'diverged_counter': self.diverged_counter}
        if verbose:
            # Concatenate sampled data
            return sampled_data, infos
        else:
            return sampled_data

