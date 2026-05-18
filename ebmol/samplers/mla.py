import torch
from torch.nn.functional import one_hot 
import torch.nn.functional as F

from ebmol.samplers.base_sampler import MCMCSampler
from torch_geometric.nn import global_mean_pool

class MLASampler(MCMCSampler):
    def __init__(self, 
                 energy_model, 
                 generate_random, 
                 step_size_pos=0.05, 
                 step_size_x=0.05,
                 phi=0.00,
                 sigma_scale=0.2,
                 batch_size=1, 
                 verbose=False, 
                 experiment=None,
                 config=None):


        
        generate_random_fkt = generate_random

        super().__init__(energy_model, generate_random_fkt,  
                         batch_size=batch_size,
                         verbose=verbose,
                         experiment=experiment,
                         config=config)

        self.step_size_pos = config.get('step_size_pos', step_size_pos)
        self.step_size_x = config.get('step_size_x', step_size_x)
        self.phi = config.get('phi', phi)
        self.temperature = config.get('temperature', 1.0)
        self.predict_atom_types = config.get('predict_atom_types', True)
        self.min_cat_x = config.get('min_cat_x', 0.01)
        self.sigma_scale_cat = config.get('sigma_scale_cat', sigma_scale)
        self.sigma_scale_pos = config.get('sigma_scale_pos', sigma_scale)
        self.verbose = verbose

    def _step(self, data, n_steps, temp=None, gradient_mask=None):

        if n_steps == 0:
            return data, {}
        data = data.detach()

        initial_data = data.clone()
        sample_path = []
        step_size_x = self.step_size_x
        step_size_pos = self.step_size_pos
        phi = self.phi

        if temp is None:
            temp = self.temperature

        per_mol_temp = temp     #either scalar or tensor with size [#batch]

        # check if temp is tensor, we have single temperature per graph. 
        if torch.is_tensor(temp):
            temp = temp[data.batch]
        else:
            temp = torch.tensor(temp, device=data.pos.device)


        # Perform Langevin steps
        prev_pos = initial_data.pos.clone() # [#batch, #atoms, 3]
        prev_data = initial_data.clone()
        for i in range(n_steps):
            
            prop_data = data.clone()
            per_sample_guidance_weights = None if not self.energy_model.composition else per_mol_temp
            grads, logp = self.energy_model.get_grad(data, gradient_mask=gradient_mask, guidance_weights=per_sample_guidance_weights)
            
            # update cat
            grads_cat = grads[1]

            if self.predict_atom_types:
                x = torch.clamp(data.x, min=self.min_cat_x)
                noise =  torch.randn(x.shape, device=x.device) * self.sigma_scale_cat
                chol_hessian = 1.0 / torch.sqrt(x)

                drift = -step_size_x * grads_cat
                sqrt_2_eta = torch.sqrt(torch.tensor(2.0 * step_size_x, device=x.device) * temp).reshape(-1,1)
                diffusion = sqrt_2_eta * chol_hessian * noise

                # softmax version
                logits = torch.log(x) + drift + diffusion
                prop_data.x = F.softmax(logits, dim=-1)

            # update pos
            grad_pos = grads[0]

            pos = data.pos.clone() # [#batch, #atoms, 3]

            drift = -step_size_pos*grad_pos  # Move in direction of gradient to increase log prob
            momentum = phi*(pos - prev_pos)
            noise = torch.randn(pos.shape, device=pos.device) * self.sigma_scale_pos * torch.sqrt(torch.tensor(2.0*step_size_pos*(1-phi), device=pos.device) * temp).reshape(-1,1)
            if gradient_mask is not None:
                noise[gradient_mask] = torch.zeros_like(noise[gradient_mask])
            delta_pos = drift + momentum + noise

            prop_pos = pos + delta_pos  

            # subtract mean 
            per_mol_mean = global_mean_pool(prop_pos, data.batch)
            prop_pos = prop_pos - per_mol_mean[data.batch]
            
            prop_data.pos = prop_pos.detach()

            if gradient_mask is not None:
                prop_data.pos[gradient_mask] = data.pos[gradient_mask]
                prop_data.x[gradient_mask] = data.x[gradient_mask]

            # reject step if nan or inf in logp
            if torch.any(torch.isnan(prop_data.x)) or torch.any(torch.isinf(prop_data.x)):
                # only reject nan/inf samples in batch
                reject_mask = torch.logical_or(torch.any(torch.isnan(prop_data.x), dim=(1)), torch.any(torch.isinf(prop_data.x), dim=(1)))
                
                # find batch indices
                reject_batch_indices = torch.unique(data.batch[reject_mask])

                # update rejection mask to cover all atoms of rejected graphs
                for reject_batch_index in reject_batch_indices:
                    reject_mask[data.batch == reject_batch_index] = True

                # keep previous data for rejected samples
                prop_data.pos[reject_mask] = prev_data.pos[reject_mask]
                prop_data.x[reject_mask] = prev_data.x[reject_mask]
                # print num of rejected samples
                print("Step {}: Rejecting {} samples out of {} due to NaN/Inf in categorical features".format(i, len(reject_batch_indices), data.batch_size))

            prev_pos = data.pos
            prev_data = data.clone()
            data = prop_data

            sample_path.append(data.detach().clone().cpu())

        #print("Percentage of categorical updates that did not decrease energy: {:.2f}%".format(100*e_decrease_updates/(n_steps*self.predict_atom_types)))
        infos = {}
        infos['sample_path'] = sample_path
            

        return data, infos

