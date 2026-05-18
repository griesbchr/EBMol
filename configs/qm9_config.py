import easydict
import os


config = easydict.EasyDict({
    "root": os.getcwd(),
    "project_name": "ebmol-qm9",
    "model_name": "MLD",
    "dataset_name": "qm9",
    "exp_appendix": "ls_msp_ereg", # appendix to experiment name
    
    # description
    "description": "",

    # load checkpoint
    "cktpt_path": "",

    # training parameters 
    "epochs": 1000,
    "lr": 5e-5,                
    "min_lr":5e-5,
    "batch_size": 64,   
    "optimizer": "adam",
    "size_prior_per_atom": True, # scale position prior, one x,y,z std per individual molecule
    
    # model architecture
    "n_feat": 5,
    "hidden_nf": 256,   #128, 256
    "out_node_nf": 1,
    "in_edge_nf": 1,
    "n_layers": 9,          # 4, 9
    "agg_method": "sum", 
    "agg_method_energy": "sum",
    "tanh": True,      
    "attention": True, 
    "energy_based": True,
    "output_embedding": "mlp", # linear, mlp
    "output_bias": False,
    "output_activation": None, # None, softplus
    "energy_feats": "h_out", # h_out, h_out_and_x, 
    "flow_matching_ot": True,

    "use_ema": True,
    "ema_decay": 0.999,
    "eval_ema": True,
    
    # loss type
    "loss_type": "restoring_field_matching", # flow_matching, restoring_field_matching
    "loss_smoothing": True,
    "sharpness_gamma": 25.0,  # controls sharpness of scaling function in flow matching extrapolation loss
    "energy_regularization": True,
    "energy_regularization_weight": 1e-3,

    # sampling parameters
    "sampler": "mla",
    "step_size_x": 0.1,       
    "step_size_pos": 0.1,    
    "sigma_scale_pos": 0.6,
    "sigma_scale_cat": 0.6,
    "phi": 0.0,
    "min_cat_x": 0.0015,

    "parallel_tempering": True,
    "temp_scales": [1.0, 0.8, 0.5, 0.35, 0.25, 0.2, 0.15, 0.125, 0.1, 0.075, 0.05],     
    "replicas_per_temp": 8, # number of replicas per temperature                 

    "withdraw_steps": 8,     #extract sample after every n swaps, has to be >= 2  
    "swap_steps": 10, #num steps between swaps                                    
    "burn_in_steps": 16, #num initial swaps to discard                             

    # relaxation sampler parameters
    "relaxation_steps": 50, #num steps to sample at zero temp                     
    "rlx_temp": 0.0,
    "rlx_min_cat_x": 0.0,
    "rlx_step_size_pos": 0.01,
    "rlx_step_size_x": 0.01,
    "rlx_phi": 0.0,


    "energy_type": "max_node",   # uses max node energy to to determine temperatures and swaps
    "reseed": True, # replace sampled data with noise after sampling

    # eval parameters
    "n_eval_samples": 400,
    "eval_stability_open_babel": False,
    "eval_open_babel": False, 
    "eval_with_conformer": False,
    "eval_limit_bonds_to_one": False,   # this one doesnt matter for qm9


    # data params
    "num_atoms": None,    # Either int for fixed number of atoms or specify range as tuple (min, max)
    "predict_atom_types": True, # whether to just condition on atom types (False) or predict them (True)
    
    # intervals
    "eval_interval": 2000, 
    "save_interval": 10000, 
    "log_interval": 200,   
    "verbose": True,

    "load_lr_scheduler": False, # load scheduler from checkpoint
    "distributed_training": False,
})  


