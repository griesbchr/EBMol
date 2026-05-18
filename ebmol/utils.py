import comet_ml
import logging
import tqdm
import torch        
import os
import inspect

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

#set up logger
def init_logger(config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(f"{config.exp_path}/log.txt")
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    if config.enable_comet:
        from .comet_cfg import COMET_API_KEY
        # Construct comet experiment
        experiment = comet_ml.Experiment(
            project_name=config.project_name,
            api_key=COMET_API_KEY,
            display_summary_level=0,
        )
        experiment.set_name(config.run_name)

        #log hyperparameters 
        experiment.log_parameters(config)

        #get current file directory  
        current_file = os.path.abspath(inspect.getfile(inspect.currentframe()))
        current_dir = os.path.dirname(current_file)
        
        # log sampler and model files
        experiment.log_code(folder=current_dir)
        experiment.log_code(folder=os.path.join(current_dir, "../samplers"),)
        experiment.log_code(folder=os.path.join(current_dir, "../configs"),)
    else:
        experiment = None

    #log config
    logger.info(f"Starting run {config.run_name}")
    logger.info("Config:")
    for k, v in config.items():
        logger.info(f"{k}: {v}")

    return logger, experiment

def old_init_logger(config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%d.%m.%y %H:%M:%S')
    #log to file
    log_file = f"{config.exp_path}/log.txt"
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    #log config
    logger.info(f"Starting run {config.run_name}")
    logger.info("Config:")
    for k, v in config.items():
        logger.info(f"{k}: {v}")

    return logger
