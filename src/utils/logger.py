import logging
import os
from datetime import datetime
import wandb

def setup_logger(log_dir):
    """로깅 설정"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def setup_wandb(config):
    """Weights & Biases 설정"""
    wandb.init(
        project=config['logging']['wandb_project'],
        entity=config['logging']['wandb_entity'],
        config=config
    ) 