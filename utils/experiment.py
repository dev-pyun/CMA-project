"""
Experiment management — creates/loads experiment directories for models and logs.
"""

import logging
import os

import torch

from utils.dir_paths import EXP_DATA_PATH

logger = logging.getLogger(__name__)


class Experiment:
    """Manages paths, configuration, and model checkpoints for an experiment."""

    def __init__(self, args, mode='train'):
        self.mode = mode
        self.exp_name = args.exp_name

        # Parse self-training stage from the experiment name  (e.g. exp1_stage2)
        if '_stage' in self.exp_name:
            parts = self.exp_name.rsplit('_stage', 1)
            self.stage = int(parts[1])
        else:
            self.stage = getattr(args, 'stage', 0)

        self.full = getattr(args, 'full', False)
        self.dropout = getattr(args, 'dropout', True)
        self.lr = getattr(args, 'learning_rate', 1e-6)
        self.inp_mode = getattr(args, 'inp_mode', 'swirndsi')
        self.weights = None  # set externally by MFB

        # Store full config for saving
        self.config = {
            'exp_name': self.exp_name,
            'stage': self.stage,
            'full': self.full,
            'dropout': self.dropout,
            'lr': self.lr,
            'inp_mode': self.inp_mode,
        }

        # Directory setup
        self.exp_folder = os.path.join(EXP_DATA_PATH, self.exp_name)
        self.model_folder = os.path.join(self.exp_folder, 'model')
        self.log_path = os.path.join(self.exp_folder, 'log')

        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        # Logging setup
        log_file = os.path.join(self.log_path, f'{mode}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
        logging.getLogger().addHandler(file_handler)

        logger.info(f'Experiment: {self.exp_name}  Mode: {mode}  Stage: {self.stage}')

    def get_trained_model_info(self, epoch=0):
        """
        Load a saved model checkpoint.

        Parameters
        ----------
        epoch : int
            Epoch to load.  0 means 'best model'.
        """
        if epoch == 0:
            model_path = os.path.join(self.model_folder, 'model_best.pth')
        else:
            model_path = os.path.join(self.model_folder, f'model_{epoch}.pth')

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

        logger.info(f'Loading model from {model_path}')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint
