"""
Model wrapper — handles training, validation, label generation,
early stopping, and checkpoint management.

Mirrors the deep-fmask Model class but adapted for Landsat 8.
"""

import logging
import os
from shutil import copyfile

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel

from dataset.network_input import get_inp_func, get_inp_channels
from network.unet import UNet
from utils.MFB import calculate_file_freq
from utils.csv_logger import print_val_csv_metrics, make_overall_statistics_csv
from utils.dir_paths import TRAIN_PATH
from utils.metrics import calculate_accuracy, calculate_confusion_matrix, Metrics

logger = logging.getLogger(__name__)

# Self-training pipeline: network grows at each stage
# Stage:        0     1     2     3
FILTER_OPTIONS = [16, 32, 24, 32]
DEPTH_OPTIONS = [5, 5, 6, 6]

NUM_CLASSES = 6


class Model:
    """
    Wraps UNet with training/inference logic for the self-training pipeline.
    """

    def __init__(self, experiment, gpu_id):
        self.exp = experiment
        # DataParallel requires the model to live on device_ids[0].
        # Use the first requested GPU as the primary device.
        if torch.cuda.is_available() and gpu_id:
            self.device = torch.device(f'cuda:{gpu_id[0]}')
        else:
            self.device = torch.device('cpu')

        self.inp_func = get_inp_func(self.exp.inp_mode)
        n_inp_channels = get_inp_channels(self.exp.inp_mode)

        if experiment.full:
            depth = DEPTH_OPTIONS[-1]
            start_filters = FILTER_OPTIONS[-1]
        else:
            start_filters = FILTER_OPTIONS[self.exp.stage]
            depth = DEPTH_OPTIONS[self.exp.stage]

        logger.info(f'Stage {self.exp.stage}: depth={depth}, filters={start_filters}')

        # Build network
        self.network = UNet(num_classes=NUM_CLASSES,
                            in_channels=n_inp_channels,
                            depth=depth,
                            start_filts=start_filters,
                            dropout=self.exp.dropout)
        self.network = DataParallel(self.network, device_ids=gpu_id)
        self.network.to(self.device)

        if experiment.mode == 'train':
            self.epoch = 0
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.exp.lr,
                weight_decay=1e-5)
            self.metrics = Metrics(self.device)

            # Early stopping
            self.patience_counter = 0
            self.patience = 10
            self.best_mIoU_moving_avg = 0.0
        else:
            # Load trained weights
            trained_model = self.exp.get_trained_model_info()
            self.network.load_state_dict(trained_model['model_state_dict'])

            self.metrics = Metrics(self.device)

            if experiment.mode == 'label_gen':
                self.stage_freq_data = []

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def forward_step(self, input_img):
        input_img = self.inp_func(input_img)
        output = self.network(input_img.to(self.device))
        return output

    def get_loss(self, output, labels, mode, fmask=None):
        labels = labels.to(self.device)
        if self.exp.weights is not None:
            w = torch.from_numpy(self.exp.weights).float().to(self.device)
        else:
            # test/predict mode: use uniform weights (no MFB available)
            w = torch.ones(NUM_CLASSES, dtype=torch.float32).to(self.device)
        loss = F.cross_entropy(output, labels, w)

        predicted_label = self.encode_label(output)
        if mode != 'train':
            self.metrics.val_confusion_matrix += calculate_confusion_matrix(
                predicted_label, labels, mode)
        else:
            acc = calculate_accuracy(predicted_label, labels, mode).detach()
            self.metrics.add_step_info(mode, loss.detach(), acc)
            return loss

        self.metrics.add_step_info(mode, loss.detach())
        return loss

    # ------------------------------------------------------------------
    # Train / valid steps
    # ------------------------------------------------------------------
    def valid_step(self, network_data, mode='test'):
        input_img = network_data[0]
        output = self.forward_step(input_img)
        _, labels, filenames = network_data

        if mode == 'train':
            labels = labels[:, 0, :, :]
        elif mode == 'label_gen':
            self.generate_train_data(
                filenames,
                self.encode_label(output, label_gen=True))
            return 0
        elif mode == 'test':
            fmask = labels[:, 0, :, :].to(self.device)
            if labels.shape[1] >= 2:
                labels = labels[:, 1, :, :]
            else:
                labels = labels[:, 0, :, :]
        elif mode == 'predict':
            self.generate_train_data(
                filenames,
                self.encode_label(output),
                label_gen=False)
            return 0

        loss = self.get_loss(output, labels, mode)
        return loss

    def train_step(self, network_data):
        loss = self.valid_step(network_data, mode='train')
        self.backward_step(loss)
        return loss

    def backward_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Encoding / pseudo-labels
    # ------------------------------------------------------------------
    @staticmethod
    def encode_label(out, label_gen=False, threshold=0.4):
        """
        Convert network output to class predictions.
        For label generation, only keep predictions with confidence ≥ threshold.
        """
        softmax_out = F.softmax(out, dim=1)
        predicted_labels = torch.argmax(softmax_out, dim=1)
        if label_gen:
            prob = torch.max(softmax_out, dim=1)[0]
            predicted_labels[prob < threshold] = 0
        return predicted_labels

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------
    def refresh_stats(self):
        train_metrics, valid_metrics, class_metrics_dict = \
            self.metrics.aggregate_metrics(self.epoch)
        make_overall_statistics_csv(
            train_metrics, valid_metrics, class_metrics_dict,
            self.epoch, self.exp.log_path)
        self.metrics.reset_metrics()
        self.epoch += 1
        self.save()
        return self.check_early_stop()

    def check_early_stop(self):
        if self.epoch <= self.patience:
            return False

        mIoU_moving_avg = np.mean(
            self.metrics.val_mIoU_history[-self.patience:])
        logger.info(f'mIoU moving avg: {mIoU_moving_avg:.4f} '
                     f'(best: {self.best_mIoU_moving_avg:.4f})')

        if mIoU_moving_avg >= self.best_mIoU_moving_avg:
            self.best_mIoU_moving_avg = mIoU_moving_avg
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                logger.info(f'Early stopping at epoch {self.epoch}')
                return True
        return False

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save(self):
        save_path = os.path.join(self.exp.model_folder,
                                 f'model_{self.epoch}.pth')
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.network.state_dict(),
            'stage': self.exp.stage,
            'full': self.exp.full,
            'inp_mode': self.inp_func.__name__,
        }, save_path)

    def save_best_model(self):
        best_epoch = np.argmax(self.metrics.val_mIoU_history)
        best_mIoU = self.metrics.val_mIoU_history[best_epoch]

        logger.info(f'Best mIoU {best_mIoU:.3%} at Epoch {best_epoch + 1}')
        print(f'Best mIoU {best_mIoU:.3%} at Epoch {best_epoch + 1}')

        src = os.path.join(self.exp.model_folder,
                           f'model_{best_epoch + 1}.pth')
        dst = os.path.join(self.exp.model_folder, 'model_best.pth')
        copyfile(src, dst)
        print_val_csv_metrics(best_epoch + 1, self.exp.log_path)

    # ------------------------------------------------------------------
    # Pseudo-label generation (writes back into H5 files)
    # ------------------------------------------------------------------
    def generate_train_data(self, filenames, labels, label_gen=True):
        """Save model predictions into the HDF5 files for the next stage."""
        n_bands = 8  # Spectral bands in H5: B1–B7 + B9

        for label, filename in zip(labels, filenames):
            label_np = label.cpu().numpy().astype(np.uint16)
            label_np = label_np[1:-1, 1:-1]  # Remove padding

            with h5py.File(filename, 'r') as hf:
                data = hf.get('data')[:]

            if label_gen:
                fmask = data[:, :, n_bands]  # QA_PIXEL labels channel
                label_np[fmask == 0] = 0  # Preserve No-Data
                target_index = n_bands + 2  # Pseudo-label channel

                new_label_freq = calculate_file_freq(label_np)
                self.save_stats(filename, new_label_freq)
            else:
                target_index = n_bands + 1  # Prediction channel

            if data.shape[-1] < (target_index + 1):
                new_shape = list(data.shape)
                new_shape[-1] = (target_index + 1) - data.shape[-1]
                data = np.append(data,
                                 np.zeros(new_shape, dtype=np.uint16),
                                 axis=-1)

            data[:, :, target_index] = label_np

            os.remove(filename)
            with h5py.File(filename, 'w') as hf:
                hf.create_dataset('data', data=data.astype(np.uint16))

    def save_stats(self, filename, new_label_freq):
        row = [os.path.basename(filename)]
        row.extend(new_label_freq.tolist())
        row = ','.join([str(i) for i in row])
        self.stage_freq_data.append(row)

    def write_stage_stats(self):
        label_stats_file = os.path.join(
            TRAIN_PATH,
            f"label_stats_stage{self.exp.config['stage'] + 1}.csv")
        with open(label_stats_file, 'w') as f:
            f.write('FILENAME,NONE_F,CLEAR_F,CLOUD_F,SHADOW_F,ICE_F,WATER_F\n')
            for i in self.stage_freq_data:
                f.write(f'{i}\n')
