import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader, \
    DSElasticDistributedSampler


class DeepSpeechDataModule(pl.LightningDataModule):

    def __init__(self,
                 labels: list,
                 data_cfg: DataConfig,
                 normalize: bool):
        super().__init__()
        self.labels = labels
        self.data_cfg = data_cfg
        self.spect_cfg = data_cfg.spect
        self.aug_cfg = data_cfg.augmentation
        self.normalize = normalize

    @property
    def is_distributed(self):
        # return self.trainer.devices > 1
        return False

    def train_dataloader(self):
        train_dataset = self._create_dataset(dtype='train')
        if self.is_distributed:
            train_sampler = DSElasticDistributedSampler(
                dataset=train_dataset,
                batch_size=self.data_cfg.batch_size
            )
        else:
            train_sampler = DSRandomSampler(
                dataset=train_dataset,
                batch_size=self.data_cfg.batch_size
            )
        train_loader = AudioDataLoader(
            dataset=train_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
            batch_sampler=train_sampler,
            autoscale_workers=self.data_cfg.autoscale_workers
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self._create_dataset(dtype='validation')
        val_loader = AudioDataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.data_cfg.batch_size,
            autoscale_workers=self.data_cfg.autoscale_workers
        )
        return val_loader

    def _create_dataset(self, dtype):
        dataset = SpectrogramDataset(
            audio_conf=self.spect_cfg,
            dtype=dtype,
            labels=self.labels,
            normalize=True,
            aug_cfg=self.aug_cfg
        )
        return dataset
