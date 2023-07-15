import json

import hydra
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything

from deepspeech_pytorch.checkpoint import FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech

import time
import numpy as np


def train(cfg: DeepSpeechConfig):
    seed_everything(cfg.seed)
    
    # Add for DLCache Evaluationde
    with open('/app/dlcache_exp.txt', "r") as f:
        line = f.read().strip()
        args = line.split(',')
    cfg.data.num_workers = int(args[0])
    cfg.data.batch_size = int(args[1])
    cfg.data.autoscale_workers = int(args[-1])
    cfg.trainer.max_epochs = int(args[2])

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    if cfg.trainer.enable_checkpointing:
        checkpoint_callback = FileCheckpointHandler(
            cfg=cfg.checkpoint
        )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
    )

    # # No training for DLCache experiments
    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )

    trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        replace_sampler_ddp=False,
        callbacks=[checkpoint_callback] if cfg.trainer.enable_checkpointing else None,
    )
    # trainer.fit(model, data_loader)
    
    for b in range(int(args[2])):
        # Add for DLCache Evaluation
        train_loader = data_loader.train_dataloader()
        train_loader = iter(train_loader)
        total_batches = len(train_loader)
        load_time = []
        start = time.time()
        
        for i in range(min(total_batches, int(args[3]))):
            t = time.time()
            batch = next(train_loader)
            latency = time.time() - t
            load_time.append(latency)
            if i % int(args[4]) == 0:
                print('Epoch {} : [{}/{}]: latency: {}'.format(b, i+1, total_batches, latency))
            time.sleep(float(args[5]))

        print('Total Read time: {}'.format(np.sum(load_time)))
        print('Total Iteration Time: {}'.format(time.time()-start))
        # np.save('/tmp/load_time.npy', load_time)

