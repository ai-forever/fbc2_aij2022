import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from catalyst.data import BalanceClassSampler, DistributedSamplerWrapper
from torch.utils.data import DataLoader
from rudalle import get_tokenizer, get_vae

from src.rudolph.model import get_rudolph_model, ruDolphModel, FP16Module
from src.rudolph import utils

from src.train.dataloader import DatasetRetriever, fb_collate_fn
from src.train.trainer import RudolphLightning
from src.train.utils import create_dataset

from omegaconf import OmegaConf
import argparse
import os  
import pandas as pd
import torch


def main(conf):
    tokenizer = get_tokenizer()
    vae = get_vae(**conf.model.vae)
    
    model = get_rudolph_model(**conf.model.rudolph, **conf.model.params)
    checkpoint = torch.load(conf.model.rudolph_weight, map_location='cpu')
    if (conf.model.rudolph_weight[-3:] == 'bin'):
        model.load_state_dict(checkpoint)
    else:
        convert_checkpoint = {k[6:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('model')}
        model.load_state_dict(convert_checkpoint)

    NUM_WORKERS = os.cpu_count()
    
    dataset_textqa = create_dataset('textqa', **conf.data.textqa) if conf.tasks.textqa == True else []
    dataset_mathqa = create_dataset('mathqa', **conf.data.mathqa) if conf.tasks.mathqa == True else []
    dataset_generation = create_dataset('generation', **conf.data.generation) if conf.tasks.generation == True else []
    dataset_captioning = create_dataset('captioning', **conf.data.captioning) if conf.tasks.captioning == True else []
    dataset_vqa = create_dataset('vqa', **conf.data.vqa) if conf.tasks.vqa == True else []
    dataset_text_recogn = create_dataset('text_recognition', **conf.data.text_recognition) if conf.tasks.text_recognition == True else []
    
    dataset = [*dataset_textqa, *dataset_mathqa, *dataset_generation, *dataset_captioning, *dataset_vqa, *dataset_text_recogn]
    df = pd.DataFrame(dataset)  
    
    df_train = df[df['stage'] == 'train']
    df_val = df[df['stage'] == 'val']
    
    wandb_logger = WandbLogger(project=conf.trainer.logger.model_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(**conf.trainer.checkpoints)

    trainer = pl.Trainer(
          logger=wandb_logger,
          callbacks=[lr_monitor, checkpoint_callback],
          **conf.trainer.pl_trainer
    )
    
    train_dataset = DatasetRetriever(
        task_ids=df_train['task_id'].values,
        left_texts=df_train['left_text'].values,
        image_paths=df_train['image_path'].values,
        right_texts=df_train['right_text'].values,
        stage='train',
        tokenizer=tokenizer,
        model_params = conf.model.params
    )
    
    train_sampler = DistributedSamplerWrapper(
           sampler=BalanceClassSampler(labels=train_dataset.get_task_labels()),
            num_replicas=conf.trainer.pl_trainer.gpus,
            rank=trainer.global_rank,
            shuffle=False
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.trainer.bs,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        collate_fn=fb_collate_fn
    )


    val_dataset = DatasetRetriever(
        task_ids=df_val['task_id'].values,
        left_texts=df_val['left_text'].values,
        image_paths=df_val['image_path'].values,
        right_texts=df_val['right_text'].values,
        stage='val',
        tokenizer=tokenizer,
        model_params = conf.model.params
    )

    val_sampler = DistributedSamplerWrapper(
            sampler=BalanceClassSampler(labels=val_dataset.get_task_labels()),
            num_replicas=conf.trainer.pl_trainer.gpus,
            rank=trainer.global_rank,
            shuffle=False
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.trainer.bs,
        sampler=val_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        collate_fn=fb_collate_fn
    )
    
    total_training_steps = int(len(train_dataloader) * conf.trainer.pl_trainer.max_epochs) #/  conf.trainer.pl_trainer.accumulate_grad_batches)

    rudolph_light = RudolphLightning(vae=vae, 
                                     model=model, 
                                     n_training_steps=total_training_steps, 
                                     task_weights=conf.trainer.task_weights,
                                     model_params=conf.model.params,
                                     model_freeze = conf.model.freeze,
                                     scheduler_conf=conf.trainer.scheduler,
                                     bs = conf.trainer.bs)

    trainer.fit(rudolph_light, train_dataloader, val_dataloader)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/fusion_train.yaml',type=str, help='config path')
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    main(conf)
