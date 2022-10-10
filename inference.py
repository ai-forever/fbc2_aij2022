from rudalle import get_tokenizer, get_vae
from src.rudolph.model import get_rudolph_model
from src.inference.utils import create_dataset
from src.inference.dataloader import DatasetRetriever
from src.inference.inference import run_inference
from torch.utils.data import DataLoader
import torch
from PIL import Image
import torchvision.transforms as T
from omegaconf import OmegaConf
import pandas as pd
import argparse 
import json
import os
import time
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def main(conf):
    tokenizer = get_tokenizer()
    vae = get_vae(dwt=conf.model.vae.dwt)
    vae = vae.to(conf.model.vae.device)
    
    model = get_rudolph_model(**conf.model.rudolph, **conf.model.params)
    checkpoint = torch.load(conf.model.rudolph_weight, map_location='cpu')
    if (conf.model.rudolph_weight[-3:] == 'bin'):
        model.load_state_dict(checkpoint)
    else:
        convert_checkpoint = {k[6:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('model')}
        model.load_state_dict(convert_checkpoint)
    model.eval()
    
    NUM_WORKERS = os.cpu_count()
    
    dataset = create_dataset(conf.data.input)
    dataset = pd.DataFrame(dataset)
     
    test_dataset = DatasetRetriever(
        ids = dataset['id'].values,
        left_text=dataset['left_text'].values,
        image_path = dataset['image_path'].values,
        tokenizer=tokenizer,
        model_params = conf.model.params,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf.model.bs,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=NUM_WORKERS
    )
    
    if (not os.path.exists(conf.data.pred_images_output)):
        os.mkdir(conf.data.pred_images_output)
    
    start = time.time()
    predictions = run_inference(test_dataloader, model, tokenizer, vae, conf.data.pred_images_output, conf.model.bs)
    print('time: ', time.time()-start)
    
    with open(conf.data.pred_output, 'w') as fp:
        json.dump(predictions, fp, ensure_ascii=False)
        
    
if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/fusion_inference.yaml',type=str, help='config path')
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    main(conf)