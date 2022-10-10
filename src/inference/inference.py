import torch
import torchvision
import transformers
import more_itertools
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm.auto import tqdm
from einops import rearrange
import torchvision.transforms as transforms
from src.rudolph import utils
from rudalle.utils import seed_everything
import re
import youtokentome as yttm
from src.inference.inference_api import ruDolphApi


def run_inference(test_dataloader, model, tokenizer, vae, save_image_path, bs):
    '''
        Generate image and text for every sample.
    '''
    pred_json = {} 
    vocab_size = model.get_param('vocab_size')
    image_tokens_per_dim = model.get_param('image_tokens_per_dim')
    l_text_seq_length = model.get_param('l_text_seq_length')
    r_text_seq_length = model.get_param('r_text_seq_length')
    image_seq_length = model.get_param('image_seq_length')
    device = model.get_param('device')
    spc_id = -1

    api = ruDolphApi(model, tokenizer, vae, bs=bs)
    
    vocab = tokenizer.tokenizer.vocab()
    allowed_token_ids = []
    for i, token in enumerate(vocab):
        allowed_token_ids.append(i)
   
    for batch in tqdm(test_dataloader):
        ids_question, left_text, images = batch
        left_text = left_text.to(device)
        images = images.to(device)
        image_tokens = vae.get_codebook_indices(images)  
        
        ## Generate images if image == torch.tensor([0,0,0...0])   
        image_tokens, idxs = api.generate_codebooks(images, image_tokens, left_text, vocab_size, top_k=1024, top_p=0.975, temperature=1.0)
        decode_images = vae.decode(image_tokens)
        
        ## Generate right texts
        texts = api.generate_tokens(image_tokens, left_text, vocab_size, 
                                    top_k=32, top_p=0.8, temperature=1.0, template = '', 
                                    allowed_token_ids = allowed_token_ids, special_token='<RT_UNK>')
            
        ## save predictions to dict
        for i in range(len(texts)):
            print('Ответ :', texts[i])
            pred_json[ids_question[i]] = [{'type': 'text', 'content': texts[i]}]       
            if i in idxs:
                path = save_image_path.split('/')[-2]
                pred_json[ids_question[i]].append({'type': 'image', 'content': f'{path}/image_{ids_question[i]}.jpg'})
                pil_image = transforms.ToPILImage()(decode_images[i])
                pil_image.save(f'{save_image_path}/image_{ids_question[i]}.jpg')
        
    return pred_json