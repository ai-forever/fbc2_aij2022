import ruclip
import pandas as pd
from PIL import Image
import os
import torch
import numpy as np 
import cv2


device = 'cuda'
clip, processor = ruclip.load('ruclip-vit-large-patch14-336', device=device)
clip_predictor = ruclip.Predictor(clip, processor, device, bs=8, quiet=True)


def calc_ruclip_metric(ref_json, pred_json, image_folder):   
    scores = []
    for key in ref_json:
        try:
            image_name = image_folder + '/'.join(pred_json[key][1]['content'].split('/')[-2:]) if pred_json[key][1]['type'] == 'image' else None
            img = Image.open(image_name).convert('RGB')

            if len(ref_json[key]) == 2:
                text = ref_json[key][1]['content']
            elif len(ref_json[key]) == 1:
                text = '. '.join(ref_json[key][0]['content'].split('. ')[1:])

            text_latents = clip_predictor.get_text_latents([text])
            image_latents = clip_predictor.get_image_latents([img])
            logits_per_image = torch.matmul(image_latents, text_latents.t())
            score = logits_per_image.view(-1)
            print(score)
            scores.append(score[0].cpu().detach().numpy()) 
        except:
            scores.append(0.0)
        
    return np.array(scores).mean()