import ruclip
import pandas as pd
from PIL import Image
import os
import torch
import numpy as np 
import cv2
import ast


device = 'cuda'
clip, processor = ruclip.load('ruclip-vit-large-patch14-336', device=device)
clip_predictor = ruclip.Predictor(clip, processor, device, bs=8, quiet=True)


def calc_ruclip_metric(caption_json, input_json, image_folder):    
    scores = []
    for key in caption_json:
        try:
            image_name = image_folder + input_json[key][1]['content'] if input_json[key][1]['type'] == 'image' else None
            img = Image.open(image_name)

            ## приходит список, берем первый элемент только
            text = caption_json[key][0]['content'] if caption_json[key][0]['type'] == 'text' else None
            with torch.no_grad():
                text_latents = clip_predictor.get_text_latents([text])
                image_latents = clip_predictor.get_image_latents([img])
            logits_per_image = torch.matmul(image_latents, text_latents.t())
            score = logits_per_image.view(-1)
#             print(score)
            scores.append(score[0].cpu().detach().numpy()) 
        except:
            scores.append(0.0)
                
    return np.array(scores).mean()
           