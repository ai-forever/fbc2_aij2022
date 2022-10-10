import torchvision.transforms as transforms
from PIL import Image
import json
import random
import pandas as pd
import os
import tarfile
import csv
import glob
import re
import string
import io


def create_dataset(task_id, dataset_path, train_input, train_output, val_input, val_output):
    data = {'train': [json.loads(open(train_input, encoding="utf-8").read()),
                      json.loads(open(train_output, encoding="utf-8").read())], 
            'val': [json.loads(open(val_input, encoding="utf-8").read()), 
                    json.loads(open(val_output, encoding="utf-8").read())]}
    
    dataset = []
    for stage in data:
        input_data = data[stage][0]
        output_data = data[stage][1]
        for key in input_data:
            left_text = []
            image_path = None
            right_text = None
            
            for i in range(len(input_data[key])):
                if input_data[key][i]['type'] == 'text':
                    left_text.append(input_data[key][i]['content'])
                if input_data[key][i]['type'] == 'image':
                    image_path = dataset_path + input_data[key][i]['content']
            
            for j in range(len(output_data[key])):
                if output_data[key][j]['type'] == 'text':
                    if isinstance(output_data[key][j]['content'], str):
                        right_text = output_data[key][j]['content']
                    else:
                        right_text = output_data[key][j]['content'][0]
                if output_data[key][j]['type'] == 'image':
                    image_path = dataset_path + output_data[key][j]['content']  
            
            left_text = ' '.join(left_text)

            dataset.append({
                'task_id': task_id,  
                'left_text': left_text,
                'image_path': image_path,
                'right_text': right_text,
                'stage': stage,
            })
            
    return dataset