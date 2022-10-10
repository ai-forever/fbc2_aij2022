import json


def create_dataset(input_json):
    input_data = json.loads(open(input_json, encoding="utf-8").read()) 
    dataset = []
    for key in list(input_data.keys()):
        image_path = None
        left_text = ''
        for i in range(len(input_data[key])):
            if input_data[key][i]['type'] == 'text':
                left_text += input_data[key][i]['content'] + ' '
            if input_data[key][i]['type'] == 'image':
                image_path = '/'.join(input_json.split('/')[:-1]) + '/' + input_data[key][i]['content']
        
        dataset.append({
                        'id': key,
                        'left_text': left_text,
                        'image_path': image_path,
                    })
    return dataset