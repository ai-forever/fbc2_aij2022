from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
import shutil
import os


tasks = {'0': 'textqa', '1': 'mathqa', '2': 'generation', '3': 'captioning', '4': 'vqa', '5': 'text_recognition'}
split_results_input = {'textqa': {}, 'mathqa': {}, 'generation': {},  'captioning': {}, 'vqa': {}, 'text_recognition': {}}
split_results_output = {'textqa': {}, 'mathqa': {}, 'generation': {}, 'captioning': {}, 'vqa': {}, 'text_recognition': {}}
split_results_pred = {'textqa': {}, 'mathqa': {}, 'generation': {}, 'captioning': {}, 'vqa': {}, 'text_recognition': {}}
    
    
def main(args):
    task_ids = json.loads(open(args.task_ids_path, encoding="utf-8").read()) 
    task_input = json.loads(open(args.task_input_path, encoding="utf-8").read()) 
    task_output = json.loads(open(args.task_output_path, encoding="utf-8").read()) 
    task_pred = json.loads(open(args.task_pred_path, encoding="utf-8").read()) 
 
    for key in task_ids:
#         if str(task_ids[key]) in tasks:
        task_name = tasks[str(task_ids[key])]
        split_results_input[task_name][key] = task_input[key]
        split_results_output[task_name][key] = task_output[key]
        split_results_pred[task_name][key] = task_pred[key]

    ## create dirs with task names
    for key in split_results_input:
        os.mkdir(args.save_path + key + '/')
    
    ## save results for every task to json
    for key in split_results_input:
        with open(f'{args.save_path}{key}/{key}_input.json', 'w') as fp:
            json.dump(split_results_input[key], fp, ensure_ascii=False)  

    for key in split_results_output:
        with open(f'{args.save_path}{key}/{key}_gt.json', 'w') as fp:
            json.dump(split_results_output[key], fp, ensure_ascii=False)  

    for key in split_results_pred:
        with open(f'{args.save_path}{key}/{key}_pred.json', 'w') as fp:
            json.dump(split_results_pred[key], fp, ensure_ascii=False)  

    ## save images for every task
    split_results = [split_results_input, split_results_output, split_results_pred] 
    for split_result in split_results:
        for key in split_result:
            print(key)
            for idx in split_result[key]:
                for i in range(len(split_result[key][idx])):
                    if split_result[key][idx][i]['type'] == 'image':
                        img_path = '/'.join(split_result[key][idx][i]['content'].split('/')[-2:])
                        print(img_path)
                        src = args.init_path + img_path
                        dst = args.save_path + key + '/' + img_path
                        save_path = args.save_path + key + '/' + img_path.split('/')[0]
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        shutil.copy(src, dst)
                        break

                    
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task_ids_path', type=str, default='../data_for_inference/sample_test/ids.json')
    parser.add_argument('--task_input_path', type=str, default='../data_for_inference/sample_test/input.json')
    parser.add_argument('--task_output_path', type=str, default='../data_for_inference/sample_test/output.json')
    parser.add_argument('--task_pred_path', type=str, default='../data_for_inference/sample_test/predictions.json')
    parser.add_argument('--init_path', type=str, default='../data_for_inference/sample_test/')
    parser.add_argument('--save_path', type=str, default='../data_for_inference/sample_test/split/')
    args = parser.parse_args()
    main(args)