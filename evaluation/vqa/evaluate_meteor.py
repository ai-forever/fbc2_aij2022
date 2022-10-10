from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import nltk
import wordtodigits as w2d
from metric.calc_meteor import calc_meteor
import json

    
def evaluate(args):
    gt_json = json.loads(open(args.ref_path, encoding="utf-8").read())
    pred_json = json.loads(open(args.pred_path, encoding="utf-8").read())
    
    quesIds = set(gt_json.keys())
    assert quesIds == set(pred_json.keys()), 'Not all questions have predicted answers!'
    
    meteor = calc_meteor(gt_json, pred_json)
    print('meteor:', meteor)
    
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ref_path', type=str, default='../../data_for_inference/split/vqa/vqa_gt.json')
    parser.add_argument('--pred_path', type=str, default='../../data_for_inference/split/vqa/vqa_pred.json')
    
    args = parser.parse_args()
    evaluate(args)