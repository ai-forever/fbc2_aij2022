from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import nltk
import wordtodigits as w2d
from metrics.calc_meteor import calc_meteor
from metrics.ruclip_score import calc_ruclip_metric
import json

    
def evaluate(args):
    gt_json = json.loads(open(args.ref_path, encoding="utf-8").read())
    pred_json = json.loads(open(args.pred_path, encoding="utf-8").read())
    gt_image_json = json.loads(open(args.gt_image_path, encoding="utf-8").read())
    image_folder = '/'.join(args.gt_image_path.split('/')[:-1]) + '/'
    
    quesIds = set(gt_json.keys())
    assert quesIds == set(pred_json.keys()), 'Not all images have predicted captions!'
    
    meteor = calc_meteor(gt_json, pred_json)
    print('meteor:', meteor)
    
    ruclip_score = calc_ruclip_metric(pred_json, gt_image_json, image_folder)
    print('ruCLIP score: ', ruclip_score)
    
    caption_metric = 1/2*(ruclip_score + meteor)
    print('caption metric: ', caption_metric)
    
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ref_path', type=str, default='../../data_for_inference/split/captioning/captioning_gt.json')
    parser.add_argument('--pred_path', type=str, default='../../data_for_inference/split/captioning/captioning_pred.json')
    parser.add_argument('--gt_image_path', type=str, default='../../data_for_inference/split/captioning/captioning_input.json')
    
    args = parser.parse_args()
    evaluate(args)