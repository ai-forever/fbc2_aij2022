import os
import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from metrics.fid import calculate_fid_given_paths
from metrics.inception import InceptionV3
from metrics.ruclip import calc_ruclip_metric
import json


def evaluate(args):
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)
     
    assert len(os.listdir(args.ref_path_images)) == len(os.listdir(args.pred_path_images)), 'Not all captions have predicted images!'
    
    ## calc fid
    fid_score = calculate_fid_given_paths(args.ref_path_images, args.pred_path_images,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    print('FID: ', fid_score)
    
    ref_json = json.loads(open(args.ref_path, encoding="utf-8").read())
    pred_json = json.loads(open(args.pred_path, encoding="utf-8").read())
    image_folder = '/'.join(args.pred_path.split('/')[:-1]) + '/'
    
    quesIds = set(ref_json.keys())
    assert quesIds == set(pred_json.keys()), 'Not all questions have predicted answers!'
    
    clip_score = calc_ruclip_metric(ref_json, pred_json, image_folder)
    print('ruCLIP score: ', clip_score)
    
    gen_score = 1/2 * (clip_score + (200 - min(200, fid_score))/200)
    print('generation metric: ', gen_score)

    
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--ref_path_images', type=str, default='../../data_for_inference/split/generation/output_images/')
    parser.add_argument('--pred_path_images', type=str, default='../../data_for_inference/split/generation/output_pr_images/')
    parser.add_argument('--ref_path', type=str, default='../../data_for_inference/split/generation/generation_input.json')
    parser.add_argument('--pred_path', type=str, default='../../data_for_inference/split/generation/generation_pred.json')
    
    args = parser.parse_args()
    evaluate(args)