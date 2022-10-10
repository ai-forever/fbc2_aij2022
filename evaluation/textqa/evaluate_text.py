import argparse
from metric.calculate_f1 import get_f1
import json


def read_json(file_name):
    with open(file_name) as f:
        result = json.load(f)
    return result


def evaluate(examples_file, preds_file):
    examples = read_json(examples_file)
    preds = read_json(preds_file)
    
    quesIds = set(examples.keys())
    assert quesIds == set(preds.keys()), 'Not all questions have predicted answers!'
        
    try:
        f1 = get_f1(examples, preds)
    except Exception:
        print('Unsupported data format')
        return 0
    return f1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default='../../data_for_inference/split/textqa/textqa_gt.json',
                        help="the path to the directory with the gold answers")
    parser.add_argument('--pred_path', type=str, default='../../data_for_inference/split/textqa/textqa_pred.json',
                        help="the path to the directory with the predictions")

    args = parser.parse_args()

    eval_results = evaluate(args.ref_path, args.pred_path)

    print('f1 score:', eval_results)