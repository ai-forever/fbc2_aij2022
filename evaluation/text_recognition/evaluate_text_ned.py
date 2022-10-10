import argparse
from metric.calculate_NED import get_NED
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
    
    exact = get_NED(examples, preds)
    return exact


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default='../../data_for_inference/split/titw/titw_gt.json',
                        help="the path to the directory with the gold answers")
    parser.add_argument('--pred_path', type=str, default='../../data_for_inference/split/titw/titw_pred.json',
                        help="the path to the directory with the predictions")

    args = parser.parse_args()

    eval_results = evaluate(args.ref_path, args.pred_path)

    print('Exact NED', eval_results)