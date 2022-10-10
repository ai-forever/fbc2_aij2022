from nltk.metrics.distance import edit_distance

def compute_NED(gt, pred):
    norm_ED = 0
    if len(gt) == 0 or len(pred) == 0:
        norm_ED += 0
    elif len(gt) > len(pred):
        norm_ED += 1 - edit_distance(pred, gt) / len(gt)
    else:
        norm_ED += 1 - edit_distance(pred, gt) / len(pred)
    return norm_ED

def get_NED(examples, preds):
    """
    Computes the f1 score from the examples (are the option of the correct answers) and the model predictions
    examples = {"0": [{"type": "text", "content": ["option_1", "option_2", ..., "option_n"]}, "1": ...}
    preds = {"0": [{"type": "text", "content": "answer_1"}, {"type": "image", "content": "link_to_the_image"}], "1": ...} 
    """
    exact_scores = {}

    for key in examples.keys():
        gold_answers = examples[key][0]['content']

        if key not in preds:
            print(f"Missing prediction for {key}")
            continue
        
        for element in preds[key]:
            
          prediction = ""
        
          if element['type'] == 'text':
            prediction = element['content']
            break
        exact_score = max(compute_NED(a, prediction) for a in gold_answers)
        print(exact_score, gold_answers, prediction)
        exact_scores[key] = exact_score
        
    return sum(exact_scores.values()) / len(examples.keys())