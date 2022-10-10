import collections
import json
import math
import re
import string
import pymorphy2
import nltk
import wordtodigits as w2d


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
morph = pymorphy2.MorphAnalyzer()


manualMap_ru = {'ноль': '0',
                 'нисколько': '0',
                 'никакой': '0',
                 'один': '1',
                 'два': '2',
                 'три': '3',
                 'четыре': '4',
                 'пять': '5',
                 'шесть': '6',
                 'семь': '7',
                 'восемь': '8',
                 'девять': '9',
                 'десять': '10'
                }

def get_lemma(word):
    word_parsed = morph.parse(word)[0]
    lemma = word_parsed.normal_form
    lemma = manualMap_ru.setdefault(lemma, lemma)
    return lemma

def normalize_answer(s):
    """Lower text and remove punctuation and extra whitespace."""

    #def remove_articles(text):
    #    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    #    return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())
    
    def normalize(text):
        return " ".join(get_lemma(word) for word in text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        exclude.discard('-')
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(normalize(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def get_exact(examples, preds):
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
        exact_score = max(compute_exact(a, prediction) for a in gold_answers)
        if (exact_score <= 0.5):
            label = re.search('[abcde]', prediction)
            if (label != None):
                prediction = prediction[label.span()[0]]
                exact_score = max(compute_exact(a, prediction) for a in gold_answers)
        print(exact_score, gold_answers, prediction)
        exact_scores[key] = exact_score
        
    return sum(exact_scores.values()) / len(examples.keys())