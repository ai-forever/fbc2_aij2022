from metric.rumeteor import meteor_score
import numpy as np
from ruwordnet import RuWordNet
import pymorphy2
import nltk
import wordtodigits as w2d
import string


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
wn = RuWordNet()
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


def distance(x, y):
    if x == 0 and y == 0:
        return 1.0
    if x > y:
        x, y = y, x
    return x / y


def get_meteor(pred, true):
    hypothesis = nltk.word_tokenize(w2d.convert(pred))
    reference = [nltk.word_tokenize(pred) for pred in true]
    if (len(reference) == 1) and (len(reference[0]) == 1):    
        # reference = [['yes']] or [['5']] or [['black']]
        if (reference[0][0].isnumeric()):
            # find number in prediction and compare with GT
            tagging = nltk.pos_tag(hypothesis)
            for tok, pos in tagging:
                if pos == 'CD':
                    try:
                        return distance(float(tok), float(reference[0][0]))
                    except:
                        # nltk pos tagging is not perfect and sometimes mistakes words for digits
                        continue
            # if no number in prediction score = 0
            return 0
        else:
            return meteor_score(reference, hypothesis, wordnet=wn, gamma=0)
    else:
        return meteor_score(reference, hypothesis, wordnet=wn, gamma=0)


def get_lemma(word):
    word_parsed = morph.parse(word)[0]
    lemma = word_parsed.normal_form
    lemma = manualMap_ru.setdefault(lemma, lemma)
    return lemma


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude).lower()
    

def calc_meteor(true_json, pred_json):
    scores = []
    for key in pred_json:
        hyps = pred_json[key]
        meteor = 0
        ## обрабатываем только первый текст, если пришла картинка пропускаем ее
        for hyp in hyps:
            if hyp['type'] == 'text':
                ## приходит строка, берем ее
                hyp = hyp['content']
                print(hyp, true_json[key][0]['content'])
                hyp = get_lemma(remove_punc(hyp))
                for i in range(len(true_json[key][0]['content'])):
                    ## у gt несколько вариантов в списке, сравниваем с каждым
                    ref = true_json[key][0]['content'][i]
                    ref = get_lemma(remove_punc(ref))
                    meteor_new = get_meteor(hyp, [ref])
                    if meteor_new > meteor:
                        meteor = meteor_new
                break
        print(meteor)
        scores.append(meteor)
    return np.mean(np.array(scores))