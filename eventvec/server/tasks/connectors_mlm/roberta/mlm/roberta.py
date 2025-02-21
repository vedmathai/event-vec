from transformers import AutoTokenizer, RobertaForMaskedLM
from collections import defaultdict
import torch
import csv
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class ConnectorMLMDatum:
    def __init__(self):
        self._text = None
        self._label = None

class ConnectorMLM:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
        self._model = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-large")

    def read_data(self):
        data = []
        with open('/home/lalady6977/oerc/projects/data/roc_masked_connectors.csv') as f:
            reader = csv.reader(f, delimiter='\t')
            for r in reader:
                datum = ConnectorMLMDatum()
                datum._label = r[1]
                datum._text = r[2].replace('[MASK]', self._tokenizer.mask_token)
                data.append(datum)
        return data

    def test_connector(self):
        data = self.read_data()
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        random.shuffle(data)

        predicted_labels = []
        required_answers = []

        for datum_i, datum in enumerate(data):
            inputs = self._tokenizer(datum._text, return_tensors="pt")

            with torch.no_grad():
                logits = self._model(**inputs).logits

            # retrieve index of <mask>
            mask_token_index = (inputs.input_ids == self._tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            predicted_token_ids = logits[0, mask_token_index][0]
            token_ids = []
            for token_id_i, token_id in enumerate(predicted_token_ids):
                token_ids.append((token_id_i, token_id))
            token_ids = sorted(token_ids, key=lambda x: x[1], reverse=True)
            ranked = []
            for token_id, token_score in token_ids[:20]:
                ranked.append(self._tokenizer.decode(token_id).strip().lower())

            required_answers.append(datum._label)

            
            seen = False
            for ranked_item in ranked:
                if ranked_item in ['and', 'so', 'because', 'though', 'but']:
                    seen = True
                    if ranked_item == datum._label:
                        tp[datum._label] += 1
                    if ranked_item != datum._label:
                        fp[ranked_item] += 1
                        fn[datum._label] += 1
                    predicted_labels.append(ranked_item)
                    break
            
            if seen is False:
                fn[datum._label] += 1
                predicted_labels.append('none')
                print(datum._text, datum._label, ranked)


                
            print(datum_i)
            if datum_i % 10 == 0 or datum_i == len(data) - 1:
                f1s = []
                for key in tp.keys():
                    f1 = 0

                    precision = 0
                    recall = 0
                    if tp[key] + fp[key] != 0:
                        precision = tp[key] / (tp[key] + fp[key])
                    if tp[key] + fn[key] != 0:
                        recall = tp[key] / (tp[key] + fn[key])
                    if precision + recall != 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    f1s.append(f1)
                print('-->', np.mean(f1s))

                classes = ['and', 'but', 'though', 'so', 'because', 'none']
                cm = confusion_matrix(required_answers, predicted_labels, labels=classes)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                disp.plot()
                plt.savefig('/home/lalady6977/Downloads/confusion_matrix_mlm_roberta.png', bbox_inches='tight')

if __name__ == '__main__':
    mlm = ConnectorMLM()
    mlm.test_connector()