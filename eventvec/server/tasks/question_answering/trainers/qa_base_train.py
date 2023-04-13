import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from collections import defaultdict


from eventvec.server.config import Config
from eventvec.server.data_handlers.qa_datahandlers.tsqa_datahandler.tsqa_datahandler import TSQADatahandler
from eventvec.server.data_handlers.qa_datahandlers.torque_datahandler.torque_datahandler import TorqueDatahandler
from eventvec.server.model.qa_models.torch_models.qa_descriminator import QuestionDescriminatorModel
from eventvec.server.model.qa_models.torch_models.qa_base import QuestionAnsweringBase
from eventvec.server.reporters.report_model.report_model import ReportModel
from eventvec.server.entry_points.time_sensitive_qa.time_sensitive_qa_generator import TSQANoiseGenerator
from eventvec.server.model.qa_models.datamodel.qa_datum import QADatum
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer



LEARNING_RATE = 2e-5
BATCH_SIZE = 32

tense_mapping = {
    "Pres": 0,
    "Past": 1,
    'Future': 2,
    None: 3,
}

future_modals = [
    'will',
    'going to',
    'would',
    'could',
    'might',
    'may',
    'can',
    'going to',
]

class QATrainBase:


    def load(self):
        self._data_handler = TorqueDatahandler()
        self._model = QuestionAnsweringBase()
        self._config = Config.instance()
        self._task_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.025, 1-.025]))
        self._tense_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.995, .99, .99, .014]))
        self._linguistic_featurizer = LinguisticFeaturizer()

        self._model_optimizer = Adam(
            self._model.parameters(),
            lr=LEARNING_RATE,
        )
        self.losses = []
        self.task_losses = []
        self._total_count = 0
        self._answer_count = 0
        self._eval_data = self._data_handler.qa_eval_data().data()
        self._train_data = self._data_handler.qa_train_data().data()
        self._featurized_context_cache = {}
        self._counter = defaultdict(int)


    def train(self):
        train_data = self._train_data
        for epochi in range(30):
            data_size = len(train_data)
            print('training started', data_size)
            for datumi, datum in enumerate(train_data):
                self._train_datum(datum)
                if (datumi + 1) % int(.1 * data_size) == 0:
                    print('epoch {}, train %: {}'.format(epochi, float(datumi) / data_size))
            self.eval()
            # print('ratio:', self._answer_count / self._total_count)
            # print(self._counter)


    def eval(self):
        eval_data = self._eval_data
        self._f1s = []
        self._exact_matches = []
        data_size = len(eval_data)
        for datumi, datum in enumerate(eval_data):
            self._infer_datum(datum)
        print('f1: {}, exact_match: {}'.format(np.mean(self._f1s), np.mean(self._exact_matches)))
        print('---' * 4)


    def _train_datum(self, qa_datum: QADatum):
        question = qa_datum.question()
        context = qa_datum.context()
        if context[0] not in self._featurized_context_cache:
            self._featurized_context_cache[context[0]] = self._linguistic_featurizer.featurize_document(context[0])
        featurized_context = self._featurized_context_cache[context[0]]
        context_i2token = {}
        for sentence in featurized_context.sentences():
            for token in sentence.tokens():
                context_i2token[token.i()] = token
        losses = []
        token_outputs, tense_answer = self._model(question, context[0])
        answer_bitmap = [[1, 0] for _ in token_outputs]
        tense_bitmap = [[0, 0, 0, 1] for _ in token_outputs]
        question_len = len(question.split())
        required_answer = []
        tokens = []
        self._total_count += len(answer_bitmap)
        for answer in qa_datum.answers():
            required_answer.append(answer.text())
            for paragraph_i, paragraph in enumerate(context):
                if paragraph_i == answer.paragraph_idx():
                    if answer.start_location() is not None and answer.end_location() is not None:
                        wordid2tokenid, tokens = self._model.wordid2tokenid(question, paragraph)
                        for i in range(answer.start_location(), answer.end_location()):
                            self._answer_count += 1
                            word_i = question_len + i
                            token_i = tokens.index('[SEP]')
                            while token_i < len(tokens) - 1:
                                if tokens[token_i].strip('▁').lower() == context_i2token[i].text().lower():
                                    break
                                token_i += 1
                            #print([(tokens[token_i+ i], i) for i in range(0, 1)], context_i2token[i].text())
                            if tokens[token_i].strip('▁').lower() == '[sep]':
                                return
                            f_token_tense = context_i2token[i].tense()
                            tense_map = [0, 0, 0, 0]
                            last_5 = [context_i2token[k].text().lower() for k in range((i-5), i) if k > 0]
                            last_5 = ' '.join(last_5)
                            if any(k in last_5 for k in future_modals):
                                tense_map[tense_mapping['Future']] = 1
                            else:
                                tense_map[tense_mapping[f_token_tense]] = 1
                            tense_bitmap[token_i] = tense_map
                            answer_bitmap[token_i] = [0, 1]
        for token_i, token in enumerate(tense_bitmap):
            self._counter[tuple(tense_bitmap[token_i])] += 1
        answer = []
        for token_i, token in enumerate(token_outputs):
            answer_tensor = torch.Tensor(answer_bitmap[token_i])
            tense_answer_tensor = torch.Tensor(tense_bitmap[token_i])
            if token[1] > token[0] and token_i < len(tokens):
                answer += [tokens[token_i]]
            loss = self._task_criterion(token, answer_tensor)
            tense_loss = self._tense_criterion(tense_answer[token_i], tense_answer_tensor)
            losses += [loss]# + tense_loss]
        self.task_losses.append(sum(i.item() for i in losses))
        #print(answer, '--->', required_answer)
        self.losses += [sum(losses)]
        losses = []
        if len(self.losses) >= BATCH_SIZE:
            sum(self.losses).backward()
            self._model_optimizer.step()
            self._model.zero_grad()
            self.losses = []
            #print('loss: ', np.mean(self.task_losses))

    def _infer_datum(self, qa_datum: QADatum):
        question = qa_datum.question()
        context = qa_datum.context()
        with torch.no_grad():
            token_outputs, tense_outputs = self._model(question, context[0])
        required_answer = []
        wordid2tokenid, tokens = self._model.wordid2tokenid(question, context[0])
        for answer in qa_datum.answers():
            required_answer.append(answer.text())
        answer = []
        token_i = 0
        while token_i < len(token_outputs):
            token = token_outputs[token_i]
            if token[1] > token[0] and token_i < len(tokens):
                answer_tokens = []
                answer_tokens += [tokens[token_i].strip('▁')]
                while token_i + 1 < len(tokens) and tokens[token_i + 1][0] not in '▁[' and tokens[token_i + 1].isalpha():
                    answer_tokens += [tokens[token_i+1].strip('▁')]
                    token_i += 1
                if len(answer_tokens) > 0:
                    answer += [''.join(answer_tokens)]
            token_i += 1
        #print(answer, '--->', required_answer)
        answer = set(answer)
        required_answer = set(required_answer)
        precision_list = []
        for i in answer:
            if i in required_answer:
                precision_list += [1]
            else:
                precision_list += [0]
        recall_list = []
        for i in required_answer:
            if i in answer:
                recall_list += [1]
            else:
                recall_list += [0]
        precision = 0
        if len(precision_list) > 0:
            precision = np.mean(precision_list)
        recall = 0
        if len(recall_list) > 0:
            recall = np.mean(recall_list)
        f1 = 0
        if len(answer) == len(required_answer) == 0:
            f1 = 1
        if len(precision_list) > 0 and len(recall_list) > 0 and precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        exact_match = 1 if answer == required_answer else 0
        self._f1s.append(f1)
        self._exact_matches.append(exact_match)



if __name__ == '__main__':
    qa_train = QATrainBase()
    qa_train.load()
    qa_train.train()
