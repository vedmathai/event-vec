import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from collections import defaultdict
from jadelogs import JadeLogger


from eventvec.server.config import Config
from eventvec.server.datamodels.qa_datamodels.qa_datum import QADatum
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.tasks.question_answering.datahandlers.datahanders_registry import DatahandlersRegistry
from eventvec.server.tasks.question_answering.models.registry import QuestionAnsweringModelsRegistry


LEARNING_RATE = 2e-5
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._datahandlers_registry = DatahandlersRegistry()
        self._models_registry = QuestionAnsweringModelsRegistry()

    def load(self, run_config):
        datahandler_class = self._datahandlers_registry.get_datahandler(run_config.dataset())
        self._datahandler = datahandler_class()
        base_model_class = self._models_registry.get_model('qa_base')
        self._base_model = base_model_class()
        self._base_model.to(device)
        self._config = Config.instance()
        self._task_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.025, 1-.025]))
        self._tense_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.995, .99, .99, .014]))
        self._linguistic_featurizer = LinguisticFeaturizer()

        self._base_model_optimizer = Adam(
            self._base_model.parameters(),
            lr=LEARNING_RATE,
        )
        self.losses = []
        self.task_losses = []
        self._total_count = 0
        self._answer_count = 0
        self._eval_data = self._datahandler.qa_eval_data().data()
        self._train_data = self._datahandler.qa_train_data().data()
        self._featurized_context_cache = {}
        self._counter = defaultdict(int)

    def train(self, run_config):
        self._jade_logger.new_experiment()
        self._jade_logger.set_experiment_type('question_answering')
        self._jade_logger.set_total_epochs(run_config.epochs())
        for epoch_i in range(run_config.epochs()):
            self._jade_logger.new_epoch()
            self._train_epoch(epoch_i, run_config)
            self._eval_epoch(epoch_i, run_config)
            
    def _train_epoch(self, epoch_i, run_config):
        jadelogger_epoch = self._jade_logger.current_epoch()
        train_data = self._train_data
        data_size = len(train_data)
        jadelogger_epoch.set_size(data_size)
        self._jade_logger.new_train_batch()
        for datum_i, datum in enumerate(train_data):
            self._train_datum(run_config, datum_i, datum)

    def _eval_epoch(self, epoch_i, run_config):
        eval_data = self._eval_data
        self._f1s = []
        self._exact_matches = []
        self._jade_logger.new_evaluate_batch()
        for datum_i, datum in enumerate(eval_data):
            self._infer_datum(run_config, datum_i, datum)

    def _train_datum(self, run_config, datum_i, qa_datum: QADatum):
        losses = []
        question = qa_datum.question()
        context = qa_datum.context()
        wordid2tokenid, tokens = self._base_model.wordid2tokenid(question, context[0])
        token_outputs, tense_answer = self._base_model(question, context[0])
        answer_bitmap = [[1, 0] for _ in token_outputs]
        tense_bitmap = [[0, 0, 0, 1] for _ in token_outputs]
        self._total_count += len(answer_bitmap)
        required_answer, answer_bitmap = self._datum2bitmap(qa_datum, answer_bitmap, tokens, run_config)
        answer = []
        for token_i, token in enumerate(token_outputs):
            answer_tensor = torch.Tensor(answer_bitmap[token_i], device=device)
            tense_answer_tensor = torch.Tensor(tense_bitmap[token_i], device=device)
            if token[1] > token[0]:
                answer += [tokens[token_i]]
            loss = self._task_criterion(token, answer_tensor)
            tense_loss = self._tense_criterion(tense_answer[token_i], tense_answer_tensor)
            losses += [loss]
        answer = self._tokens2words(answer)
        loss_item_mean = np.mean([l.item() for l in losses])
        self._jade_logger.new_train_datapoint(required_answer, answer, loss_item_mean, {"question": question})
        self.losses += [sum(losses)]
        losses = []
        if len(self.losses) >= BATCH_SIZE:
            sum(self.losses).backward()
            self._base_model_optimizer.step()
            self._base_model.zero_grad()
            self.losses = []
            self._jade_logger.new_train_batch()

    def _datum2bitmap(self, qa_datum, answer_bitmap, tokens, run_config):
        token_indices = []
        required_answer = []
        context = qa_datum.context()
        if context[0] not in self._featurized_context_cache:
            self._featurized_context_cache[context[0]] = self._linguistic_featurizer.featurize_document(context[0])
        featurized_context = self._featurized_context_cache[context[0]]
        context_i2token = {}
        for sentence in featurized_context.sentences():
            for token in sentence.tokens():
                context_i2token[token.i()] = token
        for answer in qa_datum.answers():
            required_answer.append(answer.text())
            for paragraph_i, paragraph in enumerate(context):
                if paragraph_i == answer.paragraph_idx():
                    if answer.start_location() is not None and answer.end_location() is not None:
                        answer_token_indices = self._align_answer(run_config.llm(), tokens, context, answer.start_location(), answer.end_location())
                        token_indices.extend(answer_token_indices)
        for index in token_indices:
            answer_bitmap[index] = [0, 1]
        return required_answer, answer_bitmap

    def _align_answer(self, llm, tokens, paragraph, start_index, end_index):
        if llm == 'bigbird':
            token_indices = self._align_answer_bigbird(llm, tokens, paragraph, start_index, end_index)
        if llm == 'roberta':
            token_indices = self._align_answer_roberta(llm, tokens, paragraph, start_index, end_index)
        return token_indices

    def _align_answer_bigbird(self, llm, tokens, paragraph, start_index, end_index):
        answer = paragraph[0][start_index: end_index]
        token_delimiter = '▁'
        token_i = tokens.index('[SEP]')
        summed_token_indices = []
        all_answer_indices = []
        summed_token = ""
        while token_i < len(tokens):
            token = tokens[token_i]
            if token[0] == token_delimiter:
                summed_token = token[1:]
                summed_token_indices = [token_i]
            if token[0] != token_delimiter:
                summed_token += token
                summed_token_indices.append(token_i)
            if summed_token.lower() == answer.lower():
                all_answer_indices.extend(summed_token_indices)
            token_i += 1
        return all_answer_indices
    
    def _tokens2words(self, tokens):
        all_words = []
        current_word = []
        for token in tokens:
            if token[0] == '▁':
                if len(current_word) > 0:
                    all_words.append(''.join(current_word))
                current_word = [token[1:]]
            else:
                current_word += [token]
        return all_words
                
    def _tense_featurizer(self):
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

    def _infer_datum(self, run_config, qa_datum_i, qa_datum: QADatum):
        question = qa_datum.question()
        context = qa_datum.context()
        with torch.no_grad():
            token_outputs, tense_outputs = self._base_model(question, context[0])
            answer_bitmap = [[1, 0] for _ in token_outputs]
            required_answer = []
            wordid2tokenid, tokens = self._base_model.wordid2tokenid(question, context[0])
            required_answer, answer_bitmap = self._datum2bitmap(qa_datum, answer_bitmap, tokens, run_config)
            answer = []
            token_i = 0
            losses = []
            while token_i < len(token_outputs):
                token = token_outputs[token_i]
                answer_tensor = torch.Tensor(answer_bitmap[token_i], device=device)
                loss = self._task_criterion(token, answer_tensor)
                losses.append(loss)
                if token[1] > token[0] and token_i < len(tokens):
                    answer_tokens = []
                    answer_tokens += [tokens[token_i]]
                    answer += self._tokens2words(answer_tokens)
                token_i += 1
            loss_item_mean = np.mean([l.item() for l in losses])
            self._jade_logger.new_evaluate_datapoint(required_answer, answer, loss_item_mean, {"question": question})


if __name__ == '__main__':
    qa_train = QATrainBase()
    qa_train.load()
    qa_train.train()
