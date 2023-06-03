import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from collections import defaultdict
from jadelogs import JadeLogger
import random


from eventvec.server.config import Config
from eventvec.server.datamodels.qa_datamodels.qa_datum import QADatum
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.tasks.question_answering.datahandlers.datahanders_registry import DatahandlersRegistry
from eventvec.server.tasks.question_answering.models.registry import QuestionAnsweringModelsRegistry


LEARNING_RATE = 1e-5
BATCH_SIZE = 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence_breaks = {
    'bigbird': '[SEP]',
    'roberta': '</s>',
}

token_delimiters = {
    'bigbird': '▁',
    'roberta': 'Ġ',
}

class QAReinforceTrain:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._datahandlers_registry = DatahandlersRegistry()
        self._models_registry = QuestionAnsweringModelsRegistry()

    def load(self, run_config):
        datahandler_class = self._datahandlers_registry.get_datahandler(run_config.dataset())
        self._datahandler = datahandler_class()
        base_model_class = self._models_registry.get_model('qa_base')
        self._base_model = base_model_class(run_config)
        self._base_model.to(device)
        self._config = Config.instance()
        self._task_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.025, 1-.025])).to(device)
        self._linguistic_featurizer = LinguisticFeaturizer()


        self._base_model_optimizer = Adam(
            self._base_model.parameters(),
            lr=LEARNING_RATE,
        )

        self.losses = []
        self.task_losses = []
        self._eval_data = self._datahandler.qa_eval_data().data()
        self._train_data = self._datahandler.qa_train_data().data()
        self._eval_data = self._train_data[int(len(self._train_data) * 0.8):] + self._eval_data
        self._train_data = self._train_data[:int(0.8 * len(self._train_data))]
        self._featurized_context_cache = {}

    def train(self, run_config):
        self._jade_logger.new_experiment()
        self._jade_logger.set_experiment_type('question_answering')
        self._jade_logger.set_total_epochs(run_config.epochs())
        for epoch_i in range(1, run_config.epochs()):
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
            self._train_datum(run_config, epoch_i, datum_i, datum)

    def _eval_epoch(self, epoch_i, run_config):
        eval_data = self._eval_data
        self._f1s = []
        self._exact_matches = []
        self._jade_logger.new_evaluate_batch()
        for datum_i, datum in enumerate(eval_data):
            self._infer_datum(run_config, datum_i, datum)

    def _train_datum(self, run_config, epoch_i, datum_i, qa_datum: QADatum):
        losses = []
        if qa_datum.use_in_eval() is True:
            return
        question = qa_datum.question()
        context = qa_datum.context()
        wordid2tokenid, tokens = self._base_model.wordid2tokenid(question, context[0])
        token_classification_output = self._base_model(question, context[0])[0]
        answer_bitmap = [[1, 0] for _ in token_classification_output]
        bitmaps = self._datum2bitmap(
            qa_datum, answer_bitmap, tokens, run_config)
        answer = []
        answer_bitmap = torch.Tensor(bitmaps['answer_bitmap']).to(device)

        with torch.inference_mode():
            for token_i, token in enumerate(token_classification_output):
                if token[1] > token[0]:
                    answer += [tokens[token_i]]
            penalize_list = []
            for answer_token in random.sample(answer, min(len(answer), 4)):
                new_answer = []
                after_template = "What event happened after {}".format(answer_token.strip(token_delimiters[run_config.llm()]))
                before_template = "What event happened before {}".format(answer_token.strip(token_delimiters[run_config.llm()]))
                if 'after' in question:
                    new_question = before_template
                    _, new_tokens = self._base_model.wordid2tokenid(new_question, context[0])
                    new_token_classification_output = self._base_model(new_question, context[0])[0]
                    for token_i, token in enumerate(new_token_classification_output):
                        if token[1] > token[0]:
                            new_answer += [new_tokens[token_i]]
                if 'before' in question:
                    new_question = after_template
                    _, new_tokens = self._base_model.wordid2tokenid(new_question, context[0])
                    new_token_classification_output = self._base_model(new_question, context[0])[0]
                    for token_i, token in enumerate(new_token_classification_output):
                        if token[1] > token[0]:
                            new_answer += [new_tokens[token_i]]
                new_answer = self._tokens2words(new_answer, run_config)
                if any(i not in new_answer for i in qa_datum.question_events()):
                    penalize_list.append(answer_token)
                new_answer = []
                if 'after' in question:
                    new_question = after_template
                    _, new_tokens = self._base_model.wordid2tokenid(new_question, context[0])
                    new_token_classification_output = self._base_model(new_question, context[0])[0]
                    for token_i, token in enumerate(new_token_classification_output):
                        if token[1] > token[0]:
                            new_answer += [new_tokens[token_i]]
                if 'before' in question:
                    new_question = before_template
                    _, new_tokens = self._base_model.wordid2tokenid(new_question, context[0])
                    new_token_classification_output = self._base_model(new_question, context[0])[0]
                    for token_i, token in enumerate(new_token_classification_output):
                        if token[1] > token[0]:
                            new_answer += [new_tokens[token_i]]
                new_answer = self._tokens2words(new_answer, run_config)
                answer_words = self._tokens2words(answer, run_config)
                if any(i not in answer_words for i in new_answer):
                    penalize_list.append(answer_token)


        mini_loss = []
        for token_i, token in enumerate(token_classification_output):
            loss = self._task_criterion(token_classification_output[token_i].unsqueeze(0), answer_bitmap[token_i].unsqueeze(0))
            if token[1] > token[0] and tokens[token_i] in penalize_list:
                mini_loss.append(1000 * loss)
            else:
                mini_loss.append(loss)     

        answer = self._tokens2words(answer, run_config)
        loss = sum(mini_loss)
        given_loss = [loss]
        losses += [sum(given_loss) / len(given_loss)]
        loss_item_mean = np.mean([l.item() for l in losses])
        self._jade_logger.new_train_datapoint(bitmaps['required_answer'], answer, loss_item_mean, {"question": question})
        self.losses += [sum(losses)]
        losses = []
        if len(self.losses) >= BATCH_SIZE:
            (sum(self.losses)/BATCH_SIZE).backward()
            self._base_model_optimizer.step()
            self._base_model.zero_grad()
            self.losses = []
            self._jade_logger.new_train_batch()

    def _datum2bitmap(self, qa_datum, answer_bitmap, tokens, run_config):
        token_indices = []
        required_answer = []
        context = qa_datum.context()
        for answer in qa_datum.answers():
            required_answer.append(answer.text())
            for paragraph_i, paragraph in enumerate(context):
                if paragraph_i == answer.paragraph_idx():
                    if answer.start_location() is not None and answer.end_location() is not None:
                        answer_token_indices = self._align_answer(run_config.llm(), tokens, context, answer.start_location(), answer.end_location())
                        token_indices.extend(answer_token_indices)
        for index in token_indices:
            answer_bitmap[index] = [0, 1]
        bitmaps = {
            'required_answer': required_answer,
            'answer_bitmap': answer_bitmap,
        }
        return bitmaps

    def _align_answer(self, llm, tokens, paragraph, start_index, end_index):
        if llm == 'bigbird':
            token_indices = self._align_answer_bigbird(llm, tokens, paragraph, start_index, end_index)
        if llm == 'roberta':
            token_indices = self._align_answer_roberta(llm, tokens, paragraph, start_index, end_index)
        return token_indices

    def _align_answer_bigbird(self, llm, tokens, paragraph, start_index, end_index):
        answer = paragraph[0][start_index: end_index]
        token_delimiter = token_delimiters.get(llm)
        token_i = tokens.index(sentence_breaks.get(llm))
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
    
    def _align_answer_roberta(self, llm, tokens, paragraph, start_index, end_index):
        answer = paragraph[0][start_index: end_index]
        token_delimiter = token_delimiters.get(llm)
        sentence_break = sentence_breaks.get(llm)
        token_i = tokens.index(sentence_break)
        summed_token_indices = []
        all_answer_indices = []
        summed_token = ""
        while token_i < len(tokens):
            token = tokens[token_i]
            if token[0] == token_delimiter :
                summed_token = token[1:]
                summed_token_indices = [token_i]
            if token[0] != token_delimiter and token not in [',', '.', '?', ':', ';', sentence_break]:
                summed_token += token
                summed_token_indices.append(token_i)
            if summed_token.lower() == answer.lower():
                all_answer_indices.extend(summed_token_indices)
            token_i += 1
        return all_answer_indices
    
    def _token2featurized_tokens(self, tokens, featurized_document, run_config):
        token_delimiter = token_delimiters.get(run_config.llm())
        sentence_break = sentence_breaks.get(run_config.llm())
        index = tokens.index(sentence_break) + 2
        featurized_tokens = []
        token2ftoken = {}
        for sentence in featurized_document.sentences():
            for token in sentence.tokens():
                featurized_tokens.append(token)
        ftoken_index = 0
        acc_indices = []
        acc = ""
        while index < len(tokens) and ftoken_index < len(featurized_tokens):
            current_ftoken = featurized_tokens[ftoken_index]
            current_ftoken_text = current_ftoken.text().lower()
            current_token = tokens[index].strip(token_delimiter).lower()
            if current_token != current_ftoken_text[len(acc):len(current_token) + len(acc)]:
                ftoken_index += 1
                continue
            else:
                acc_indices += [index]
                acc += current_token
            if acc == current_ftoken_text:
                for i in acc_indices:
                    token2ftoken[i] = current_ftoken
                acc = ""
                acc_indices = []
            index += 1
        return token2ftoken

    def _tokens2words(self, tokens, run_config):
        token_delimiter = token_delimiters.get(run_config.llm())
        sentence_break = sentence_breaks.get(run_config.llm())
        all_words = []
        current_word = []
        for token in tokens:
            if token[0] == token_delimiter:
                if len(current_word) > 0:
                    all_words.append(''.join(current_word))
                current_word = [token[1:]]
            elif token not in [sentence_break]:
                current_word += [token]
        if len(current_word) > 0:
            all_words.append(''.join(current_word))
        return all_words
                
    def _infer_datum(self, run_config, qa_datum_i, qa_datum: QADatum):
        question = qa_datum.question()
        context = qa_datum.context()
        if qa_datum.use_in_eval() is True:
            return
        with torch.no_grad():
            token_classification_output = self._base_model(question, context[0])[0]
            answer_bitmap = [[1, 0] for _ in token_classification_output]
            required_answer = []
            wordid2tokenid, tokens = self._base_model.wordid2tokenid(question, context[0])
            bitmaps = self._datum2bitmap(
                qa_datum, answer_bitmap, tokens, run_config
            )
            answer = []
            token_i = 0
            losses = []
            answer_tensor = torch.Tensor(answer_bitmap).to(device)
            loss = self._task_criterion(token_classification_output, answer_tensor)
            losses.append(loss)
            for token_i, token in enumerate(token_classification_output):
                if token[1] > token[0]:
                    answer += [tokens[token_i]]
            answer = self._tokens2words(answer, run_config)
            loss_item_mean = np.mean([l.item() for l in losses])
            if run_config.use_best_of_annotators() is True:
                best_required_answer = self._find_best_required_answer(answer, qa_datum)
            else:
                best_required_answer = self._voted_required_answer(qa_datum)
            self._jade_logger.new_evaluate_datapoint(best_required_answer, answer, loss_item_mean, {"question": question})

    def _voted_required_answer(self, qa_datum):
        expected_label = []
        for answer in qa_datum.answers():
            expected_label.append(answer.text())
        return expected_label

    def _find_best_required_answer(self, predicted_answer, qa_datum):
        best_f1 = -0.00001
        best_answer = []
        f1s = []
        for alternate_answer_set in qa_datum.alternate_answer_sets():
            expected_label = []
            for answer in alternate_answer_set:
                expected_label.append(answer.text())
            score = self._answers_f1_score(predicted_answer, expected_label)
            f1 = score['f1']
            f1s.append(f1)
            if f1 > best_f1:
                best_f1 = f1
                best_answer = expected_label
        return best_answer
        
    def _answers_f1_score(self, predicted_label, expected_label):
        precisions = []
        recalls = []
        f1s = []
        exact_matches = []
        precision_list = []
        recall_list = []
        for pred in predicted_label:
            if pred in expected_label:
                precision_list += [1]
            else:
                precision_list += [0]
        for expected in expected_label:
            if expected in predicted_label:
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
        if len(predicted_label) == 0:
            precision = 1
        if len(expected_label) == 0:
            recall = 1
        if len(predicted_label) == len(expected_label) == 0:
            f1 = 1
        elif (precision + recall) != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        precisions += [precision]
        recalls += [recall]
        f1s += [f1]
        exact_matches += [1 if expected_label == predicted_label else 0]
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match': np.mean(exact_matches),
        }


if __name__ == '__main__':
    qa_train = QATrainBase()
    qa_train.load()
    qa_train.train()
