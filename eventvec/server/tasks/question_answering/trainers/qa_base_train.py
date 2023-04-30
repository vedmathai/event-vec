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
from eventvec.server.tasks.question_answering.trainers.optimization import BertAdam
from eventvec.server.tasks.question_answering.trainers.question_clusterer import QuestionClusterer


LEARNING_RATE = 1e-5
BATCH_SIZE = 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tense_mapping = {
    "Pres": 0,
    "Past": 1,
    'Future': 2,
    None: 3,
}

tense_num = len(tense_mapping.keys())

pos_mapping = {
    "VERB": 0,
    "AUX": 1,
    "NOUN": 2,
    "OTHERS": 3,
    None: 4,
}

pos_num = len(pos_mapping.keys())

aspect_mapping = {
    "Perf": 0,
    "Prog": 1,
    None: 2,
}

aspect_num = len(aspect_mapping.keys())

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

sentence_breaks = {
    'bigbird': '[SEP]',
    'roberta': '</s>',
}

token_delimiters = {
    'bigbird': '▁',
    'roberta': 'Ġ',
}

class QATrainBase:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._datahandlers_registry = DatahandlersRegistry()
        self._models_registry = QuestionAnsweringModelsRegistry()
        self._aspects_counter = defaultdict(int)
        self._question_clusterer = QuestionClusterer()

    def load(self, run_config):
        datahandler_class = self._datahandlers_registry.get_datahandler(run_config.dataset())
        self._datahandler = datahandler_class()
        base_model_class = self._models_registry.get_model('qa_base')
        self._base_model = base_model_class(run_config)
        self._base_model.to(device)
        self._config = Config.instance()
        self._task_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.025, 1-.025])).to(device)
        self._question_event_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.025, 1-.025])).to(device)
        self._tense_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.995, .99, .99, .014])).to(device)
        self._aspect_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.99, .99, 0.14])).to(device)
        self._pos_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.99, .99, .99, .99, 0.14])).to(device)
        self._question_classification_criterion =  nn.CrossEntropyLoss()
        self._linguistic_featurizer = LinguisticFeaturizer()


        param_optimizer = list(self._base_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
        #self._base_model_optimizer = BertAdam(optimizer_grouped_parameters,
        #                     lr=1e-5,
        #                     warmup=0.1,
        #                     t_total=24000 * 10)
        
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

    def train(self, run_config):
        self._jade_logger.new_experiment()
        self._jade_logger.set_experiment_type('question_answering')
        self._jade_logger.set_total_epochs(run_config.epochs())
        for epoch_i in range(run_config.epochs()):
            print(epoch_i)
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
        token_classification_output, question_event_classification_output, tense_classification_output, aspect_classification_output, pos_classification_output, question_classification_output = self._base_model(question, context[0])
        answer_bitmap = [[1, 0] for _ in token_classification_output]
        question_event_bitmap = [[1, 0] for _ in token_classification_output]
        tense_bitmap = [[0, 0, 0, 1] for _ in token_classification_output]
        aspect_bitmap = [[0, 0, 1] for _ in token_classification_output]
        pos_bitmap = [[0, 0, 0, 0, 1] for _ in token_classification_output]
        cluster_bitmap = self._question_clusterer.cluster(question)
        cluster_bitmap = cluster_bitmap
        self._total_count += len(answer_bitmap)
        bitmaps = self._datum2bitmap(
            qa_datum, answer_bitmap, tense_bitmap, aspect_bitmap, pos_bitmap,
            question_event_bitmap, tokens, run_config)
        answer = []
        answer_bitmap = torch.Tensor(bitmaps['answer_bitmap']).to(device)
        tense_bitmap = torch.Tensor(bitmaps['tense_bitmap']).to(device)
        aspect_bitmap = torch.Tensor(bitmaps['aspect_bitmap']).to(device)
        pos_bitmap = torch.Tensor(bitmaps['pos_bitmap']).to(device)
        question_event_bitmap = torch.Tensor(bitmaps['question_event_bitmap']).to(device)
        question_classification_bitmap = torch.Tensor(cluster_bitmap).to(device).unsqueeze(0)
        loss = self._task_criterion(token_classification_output, answer_bitmap)
        question_event_loss = self._question_event_criterion(question_event_classification_output, question_event_bitmap)
        tense_loss = self._tense_criterion(tense_classification_output, tense_bitmap)
        aspect_loss = self._aspect_criterion(aspect_classification_output, aspect_bitmap)
        pos_loss = self._pos_criterion(pos_classification_output, pos_bitmap)
        question_classification_loss = self._question_classification_criterion(question_classification_output, question_classification_bitmap)

        for token_i, token in enumerate(token_classification_output):
            #answer_tensor = torch.Tensor(answer_bitmap[token_i]).to(device)
            #tense_answer_tensor = torch.Tensor(tense_bitmap[token_i]).to(device)
            if token[1] > token[0]:
                answer += [tokens[token_i]]
        given_loss = [loss]
        if run_config.use_tense() is True:
            given_loss += [loss] + [tense_loss]
        if run_config.use_question_event() is True:
            given_loss += [loss] + [question_event_loss]
        if run_config.use_aspect() is True:
            given_loss += [loss] + [aspect_loss]
        if run_config.use_part_of_speech() is True:
            given_loss += [loss] + [pos_loss]
        if run_config.use_question_classification() is True:
            given_loss += [loss] + [question_classification_loss]
        losses += [sum(given_loss) / len(given_loss)]
        answer = self._tokens2words(answer, run_config)
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

    def _datum2bitmap(self, qa_datum, answer_bitmap, tense_bitmap, aspect_bitmap, pos_bitmap, question_event_bitmap, tokens, run_config):
        token_indices = []
        token2tense = {}
        token2aspect = {}
        token2pos = {}
        required_answer = []
        context = qa_datum.context()
        if context[0] not in self._featurized_context_cache:
            self._featurized_context_cache[context[0]] = self._linguistic_featurizer.featurize_document(context[0])
        featurized_context = self._featurized_context_cache[context[0]]
        context_i2token = {}
        for sentence in featurized_context.sentences():
            for token in sentence.tokens():
                context_i2token[token.idx()] = token
        token2featurized_token = self._token2featurized_tokens(tokens, featurized_context, run_config)
        for answer in qa_datum.answers():
            required_answer.append(answer.text())
            for paragraph_i, paragraph in enumerate(context):
                if paragraph_i == answer.paragraph_idx():
                    if answer.start_location() is not None and answer.end_location() is not None:
                        answer_token_indices = self._align_answer(run_config.llm(), tokens, context, answer.start_location(), answer.end_location())
                        token_indices.extend(answer_token_indices) 
                        token = context_i2token.get(answer.start_location())
                        for index in answer_token_indices:
                            token2tense[index] = token.tense() if token is not None else None
                            if any(future_modal in paragraph[max(0, answer.start_location() - 20): answer.start_location()].lower() for future_modal in future_modals):
                                token2tense[index] = 'Future'
                            token2aspect[index] = token.aspect() if token is not None else None
                            token2pos[index] = token.pos() if token is not None else None
        sentence_break = sentence_breaks.get(run_config.llm())
        start = tokens.index(sentence_break) + 2
        for token_i in range(start, len(tokens) - 1):
            if token_i not in token2featurized_token or not ((run_config.use_question_event() is True and token2featurized_token[token_i].text().lower() in qa_datum.question_events()) or token_i in token_indices):
                continue
            featurized_token = token2featurized_token[token_i]
            original_featurized = featurized_token
            if run_config.use_root_verb() is True:
                while not (featurized_token.pos() in ['AUX', 'VERB'] or featurized_token.dep() == 'ROOT'):
                    featurized_token = featurized_token.parent()
            if featurized_token.text().lower() in qa_datum.question_events():
                question_event_bitmap[token_i] = [0, 1]
            tense_array = [0] * tense_num
            aspect_array = [0] * aspect_num
            pos_array = [0] * pos_num
            tense = featurized_token.tense()
            if featurized_token.pos() in ['VERB', 'AUX']:
                if any(future_modal in context[0][max(0, featurized_token.idx() - 20): featurized_token.idx()].lower() for future_modal in future_modals):
                    tense = 'Future'
            tense_array[tense_mapping[tense]] = 1
            tense_bitmap[token_i] = tense_array
            aspect = featurized_token.aspect()
            aspect_array[aspect_mapping[aspect]] = 1
            aspect_bitmap[token_i] = aspect_array
            pos = featurized_token.pos()
            mapped_pos = pos_mapping.get(pos, pos_mapping['OTHERS'])
            pos_array[mapped_pos] = 1
            pos_bitmap[token_i] = pos_array
        for index in token_indices:
            answer_bitmap[index] = [0, 1]
        bitmaps = {
            'required_answer': required_answer,
            'answer_bitmap': answer_bitmap,
            'tense_bitmap': tense_bitmap,
            'aspect_bitmap': aspect_bitmap,
            'pos_bitmap': pos_bitmap,
            'question_event_bitmap': question_event_bitmap,
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
        return all_words
                
    def _infer_datum(self, run_config, qa_datum_i, qa_datum: QADatum):
        question = qa_datum.question()
        context = qa_datum.context()
        with torch.no_grad():
            token_classification_output, question_event_classification_output, tense_classification_output, aspect_classification_output, pos_classification_output, question_classification_output = self._base_model(question, context[0])
            answer_bitmap = [[1, 0] for _ in token_classification_output]
            question_event_bitmap = [[1, 0] for _ in token_classification_output]
            tense_bitmap = [[0, 0, 0, 1] for _ in token_classification_output]
            aspect_bitmap = [[0, 0, 1] for _ in token_classification_output]
            pos_bitmap = [[0, 0, 0, 0, 1] for _ in token_classification_output]
            required_answer = []
            wordid2tokenid, tokens = self._base_model.wordid2tokenid(question, context[0])
            bitmaps = self._datum2bitmap(
                qa_datum, answer_bitmap, tense_bitmap, aspect_bitmap, pos_bitmap,
                question_event_bitmap, tokens, run_config
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
            best_required_answer = self._find_best_required_answer(answer, qa_datum)
            self._jade_logger.new_evaluate_datapoint(best_required_answer, answer, loss_item_mean, {"question": question})

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
