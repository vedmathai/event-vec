import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer


from eventvec.server.config import Config
from eventvec.server.data_handlers.qa_datahandlers.tsqa_datahandler.tsqa_datahandler import TSQADatahandler
from eventvec.server.model.qa_models.torch_models.qa_descriminator import QuestionDescriminatorModel
from eventvec.server.model.qa_models.torch_models.qa_base import QuestionAnsweringBase
from eventvec.server.reporter.report_model.report_model import ReportModel
from eventvec.server.train.time_sensitive_qa.time_sensitive_qa_generator import TSQANoiseGenerator

LEARNING_RATE = 5e-5


class QATrain:
    
    def load(self):
        self._data_handler = TSQADatahandler()
        self._model = QuestionAnsweringBase()
        self._config = Config.instance()
        self._task_criterion = nn.CrossEntropyLoss()
        self._discriminator_criterion = nn.CrossEntropyLoss()

        self._model_optimizer = Adam(
            self._model.parameters(),
            lr=LEARNING_RATE,
        )
        self.task_losses = []
        self.discriminator_losses = []
        self._tsqa_noise_generator = TSQANoiseGenerator()

    def train(self):
        for epochi in range(5):
            data = self._data_handler.qa_data().data()
            data_size = len(data)
            for datumi, datum in enumerate(data):
                print(float(datumi) / data_size)
                self._train_datum(datum)

    def _train_datum(self, qa_datum):
        question = qa_datum.question()
        context = qa_datum.context()

        outputs = []
        for answer in qa_datum.answers():
            for paragraph_i, paragraph in enumerate(context):
                if paragraph_i == answer.paragraph_idx():
                    if answer.start_location() is not None and answer.end_location() is not None:
                        discriminator_label, parapraph = self.noisify_context(question, paragraph, answer.start_location(), answer.end_location())
                        start_output, end_output, discriminator_output = self._model(question, paragraph)
                        start_token_idx = torch.argmax(start_output[0])
                        end_token_idx = torch.argmax(end_output[0])
                        wordid2tokenid = self._model.wordid2tokenid(question, paragraph)
                        question_len = len(question.split())
                        start_token_idx = wordid2tokenid[question_len + answer.start_location()]
                        end_token_idx = wordid2tokenid[question_len + answer.end_location()]
                        start_label_vector = [0] * start_output.shape[1]
                        start_label_vector[start_token_idx] = 1
                        end_label_vector = [0] * end_output.shape[1]
                        end_label_vector[end_token_idx] = 1
                        if discriminator_label[0] == 0:
                            start_label_vector[start_token_idx] = 0
                            end_label_vector[end_token_idx] = 0
                        start_label_vector = torch.Tensor(start_label_vector).unsqueeze(0)
                        end_label_vector = torch.Tensor(end_label_vector).unsqueeze(0)
                        loss_start = self._task_criterion(start_output, start_label_vector)
                        loss_end = self._task_criterion(end_output, end_label_vector)
                        discriminator_label_vector = torch.Tensor(discriminator_label)
                        print(discriminator_output, discriminator_label)
                        discriminator_loss = self._discriminator_criterion(discriminator_output, discriminator_label_vector)
                        loss = loss_start + loss_end + discriminator_loss
                        self.task_losses.append(loss_start.item() + loss_end.item())
                        start_idx = torch.argmax(start_output)
                        end_idx = torch.argmax(end_output)
                        if start_idx <= end_idx:
                            print(' '.join(paragraph.split()[start_idx: end_idx]), '--->', answer.text())
                            # move this left from below for with discriminator loss
                        self.discriminator_losses.append(discriminator_loss.item())
                        loss.backward()
                        self._model_optimizer.step()
                        self._model.zero_grad()
                        print(np.mean(self.task_losses), np.mean(self.discriminator_losses))

    def noisify_context(self, question, context, start_token, end_token):
        if random.random() < 0.5:
            generated_context = self._tsqa_noise_generator.generate(question, context, start_token, end_token)
            return [0, 1], generated_context
        return [1, 0], context

    def remaining(self):    
        if True:
            output = self._process_outputs(question, context, output)
            if output[0] != '[SEP]':
                outputs.append(output)

        expected_targets = tsqa_datum.targets()
        outputs = sorted(outputs, key=lambda x: x[1])[-1:]
        if len(outputs) > 0 and len(expected_targets) > 0:
            expected_targets = [''.join(i.split()) for i in expected_targets]
            if ''.join(outputs[0][0].split()) in expected_targets:
                self._correct += 1
        if len(outputs) == 0 and len(expected_targets) == 0:
            self._correct += 1
        self._count += 1
        print(outputs, expected_targets, float(self._correct) / self._count)


    def _process_outputs(self, question, context, output):
        encoding = self._tokenizer.encode_plus(text=question,text_pair=context)
        inputs = encoding['input_ids']  #Token embeddings
        tokens = self._tokenizer.convert_ids_to_tokens(inputs) #input tokens

        start_index = torch.argmax(output.start_logits)
        start_index_prob = torch.max(output.start_logits)
        end_index = torch.argmax(output.end_logits)
        end_index_prob = torch.max(output.end_logits)
        weight = start_index_prob + end_index_prob
        answer = ' '.join(tokens[start_index:end_index+1])
        corrected_answer = re.sub('▁', '', answer)
        return corrected_answer, weight.item()

if __name__ == '__main__':
    qa_train = QATrain()
    qa_train.load()
    qa_train.train()
