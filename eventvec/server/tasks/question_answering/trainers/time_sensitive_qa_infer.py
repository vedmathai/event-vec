import re
import torch
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer


from eventvec.server.config import Config
from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datareader import TSQADataReader
from eventvec.server.reporter.report_model.report_model import ReportModel


class TimeSensitiveQAInfer:
    
    def load(self):
        self._data_reader = TSQADataReader()
        self._model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc", block_size=16, num_random_blocks=2)
        self._tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
        self._config = Config.instance()

    def infer(self):
        for document in self._data_reader.tsqa_documents():
            self._infer_document(document)

    def _infer_document(self, tsqa_document):
        self._count = 0
        self._correct = 0
        for tsqa_datum in tsqa_document.tsqa_data():
            self._infer_datum(tsqa_datum)


    def _infer_datum(self, tsqa_datum):
        question = tsqa_datum.question()
        outputs = []
        for paragraph in tsqa_datum.paragraphs():
            context = paragraph.text()
            encoded_input = self._tokenizer(question, context, return_tensors='pt')
            
            if len(set(context.split())) > 800:
                return 
            output = self._model(**encoded_input)
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
    tsqa_infer = TimeSensitiveQAInfer()
    tsqa_infer.load()
    tsqa_infer.infer()
