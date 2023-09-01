from collections import defaultdict

from eventvec.server.config import Config
from eventvec.server.datamodels.qa_datamodels.qa_datum import QADatum
from eventvec.server.datamodels.qa_datamodels.qa_dataset import  QADataset
from eventvec.server.datamodels.qa_datamodels.qa_answer import  QAAnswer
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.tasks.question_answering.datahandlers.datahanders_registry import DatahandlersRegistry
from eventvec.server.datamodels.featurized_document_datamodel.featurized_sentence import FeaturizedSentence


class DateQuestionCreator:
    def __init__(self):
        self._datahandlers_registry = DatahandlersRegistry()

    def load(self):
        datahandler_class = self._datahandlers_registry.get_datahandler('torque')
        self._datahandler = datahandler_class()
        self._config = Config.instance()
        self._linguistic_featurizer = LinguisticFeaturizer()


        self._eval_data = self._datahandler.qa_eval_data().data()
        self._train_data = self._datahandler.qa_train_data().data()
        self._eval_data = self._train_data[int(len(self._train_data) * 0.8):] + self._eval_data
        self._train_data = self._train_data[:int(0.8 * len(self._train_data))]
        self._featurized_context_cache = {}
        self._poss = defaultdict(int)

    def create(self):
        self._new_data = []
        self._create_train()
        print(self._poss)
        return self._new_data

    def _create_train(self):
        train_data = self._train_data
        data_size = len(train_data)
        
        for datum_i, datum in enumerate(train_data):
            self._datum2questions(datum_i, datum)

    def _create_eval(self):
        eval_data = self._eval_data
        for datum_i, datum in enumerate(eval_data):
            self._datum2questions(datum_i, datum)

    def _datum2questions(self, qa_datum_i, qa_datum):
        token_indices = []
        token2pos = {}
        context = qa_datum.context()
        if context[0] not in self._featurized_context_cache:
            self._featurized_context_cache[context[0]] = self._linguistic_featurizer.featurize_document(context[0])
        featurized_context = self._featurized_context_cache[context[0]]
        context_i2token = {}
        answer_tokens = []
        for sentence in featurized_context.sentences():
            for token in sentence.tokens():
                context_i2token[token.idx()] = token
                if token.text() in qa_datum.question_events():
                    answer_tokens.append(token)
        for answer in qa_datum.answers():
            for paragraph_i, paragraph in enumerate(context):
                if paragraph_i == answer.paragraph_idx():
                    if answer.start_location() is not None and answer.end_location() is not None:
                        answer_token = context_i2token.get(answer.start_location())
        for sentence in featurized_context.sentences():
            for token in sentence.tokens():
                if token.entity_type() == 'DATE' and token.parent() is not None and token.parent().entity_type() != 'DATE':
                    temp = token
                    while temp.parent() is not None:
                        temp = temp.parent()
                        if temp.idx() in [i.idx() for i in answer_tokens]:
                            path = FeaturizedSentence.dependency_path_between_tokens(token, temp)
                            path = sorted(path, key=lambda x: x.idx())
                            for i in path:
                                if i.pos() == 'ADP':
                                    self._poss[i.text().lower()] += 1
                                if i.text().lower() in ['in']:#, 'on', 'at', 'for']:
                                    
                                    dates = self._child_dates(i)
                                    questions = self._create_in_questions(dates, i.text().lower(), qa_datum, temp)
                                    self._new_data.extend(questions)
                #tokens = sorted(tokens, key=lambda x: x.idx())
                #print(token.parent().text(), ' '.join([i.text() for i in tokens]))

    def _child_dates(self, token):
        dates = []
        for child_date in token.all_children():
            if child_date.entity_type() == 'DATE':
                dates += [child_date]
                for grand_child_date in child_date.all_children():
                    dates += [grand_child_date]
        return dates
    
    def _create_in_questions(self, dates, prep, qa_datum, ans_token):
        if len(dates) > 0:
            dates = sorted(dates, key = lambda x: x.idx())
            dates = [k.text() for k in dates]
            abd = None
            preps = ['after','before', 'when', 'while', 'during']
            for prep in preps:
                if prep in qa_datum.question():
                    abd = prep
            new_question_2 = ''
            if abd is not None:
                abd_point = qa_datum.question().index(abd)
                new_question_2 = qa_datum.question()[:abd_point + len(abd)] + ' {}?'
                new_question = qa_datum.question()[:abd_point] + '{} {}?'.format(prep, ' '.join(dates))
                new_question_2 = new_question_2.format(' '.join(dates))
                new_context = qa_datum.context()[0]
                #print(' '.join([i.text() for i in path]), '|', qa_datum.question())
                new_answer = QAAnswer()
                new_answer.set_text(ans_token.text())
                new_answer.set_start_location(ans_token.idx())
                new_answer.set_end_location(ans_token.idx() + len(ans_token.text()))
                new_answer.set_paragraph_idx(0)
                new_datum = QADatum()
                new_datum.set_question(new_question)
                new_datum.set_context(new_context)
                new_datum.add_answer(new_answer)
                new_datum_2 = QADatum()
                if len(new_question_2) > 0:
                    new_datum_2 = QADatum.from_dict(qa_datum.to_dict())
                    new_datum_2.set_question(new_question_2)
                    return [new_datum, new_datum_2]
                else:
                    return [new_datum]
        return []

if __name__ == '__main__':
    converter = DateQuestionCreator()
    converter.load()
    converter.create()
