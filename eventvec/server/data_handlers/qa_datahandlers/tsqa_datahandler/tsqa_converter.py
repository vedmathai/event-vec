from eventvec.server.model.qa_models.datamodel.qa_dataset import  QADataset
from eventvec.server.model.qa_models.datamodel.qa_datum import  QADatum
from eventvec.server.model.qa_models.datamodel.qa_answer import  QAAnswer


class TSQAConverter:

    def convert(self, tsqa_documents) -> QADataset:
        qa_dataset = QADataset()
        for tsqa_document in tsqa_documents:
            tsqa_annotation_doc = tsqa_document.tsqa_annotation_document()
            for annotation in tsqa_annotation_doc.tsqa_annotations().values():
                self.annotation2qa_data(annotation, tsqa_document, qa_dataset)
        qa_dataset.set_name("TSQA_dataset")
        return qa_dataset

    def annotation2qa_data(self, annotation, tsqa_document, qa_dataset):
        for annotation_question in annotation.questions():
            qa_datum = QADatum()
            datum = tsqa_document.id2datum(annotation_question.idx())
            qa_datum.set_id(annotation_question.idx())
            question_text = datum.question()
            qa_datum.set_question(question_text)
            qa_datum.set_context(annotation.paragraphs())
            for annotation_answer in annotation_question.tsqa_annotation_answers():
                qa_answer = QAAnswer()
                qa_answer.set_paragraph_idx(annotation_answer.para())
                paragraph = annotation.paragraphs()[annotation_answer.para()]
                char2wordidx = self._char2wordidx(paragraph)
                from_token = self._find_nearest(char2wordidx, annotation_answer.from_char())
                end_token = self._find_nearest(char2wordidx, annotation_answer.end_char())
                qa_answer.set_start_location(from_token)
                qa_answer.set_end_location(end_token)
                qa_answer.set_text(annotation_answer.answer())
                qa_datum.add_answer(qa_answer)
            qa_dataset.add_datum(qa_datum)

    def _char2wordidx(self, sentence):
        word_i = 1
        char2wordidx = {0: 0}
        for ii, i in enumerate(sentence):
            if i == ' ':
                char2wordidx[ii] = word_i
                word_i += 1
        return char2wordidx

    def _find_nearest(self, char2wordidx, idx):
        for i in range(0, 4):
            if idx + i in char2wordidx:
                return char2wordidx[idx+i]
            if idx - i in char2wordidx:
                return char2wordidx[idx-i]
