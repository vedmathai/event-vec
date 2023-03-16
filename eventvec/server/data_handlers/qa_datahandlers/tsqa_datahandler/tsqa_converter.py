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
            for annotation_answer in annotation_question.tsqa_annotation_answers():
                qa_answer = QAAnswer()
                qa_answer.set_paragraph(annotation_answer.para())
                qa_answer.set_start_location(annotation_answer.from_token())
                qa_answer.set_end_location(annotation_answer.end_token())
                qa_answer.set_text(annotation_answer.answer())
                qa_datum.add_answer(qa_answer)
            qa_datum.set_context(annotation.paragraphs())
            qa_dataset.add_datum(qa_datum)
