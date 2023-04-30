from eventvec.server.datamodels.qa_datamodels.qa_dataset import  QADataset
from eventvec.server.datamodels.qa_datamodels.qa_datum import  QADatum
from eventvec.server.datamodels.qa_datamodels.qa_answer import  QAAnswer

class TorqueConverter:

    def convert(self, torque_documents) -> QADataset:
        qa_dataset = QADataset()
        for torque_document in torque_documents:
            data_length = len(torque_document.data())
            for datum in torque_document.data():
                self._torque_datum2qa_data(datum, qa_dataset)
        qa_dataset.set_name("Torque_dataset")
        return qa_dataset

    def _torque_datum2qa_data(self, torque_datum, qa_dataset):
        answer_spans = set([i.lower() for i in torque_datum.events().answer().spans()])
        for question in torque_datum.question_answer_pairs().questions():
            qa_datum = QADatum()
            question_text = question.question()
            question_tokens = set(question_text.lower().split())
            common = []
            for answer_token in answer_spans:
                for question_token in question_tokens:
                    if answer_token in question_token:
                        common.append(answer_token)
            qa_datum.set_question_events(common)
            qa_datum.set_question(question_text)
            qa_datum.set_context([torque_datum.passage()])
            for answeri, answer in enumerate(question.answer().indices()):
                qa_answer = QAAnswer()
                qa_answer.set_paragraph_idx(0)
                qa_answer.set_start_location(answer[0])
                qa_answer.set_end_location(answer[1])
                answer_text = question.answer().spans()[answeri]
                qa_answer.set_text(answer_text)
                qa_datum.add_answer(qa_answer)

            for alternate_answer in question.alternate_answers():
                qa_alternate_answers = []
                for answeri, answer in enumerate(alternate_answer.indices()):
                    qa_answer = QAAnswer()
                    qa_answer.set_paragraph_idx(0)
                    qa_answer.set_start_location(answer[0])
                    qa_answer.set_end_location(answer[1])
                    answer_text = alternate_answer.spans()[answeri]
                    qa_answer.set_text(answer_text)
                    qa_alternate_answers.append(qa_answer)
                qa_datum.add_alternate_answer_set(qa_alternate_answers)
            qa_dataset.add_datum(qa_datum)
