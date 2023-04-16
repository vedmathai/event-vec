from eventvec.server.datamodels.qa_datamodels.qa_dataset import  QADataset
from eventvec.server.datamodels.qa_datamodels.qa_datum import  QADatum
from eventvec.server.datamodels.qa_datamodels.qa_answer import  QAAnswer

question_set_1 = ['What event has already happened?',  "What is happening now?", "What will happen in the future?"]
question_set_2 = ["What event has already finished?", "What event has begun but has not finished?", "What events have already finished?", "What events have begun but has not finished?"]
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
        for question in torque_datum.question_answer_pairs().questions():
            qa_datum = QADatum()
            question_text = question.question()
            if question_text not in question_set_2:
                pass
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
            qa_dataset.add_datum(qa_datum)
