from eventvec.server.data.torque.datahandlers.torque_converter import TorqueConverter
from eventvec.server.data.torque.readers.torque_datareader import TorqueDataReader


class TorqueDatahandler:
    def __init__(self):
        self._torque_converter = TorqueConverter()
        self._torque_datareader = TorqueDataReader()

    def qa_train_data(self):
        torque_documents = self._torque_datareader.torque_train_dataset()
        qa_data = self._torque_converter.convert(torque_documents)
        return qa_data

    def qa_eval_data(self):
        torque_documents = self._torque_datareader.torque_eval_dataset()
        qa_data = self._torque_converter.convert(torque_documents)
        return qa_data
