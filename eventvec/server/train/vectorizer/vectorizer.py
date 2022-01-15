import gc
from eventvec.server.data_handlers.data_handler import DataHandler

from eventvec.utils.data_readers.fervous_wikipedia_reader import FerverousDataset
from eventvec.server.train.vectorizer.document_parser import DocumentParser

class Vectorizer:
    def __init__(self):
        self.document_parser = DocumentParser()
        self.data_handler = DataHandler()
        
    def load(self):
        self.data_handler.load()

    def get_dataset(self):
        dataset = FerverousDataset()
        dataset.load()
        text_i = 0
        while text_i < 2:
            text = dataset.get_next_article()
            document = self.document_parser.parse(text)
            for relationship in document.relationships():
                event_1 = relationship.event_1()
                event_2 = relationship.event_2()
                self.data_handler.set_event_input_tensors(event_1)
                self.data_handler.set_event_input_tensors(event_2)
                target = self.data_handler.targetTensor(relationship.relationship(), relationship.relationship_score())
                print(target)


if __name__ == '__main__':
    vectorizer = Vectorizer()
    vectorizer.load()
    vectorizer.get_dataset()
