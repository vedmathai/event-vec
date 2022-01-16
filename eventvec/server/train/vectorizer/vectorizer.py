from eventvec.server.train.vectorizer.train import Trainer 
from eventvec.utils.data_readers.fervous_wikipedia_reader import FerverousDataset
from eventvec.server.train.vectorizer.document_parser import DocumentParser

class Vectorizer:
    def __init__(self):
        self.document_parser = DocumentParser()
        self.trainer = Trainer()
        
    def load(self):
        self.trainer.load()

    def get_dataset(self):
        dataset = FerverousDataset()
        dataset.load()
        text_i = 0
        while True:
            text = dataset.get_next_article()
            document = self.document_parser.parse(text)
            self.trainer.train_document(document)


if __name__ == '__main__':
    vectorizer = Vectorizer()
    vectorizer.load()
    vectorizer.get_dataset()
