from eventvec.server.train.vectorizer.train import Trainer 
from eventvec.utils.data_readers.fervous_wikipedia_reader import FerverousDataset
from eventvec.utils.dates_document import create_dates_document
from eventvec.server.train.vectorizer.document_parser import DocumentParser
from eventvec.utils.dates_document import create_dates_document


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
        dates_text = create_dates_document()
        #for text in dates_text:
        #    document = self.document_parser.parse(text)
        #    self.trainer.train_document(document)
        while True:
            text = dataset.get_next_article()
            if text is not None:
                document = self.document_parser.parse(text)
                self.trainer.train_document(document)


if __name__ == '__main__':
    vectorizer = Vectorizer()
    vectorizer.load()
    vectorizer.get_dataset()
