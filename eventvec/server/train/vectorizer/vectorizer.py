import gc

from eventvec.utils.data_readers.fervous_wikipedia_reader import FerverousDataset
from eventvec.server.train.vectorizer.document_parser import DocumentParser

class Vectorizer:
    def __init__(self):
        self.document_parser = DocumentParser()

    def get_dataset(self):
        dataset = FerverousDataset()
        dataset.load()
        total_events = 0
        total_relationships = 0
        for text_i, text in enumerate(dataset.contents()):
            document = self.document_parser.parse(text)
            events_len = len(document.events())
            relationships_len = len(document.relationships())
            total_events += events_len
            total_relationships += relationships_len
            sentence_length = len(text.split('.'))
            print(f'{text_i} - {events_len} - {relationships_len} - {total_events} - {total_relationships} - {sentence_length}')


if __name__ == '__main__':
    vectorizer = Vectorizer()
    vectorizer.get_dataset()
