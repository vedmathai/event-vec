import os
import torch
import torch.nn as nn

from eventvec.server.data_handlers.data_handler import DataHandler
from eventvec.server.model.torch_models.eventvec.event_parts_torch_model import EventPartsRNN
from eventvec.server.model.torch_models.eventvec.event_relationship_torch_model import EventRelationshipModel
from eventvec.server.model.torch_models.eventvec.event_torch_model import EventModel
from eventvec.server.train.vectorizer.document_parser import DocumentParser
from eventvec.server.model.report_models.report import Report
from eventvec.server.model.report_models.events_report import EventsReport


CHECKPOINT_PATH = 'local/checkpoints/checkpoint.tar'
REPORT_PATH = 'local/reports/report.json'
LEARNING_RATE = 1e-4
HIDDEN_LAYER_SIZE = 50
OUTPUT_LAYER_SIZE = 50
CHECKPOINT_PATH = 'local/checkpoints/checkpoint_infer.tar'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
SAVE_EVERY = 10000

class Infer:
    def __init__(self):
        self._document_parser = DocumentParser()
        self._data_handler = DataHandler(device)

    def infer(self, text):
        document = self._document_parser.parse(text)
        report = self.create_report(document)
        report.to_file(REPORT_PATH)

            
    def load(self):
        self._data_handler.load()
        self._event_parts_model = EventPartsRNN(self._data_handler.n_words(), HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, device=device)
        self._event_model = EventModel(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, device=device)
        self._event_relationship_model = EventRelationshipModel(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, self._data_handler.n_categories(), device=device)
        self.load_checkpoint()

    def event_phrase_vectorizer(self, phrase_tensor):
        encoder_output= self._event_parts_model.initOutput()
        encoder_hidden = self._event_parts_model.initHidden()
        for ei in range(len(phrase_tensor)):
            encoder_output, encoder_hidden = self._event_parts_model(
                phrase_tensor[ei], encoder_hidden)
        return encoder_output

    def event_vectorizer(self, event):
        self._data_handler.set_event_input_tensors(event)
        event_verb_vector = self.event_phrase_vectorizer(event.verb_tensor())
        event_subject_vector = self.event_phrase_vectorizer(event.subject_tensor())
        event_object_vector = self.event_phrase_vectorizer(event.object_tensor())
        event_date_vector = self.event_phrase_vectorizer(event.date_tensor())
        event_vector = self._event_model(
            event_verb_vector, event_subject_vector, event_object_vector, event_date_vector)
        return event_vector

    def event_relationship_vectorizer(self, event_1, event_2):
        event_1_vector = self.event_vectorizer(event_1)
        event_2_vector = self.event_vectorizer(event_2)
        event_relationship_vector = self._event_relationship_model(
            event_1_vector, event_2_vector)
        return event_relationship_vector

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            self._event_parts_model.load_state_dict(checkpoint['event_parts_model_state_dict'])
            self._event_model.load_state_dict(checkpoint['event_model_state_dict'])
            self._event_relationship_model.load_state_dict(checkpoint['event_relationship_model_state_dict'])

            self._event_parts_model.eval()
            self._event_model.eval()
            self._event_relationship_model.eval()

    def create_report(self, document):
        event2vector = {}
        report = Report()
        for event in document.events():
            event_vector = self.event_vectorizer(event)
            event2vector[event] = event_vector
        for event_1, event_vector1 in event2vector.items():
            for event_2, event_vector2 in event2vector.items():
                cos = nn.CosineSimilarity(dim=1)
                sim = cos(event_vector1, event_vector2).item()
                relationships = self.event_relationship_vectorizer(event_1, event_2)
                relationship_dict = {}
                categories = self._data_handler.categories()
                for ii, i in enumerate(relationships[0]):
                    relationship_dict[categories[ii]] = i.item()

                events_report = EventsReport.create(
                    event_1, event_2, relationship_dict, sim
                )
                report.add_events_report(events_report)
        return report



"""
def sample(category, start_letter='A'):
    with torch.no_grad():
        category_tensor = categoryTensor(category).cuda()
        input = inputTensor(start_letter).cuda()
        hidden = rnn.initHidden().cuda()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters -1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter).cuda()
    return output_name
"""


if __name__ == '__main__':
    infer = Infer()
    infer.load()
    #infer.infer('Rosberg moved to williams. Rosberg retires.')

    infer.infer('Corporation televises formula one. Thompson succeeded Michael Jackson. BBC appoints Thompson as its Director-General in 2010. ')
    #infer.infer("Rosberg first drove in F1 with Williams from 2006 to 2009 and achieved two podium finishes for the team in 2008. For 2010, he moved to Mercedes, partnering fellow German and seven-time world champion Michael Schumacher. Rosberg took his first career win at the 2012 Chinese Grand Prix. He was the teammate of former karting friend and eventual seven-time World Drivers' Champion, Lewis Hamilton, from 2013 to 2016, twice finishing runner-up to his teammate before a title win in 2016.")
