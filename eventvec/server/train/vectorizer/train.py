import numpy as np
from torch import optim, nn
import torch
import time

from eventvec.server.data_handlers.data_handler import DataHandler
from eventvec.server.model.torch_models.eventvec.event_parts_torch_model import EventPartsRNN
from eventvec.server.model.torch_models.eventvec.event_relationship_torch_model import EventRelationshipModel
from eventvec.server.model.torch_models.eventvec.event_torch_model import EventModel

LEARNING_RATE = 1e-4
HIDDEN_LAYER_SIZE = 50
OUTPUT_LAYER_SIZE = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:

    def __init__(self):
        self._relationship_counter = 0
        self._total_loss = 0
        self._all_losses = []
        self._criterion = nn.CrossEntropyLoss()
        self._data_handler = DataHandler()

    def load(self):
        self._data_handler.load()
        self._event_parts_model = EventPartsRNN(self._data_handler.n_words(), HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, device=device)
        self._event_model = EventModel(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, device=device)
        self._event_relationship_model = EventRelationshipModel(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, self._data_handler.n_categories(), device=device)
        self._event_model_optimizer = optim.SGD(self._event_model.parameters(), lr=LEARNING_RATE)
        self._event_parts_optimizer = optim.SGD(self._event_parts_model.parameters(), lr=LEARNING_RATE)
        self._event_relationship_optimizer = optim.SGD(self._event_relationship_model.parameters(), lr=LEARNING_RATE)

    def zero_grad(self):
        self._event_model.zero_grad()
        self._event_parts_model.zero_grad()
        self._event_relationship_model.zero_grad()

    def optimizer_step(self):
        self._event_model_optimizer.step()
        self._event_parts_optimizer.step()
        self._event_relationship_optimizer.step()

    def train_step(self, relationship):
        self.zero_grad()
        event_predicted_vector = self.event_relationship_vectorizer(relationship)
        target_relationship, target_score = self.get_target(relationship)
        event_prediction_loss = self._criterion(event_predicted_vector, target_relationship)
        loss = event_prediction_loss
        loss.backward()
        self.optimizer_step()
        return loss

    def event_phrase_vectorizer(self, phrase_tensor):
        encoder_output= self._event_parts_model.initOutput()
        hidden = self._event_parts_model.initHidden()
        for ei in range(len(phrase_tensor)):
            encoder_output, encoder_hidden = self._event_parts_model(
                phrase_tensor[ei], hidden)
        return encoder_output

    def event_vectorizer(self, event):
        self._data_handler.set_event_input_tensors(event)
        event_verb_vector = self.event_phrase_vectorizer(event.verb_tensor())
        event_subject_vector = self.event_phrase_vectorizer(event.subject_tensor())
        event_object_vector = self.event_phrase_vectorizer(event.object_tensor())
        event_vector = self._event_model(
            event_verb_vector, event_subject_vector, event_object_vector)
        return event_vector

    def event_relationship_vectorizer(self, relationship):
        event_1 = relationship.event_1()
        event_2 = relationship.event_2()
        event_1_vector = self.event_vectorizer(event_1)
        event_2_vector = self.event_vectorizer(event_2)
        event_relationship_vector = self._event_relationship_model(
            event_1_vector, event_2_vector)
        return event_relationship_vector

    def get_target(self, relationship):
        relationship_type = relationship.relationship()
        relationship_score = relationship.relationship_score()
        relationship_target, target_score = self._data_handler.targetTensor(relationship_type, relationship_score)
        return relationship_target, target_score

    def train_document(self, document):
        start = time.time()
        for relationship in document.relationships():
            loss = self.train_step(relationship)
            self._relationship_counter += 1
            self._all_losses += [loss.item()]
            print(np.mean(self._all_losses))

"""
            if iter % print_every == 0:
                print('%s (%d %d%%) %.4f' %(timeSince(start), iter, iter/n_iters*100, loss))

                all_losses.append(total_loss/plot_every)
                total_loss = 0
                if iter % (plot_every * print_every) == 0:
                    plt.figure()
                    plt.plot(all_losses)
                    plt.show()
"""
