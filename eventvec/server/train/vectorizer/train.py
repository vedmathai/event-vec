import numpy as np
import os
from torch import optim, nn
import torch
import time

from eventvec.server.data_handlers.data_handler import DataHandler
from eventvec.server.model.torch_models.eventvec.event_parts_torch_model import EventPartsRNN
from eventvec.server.model.torch_models.eventvec.event_relationship_torch_model import EventRelationshipModel
from eventvec.server.model.torch_models.eventvec.event_torch_model import EventModel

LEARNING_RATE = 1e-2
HIDDEN_LAYER_SIZE = 50
OUTPUT_LAYER_SIZE = 50
CHECKPOINT_PATH = 'local/checkpoints/checkpoint.tar'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 2000

class Trainer:

    def __init__(self):
        self._relationship_counter = 0
        self._total_loss = 0
        self._all_losses = []
        self._criterion = nn.MSELoss()
        self._data_handler = DataHandler(device)
        self._iteration = 0
        self._last_iteration = 0

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
        relationship_target = self.get_target(relationship)
        event_prediction_loss = self._criterion(event_predicted_vector, relationship_target)
        loss = event_prediction_loss
        loss.backward()
        self.optimizer_step()
        return loss

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

    def event_relationship_vectorizer(self, relationship):
        event_1 = relationship.event_1()
        event_2 = relationship.event_2()
        event_1_vector = self.event_vectorizer(event_1)
        event_2_vector = self.event_vectorizer(event_2)
        event_relationship_vector = self._event_relationship_model(
            event_1_vector, event_2_vector)
        return event_relationship_vector

    def get_target(self, relationship):
        relationship_distribution = relationship.relationship_distribution()
        relationship_target = self._data_handler.targetTensor(relationship_distribution)
        return relationship_target

    def train_document(self, document):
        if self._iteration == 0:
            self.load_checkpoint()
        start = time.time()
        for relationship in document.relationships():
            loss = self.train_step(relationship)
            self._relationship_counter += 1
            self._all_losses += [loss.item()]
            self._iteration += 1
            if (self._iteration - self._last_iteration) % SAVE_EVERY == 0:
                self.create_checkpoint()
                self._last_iteration = self._iteration
        print(np.mean(self._all_losses), self._iteration)

        

    def create_checkpoint(self):
        torch.save({
            'iteration': self._iteration,
            'event_parts_model_state_dict': self._event_parts_model.state_dict(),
            'event_model_state_dict': self._event_model.state_dict(),
            'event_relationship_model_state_dict': self._event_relationship_model.state_dict(),
            'event_model_optimizer_state_dict': self._event_model_optimizer.state_dict(),
            'event_parts_optimizer_dict': self._event_parts_optimizer.state_dict(),
            'event_relationship_optimizer_dict': self._event_relationship_optimizer.state_dict(),
            'all_losses': self._all_losses,
            }, CHECKPOINT_PATH)
        print('Checkpoint created.')

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            self._event_parts_model.load_state_dict(checkpoint['event_parts_model_state_dict'])
            self._event_model.load_state_dict(checkpoint['event_model_state_dict'])
            self._event_relationship_model.load_state_dict(checkpoint['event_relationship_model_state_dict'])

            self._event_model_optimizer.load_state_dict(checkpoint['event_model_optimizer_state_dict'])
            self._event_parts_optimizer.load_state_dict(checkpoint['event_parts_optimizer_dict'])
            self._event_relationship_optimizer.load_state_dict(checkpoint['event_relationship_optimizer_dict'])

            self._iteration = checkpoint['iteration']
            self._all_losses = checkpoint['all_losses']
            self._event_parts_model.train()
            self._event_model.train()
            self._event_relationship_model.train()


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
