from calendar import EPOCH
import numpy as np
import os
from torch import optim, nn
import torch
from tqdm import tqdm
import time
from torch.optim import Adam


from eventvec.server.model.bert_models.bert_relationship_model import BertRelationshipClassifier  # noqa
from eventvec.server.data_handlers.bert_datahandler import BertDataHandler


EPOCHS = 15
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
HIDDEN_LAYER_SIZE = 50
OUTPUT_LAYER_SIZE = 50
CHECKPOINT_PATH = 'local/checkpoints/checkpoint.tar'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10000000


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self._data[idx][1])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputslabels
        return self._data[idx][0]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class Trainer:
    def __init__(self):
        self._relationship_counter = 0
        self._total_loss = 0
        self._all_losses = []
        self._data_handler = BertDataHandler()
        self._iteration = 0
        self._last_iteration = 0
        self._loss = None

    def load(self):
        self._data_handler.load()
        train_data, val_data, test_data = self._data_handler.split_data()
        self._train_dataset = Dataset(train_data)
        self._val_dataset = Dataset(val_data)
        self._test_dataset = Dataset(test_data)
        self._model = BertRelationshipClassifier()
        self._model_optimizer = Adam(
            self._model.parameters(), lr=LEARNING_RATE
        )
        weights = self._data_handler.label_weights()
        weights = torch.from_numpy(np.array(weights)).to(device)
        self._criterion = nn.CrossEntropyLoss(weight=weights)

    def zero_grad(self):
        self._model.zero_grad()

    def optimizer_step(self):
        self._model_optimizer.step()

    def train_step(self, datum):
        train_input, target = datum
        event_predicted_vector = self.classify(
            train_input['input_ids'],
            train_input['attention_mask'],
            train_input['token_type_ids'],
        )
        relationship_target = np.array([0 for i in range(len(self._data_handler.classes()))]).astype(float)
        relationship_target[target] = 1
        relationship_target = torch.from_numpy(relationship_target).to(device)
        relationship_target = relationship_target.unsqueeze(0)
        event_prediction_loss = self._criterion(
            event_predicted_vector, relationship_target
        )
        if self._loss is None:
            self._loss = event_prediction_loss
        else:
            self._loss += event_prediction_loss
        return event_prediction_loss

    def classify(self, input_ids, attention_mask, token_type_ids):
        mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        input_id = input_ids.squeeze(1).to(device)
        output = self._model(input_id, mask, token_type_ids)
        return output

    def train_epoch(self):
        if self._iteration == 0:
            self.load_checkpoint()
        self.zero_grad()
        for datum in tqdm(self._train_dataset):
            loss = self.train_step(datum)
            self._all_losses += [loss.item()]
            self._iteration += 1
            if (self._iteration - self._last_iteration) % SAVE_EVERY == 0:
                self.create_checkpoint()
                self._last_iteration = self._iteration
            if self._loss is not None and self._iteration % 10 == 0:
                self._loss.backward()
                self.optimizer_step()
                self.zero_grad()
                self._loss = None
        print('train_loss: ', sum(self._all_losses) / self._iteration)
        self.train_evaluate()

    def train(self):
        for epoch in range(EPOCHS):
            self._epoch = epoch
            self.train_epoch()

    def train_evaluate(self):
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for datumi, datum in enumerate(self._val_dataset):

                train_input, target = datum
                event_predicted_vector = self.classify(
                    train_input['input_ids'],
                    train_input['attention_mask'],
                    train_input['token_type_ids'],
                )
                relationship_target = np.array([0 for i in range(len(self._data_handler.classes()))]).astype(float)
                relationship_target[target] = 1
                relationship_target = torch.from_numpy(relationship_target).to(device)
                relationship_target = relationship_target.unsqueeze(0)
                batch_loss = self._criterion(
                    event_predicted_vector,
                    relationship_target
                )
                total_loss_val += batch_loss.item()
                if datumi < 5:
                    print(event_predicted_vector)
                if event_predicted_vector.argmax(dim=1).item() == target:
                    total_acc_val += 1
        val_data_size = len(self._val_dataset)
        print(
            'accuracy:', f'{total_acc_val / val_data_size: .3f}',
            'total_loss', f'{total_loss_val /  val_data_size: .3f}'
        )
        return {
            'total_loss': total_loss_val,
            'total_accuracy': total_acc_val
        }

    def print_epoch_stats(epoch_num, train_loss, train_data_size,
                          train_accuracy, val_loss, val_data_size,
                          val_accuracy):
        print({
            'Epochs': epoch_num + 1,
            'Train Loss': f'{train_loss / len(train_data_size): .3f}',
            'Train Accuracy': f'{train_accuracy / train_data_size: .3f}',
            'Evaluation Loss': f'{val_loss /  val_data_size: .3f}',
            'Evaluation Accuracy': f'{val_accuracy / val_data_size: .3f}'
        })

    def create_checkpoint(self):
        torch.save({
            'iteration': self._iteration,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._model_optimizer.state_dict(),
            'all_losses': self._all_losses,
            }, CHECKPOINT_PATH)
        print('Checkpoint created.')

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            self._model.load_state_dict(checkpoint['model_state_dict'])

            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self._iteration = checkpoint['iteration']
            self._all_losses = checkpoint['all_losses']
            self._model.train()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load()
    trainer.train()
