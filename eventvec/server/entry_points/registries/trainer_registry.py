from eventvec.server.tasks.question_answering.trainers.qa_base_train import QATrainBase

class TrainerRegistry:
    _registry = {
        "qa_trainer": QATrainBase,
    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
