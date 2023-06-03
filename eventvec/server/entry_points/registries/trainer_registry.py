from eventvec.server.tasks.question_answering.trainers.qa_base_train import QATrainBase
from eventvec.server.tasks.question_answering.trainers.qa_reinforce_train import QAReinforceTrain


class TrainerRegistry:
    _registry = {
        "qa_trainer": QATrainBase,
        "qa_reinforce_trainer": QAReinforceTrain,

    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
