from eventvec.server.tasks.question_answering.trainers.qa_base_train import QATrainBase
from eventvec.server.tasks.question_answering.trainers.qa_reinforce_train import QAReinforceTrain
from eventvec.server.tasks.relationship_classification.trainers.train import RelationshipClassificationTrain


class TrainerRegistry:
    _registry = {
        'relationship_trainer': RelationshipClassificationTrain,
        "qa_trainer": QATrainBase,
        "qa_reinforce_trainer": QAReinforceTrain,

    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
