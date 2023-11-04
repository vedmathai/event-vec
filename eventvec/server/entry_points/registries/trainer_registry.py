from eventvec.server.tasks.question_answering.trainers.qa_base_train import QATrainBase
from eventvec.server.tasks.question_answering.trainers.qa_reinforce_train import QAReinforceTrain
from eventvec.server.tasks.relationship_classification.trainers.train import RelationshipClassificationTrain
from eventvec.server.tasks.factuality_estimator.trainers.train import FactualityEstimationTrain


class TrainerRegistry:
    _registry = {
        "relationship_trainer": RelationshipClassificationTrain,
        "factuality_trainer": FactualityEstimationTrain,
        "qa_trainer": QATrainBase,
        "qa_reinforce_trainer": QAReinforceTrain,
    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
