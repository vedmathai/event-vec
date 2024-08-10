from eventvec.server.tasks.question_answering.trainers.qa_base_train import QATrainBase
from eventvec.server.tasks.question_answering.trainers.qa_reinforce_train import QAReinforceTrain
from eventvec.server.tasks.relationship_classification.trainers.train import RelationshipClassificationTrain
from eventvec.server.tasks.factuality_estimator.trainers.train import FactualityEstimationTrain
from eventvec.server.tasks.entailment_classification.trainers.train import NLIClassificationTrain
from eventvec.server.tasks.factuality_estimator.trainers.train_ml import FactualityEstimationTrainML
from eventvec.server.tasks.connectors_mlm.roberta.trainers.train import NLIConnectorClassificationTrain


class TrainerRegistry:
    _registry = {
        "relationship_trainer": RelationshipClassificationTrain,
        "factuality_trainer": FactualityEstimationTrain,
        "factuality_trainer_ml": FactualityEstimationTrainML,
        'nli_trainer': NLIClassificationTrain,
        "qa_trainer": QATrainBase,
        "qa_reinforce_trainer": QAReinforceTrain,
        'nli_connector_trainer': NLIConnectorClassificationTrain,
    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
