from eventvec.server.tasks.question_answering.models.qa_base import QuestionAnsweringBase


class QuestionAnsweringModelsRegistry:
    _registry = {
        "qa_base": QuestionAnsweringBase,
    }

    def get_model(self, model):
        return self._registry.get(model)
