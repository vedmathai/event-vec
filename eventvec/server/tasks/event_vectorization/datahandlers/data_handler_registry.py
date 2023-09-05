from eventvec.server.tasks.relationship_classification.datahandlers.model_datahandler import BertDataHandler
from eventvec.server.data.book_corpus.book_corpus_datahandlers.book_corpus_llm_datahandler import BookCorpusLLMDatahandler  # noqa


class DataHandlerRegistry:
    registry = {
        'bert_data_handler': BertDataHandler,
        'book_corpus_llm_datahandler': BookCorpusLLMDatahandler
    }

    def get_data_handler(self, data_handler_name):
        data_handler = DataHandlerRegistry.registry.get(data_handler_name)
        data_handler = data_handler()
        return data_handler
