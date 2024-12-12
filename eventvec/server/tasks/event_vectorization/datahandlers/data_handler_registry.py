from eventvec.server.tasks.relationship_classification.datahandlers.model_datahandler import BertDataHandler
from eventvec.server.data.book_corpus.book_corpus_datahandlers.book_corpus_llm_datahandler import BookCorpusLLMDatahandler  # noqa
from eventvec.server.tasks.entailment_classification.datahandlers.nli_datahandler import NLIDataHandler  # noqa
from eventvec.server.tasks.connectors_mlm.roberta.datahandlers.connector_datahandler import ConnectorsDatahandler  # noqa
from eventvec.server.tasks.event_ordering_nli.roberta.datahandlers.temporal_datahandler import TemporalDatahandler  # noqa


class DataHandlerRegistry:
    registry = {
        'bert_data_handler': BertDataHandler,
        'book_corpus_llm_datahandler': BookCorpusLLMDatahandler,
        'nli_datahandler': NLIDataHandler,
        'connectors_datahandler': ConnectorsDatahandler,
        'temporal_datahandler': TemporalDatahandler,
    }

    def get_data_handler(self, data_handler_name):
        data_handler = DataHandlerRegistry.registry.get(data_handler_name)
        data_handler = data_handler()
        return data_handler
