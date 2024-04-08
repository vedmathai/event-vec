import pickle
from jadelogs import JadeLogger


from eventvec.server.config import Config
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer


class FactualityRegressionInferML:
    def __init__(self):
        self._config = Config.instance()
        self._jade_logger = JadeLogger()
        self._factuality_categorizer = FactualityCategorizer()

        self._model_location = self._jade_logger.file_manager.data_filepath('ml_factuality_model.pkl')
        self._regr = pickle.load(open(self._model_location, 'rb'))

    def load(self, run_config={}):
        pass

    def _fix_datum(self, sentence, event_string):
        category = self._factuality_categorizer.categorize(sentence, event_string)
        x = []
        for key, value in sorted(category.to_dict().items(), key=lambda x: x[0]):
            if value is True:
                x.append(1)
            else:
                x.append(0)
        return x, category
    
    def infer(self, sentence, event_string):
        x, category = self._fix_datum(sentence, event_string)
        prediction = self._regr.predict([x])
        return float(prediction[0])


if __name__ == '__main__':
    fci = FactualityRegressionInferML()
    fci.load()
    label = fci.infer('I am not sure if George have gone back to school', 'gone')
    print(label)
