from collections import defaultdict
import numpy as np
import csv

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.data.factuality.factuality_readers.factuality_reader import FactualityReader  # noqa
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer
from eventvec.experiments.climate.climate_data import ClimateData

quantifiers = ['all', 'every', 'each', 'any', 'no', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'few', 'a few', 'little', 'a little', 'many', 'much', 'more', 'most', 'some', 'several', 'both', 'either', 'neither', 'all', 'most', 'some', 'several', 'few', 'many', 'much', 'any', 'more', 'less', 'both', 'either', 'neither', 'a lot of', 'little', 'plenty', 'more', 'less']
temporal = ['after', 'before', 'during', 'while', 'tomorrow', 'days', 'yesterday', 'weeks', 'months', 'years', 'today', 'now', 'soon', 'later', 'then', 'next', 'last',]


class ClimateTag():
    def __init__(self):
        self._factuality_reader = FactualityReader()
        self._factuality_categorizer = FactualityCategorizer()
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._data_location = '/home/lalady6977/oerc/projects/data/sampled_climate_data.csv'
        self._write_location = '/home/lalady6977/oerc/projects/data/sampled_climate_data_tagged.csv'
        self._exaggeration_location = '/home/lalady6977/oerc/projects/local_jade/jade_front/event-vec/data/exaggeration.csv'
        self._climate_data = ClimateData()


    def tag(self):
        data = self._climate_data.read_data(self._data_location)
        seen = set()
        with open(self._exaggeration_location) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                seen.add(row[4].strip())
        for datumi, datum in enumerate(data[1:]):
            if datum.msg_id_parent().strip() in seen:
                continue
            seen.add(datum.msg_id_parent().strip())
            self._climate_data.append_tagged(datum)
            tagged = False
            featurized_doc = self._linguistic_featurizer.featurize_document(datum.body_parent())
            for sentence in featurized_doc.sentences():
                for token in sentence.tokens():
                    if token.text() in ["n't"]:
                        #features_array = self._factuality_categorizer.categorize(datum.body_parent(), token.text())
                        #if any(i is True for i in features_array.to_dict().values()):
                        tagged = True
                        tagged_datum = datum.copy()
                        #tagged_datum.set_body_parent_credence_roots(token.text())
                        #reasons = [k for k in features_array.to_dict().keys() if features_array.to_dict()[k] is True]
                        #reasons = ', '.join(reasons)
                        tagged_datum.set_body_parent_credence_roots_reason(token.text())
                        #self._climate_data.append_tagged(tagged_datum)
                #if tagged is True:
                #    self._climate_data.append_tagged(tagged_datum)

                
                #    self._climate_data.append_tagged(tagged_datum)
            featurized_doc = self._linguistic_featurizer.featurize_document(datum.body_child())
            for sentence in featurized_doc.sentences():
                for token in sentence.tokens():
                    if token.text() in ["n't", 'not', 'never']:
                        #features_array = self._factuality_categorizer.categorize(datum.body_child(), token.text())
                        #if any(i is True for i in features_array.to_dict().values()):
                        tagged = True
                        tagged_datum = datum.copy()
                        #reasons = [k for k in features_array.to_dict().keys() if features_array.to_dict()[k] is True]
                        #reasons = ', '.join(reasons)
                        #tagged_datum.set_body_child_credence_roots_reason(reasons)
                        tagged_datum.set_body_child_credence_roots(token.text())
                #if tagged is True:
                #    self._climate_data.append_tagged(tagged_datum)
                #if any(i in sentence.text().split() for i in quantifiers): #['and', 'but', 'because', 'so', 'therefore', 'however']):
                #    self._climate_data.append_tagged(datum)
                            #if features_array.is_subordinate_of_said() or features_array._is_subordinate_of_believe:
                            #    self._climate_data.append_tagged(tagged_datum)
            #if tagged is False:
            #    self._climate_data.append_tagged(datum)
            self._climate_data.write_data(self._write_location)


if __name__ == '__main__':
    tagger = ClimateTag()
    tagger.tag()