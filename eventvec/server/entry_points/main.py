import pprint

from timebank_embedding.data_handler import DataHandler
from timebank_embedding.variables import Variables



variables = Variables()
data_handler = DataHandler(variables)

data_handler.load_data()
data_handler.generate_word2index()
data_handler.generate_event_sets()



#pprint.pprint(data_handler._file_name2event_set)
print('reached')
print(data_handler.inputTensor('and'))