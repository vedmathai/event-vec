import random

from eventvec.server.tasks.event_ordering_nli.data_creator.create_cells import CreateCells
from eventvec.server.tasks.event_ordering_nli.data_creator.populate_cells import PopulateCells
from eventvec.server.tasks.event_ordering_nli.data_creator.parameters import parameters


parameter = 'temporal_nli_test'        

if parameters[parameter]['random_seed']:
    random.seed(0)

if __name__ == '__main__':
    for parameter in parameters.keys():
        creator = CreateCells(parameter)
        creator.create_data()
        creator.write_data()

        populator = PopulateCells(parameter)
        populator.read_data()
        populator.populate_data()
        populator.write_data()