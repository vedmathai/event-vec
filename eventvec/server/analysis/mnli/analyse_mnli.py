from collections import defaultdict
from math import log


from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader

from eventvec.server.common.lists.said_verbs import said_verbs, future_said_verbs, future_modals


class AnalyseMNLI():
    def __init__(self):
        self._datahandler = MNLIDataReader()
        
    def entropy(self, d):
        total = sum(d.values())
        for i in d:
            d[i] /= total
        entropy = 0
        for i in d:
            if d[i] != 0:
                entropy -= d[i] * log(d[i], 2)
        return entropy

    def analyse(self):
        elements = self._datahandler.read_file(None)
        """
        overall_counter = defaultdict(int)
        for element in elements:
            counter = 0
            for i in element[0]:
                if i == element[1]:
                    counter += 1
            if counter == 0:
                print(element[0], element[1])
            overall_counter[counter] += 1
        print(overall_counter)
        """

        diff = 1.584/10
        divisions = [(diff*i, diff*(i+1)) for i in range(10)]
        entropies = []
        factuality_entropies = defaultdict(int)
        non_factuality_entropies = defaultdict(int)
        check = set(future_modals) | set(future_said_verbs) | set(said_verbs)
        for element in elements:
            counter = defaultdict(int)
            for i in element[0]:
                counter[i] += 1
            entropy = self.entropy(counter)
            if (any (i in element[2] for i in set(check)) or any (i in element[3] for i in set(check))):
                for di, d in enumerate(divisions):
                    if d[0] <= entropy < d[1]:
                        factuality_entropies[entropy] += 1
                        break
            else:
                for di, d in enumerate(divisions):
                    if d[0] <= entropy < d[1]:
                        non_factuality_entropies[entropy] += 1
                        break
        total = sum(factuality_entropies.values())
        print([(k, i/total) for k, i in sorted(factuality_entropies.items(), key=lambda x: x[0])], total)
        total = sum(non_factuality_entropies.values())
        print([(k, i/total) for k, i in sorted(non_factuality_entropies.items(), key=lambda x: x[0])], total)
        print(total)


if __name__ == '__main__':
    analyse_mnli = AnalyseMNLI()
    analyse_mnli.analyse()
