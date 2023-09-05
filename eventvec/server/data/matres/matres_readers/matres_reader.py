import os
from typing import Dict

from eventvec.server.data.abstract import AbstractDatareader


class MatresDataReader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self.folder = self._config.matres_data_location()

    def timebank_file(self):
        return os.path.join(self.folder, 'timebank.txt')
    
    def aquaint_file(self):
        return os.path.join(self.folder, 'aquaint.txt')
    
    def platinum_file(self):
        return os.path.join(self.folder, 'platinum.txt')

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def matres_dict(self, dataset_name) -> Dict:
        file2rel = {}
        filename2fn = {
            'timebank': self.timebank_file,
            'aquaint': self.aquaint_file,
            'platinum': self.platinum_file,
        }
        filepathfn = filename2fn.get(dataset_name)
        filepath = filepathfn()
        with open(filepath) as f:
            for l in f:
                newsfilename, from_verb, to_verb, from_id, to_id, relationship = l.strip().split()
                file2rel[newsfilename, from_id, to_id] = (from_verb, to_verb, relationship)
        return file2rel


if __name__ == '__main__':
    mdr = MatresDataReader()
    print(mdr.matres_dict('timebank'))
