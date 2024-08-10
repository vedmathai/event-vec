class MNLIDatum():
    def __init__(self):
        self._uid = None
        self._sentence_1 = None
        self._sentence_2 = None
        self._label = None
        self._entropy = None
        self._label_dist = None
        self._type = None
        self._annotator_labels = []

    def set_uid(self, uid):
        self._uid = uid

    def set_sentence_1(self, sentence_1):
        self._sentence_1 = sentence_1

    def set_sentence_2(self, sentence_2):
        self._sentence_2 = sentence_2

    def set_label(self, label):
        self._label = label

    def set_entropy(self, entropy):
        self._entropy = entropy

    def set_label_dist(self, label_dist):
        self._label_dist = label_dist

    def set_annotator_labels(self, annotator_labels):
        self._annotator_labels = annotator_labels

    def set_type(self, type):
        self._type = type

    def uid(self):
        return self._uid

    def sentence_1(self):
        return self._sentence_1
    
    def sentence_2(self):
        return self._sentence_2
    
    def label(self):
        return self._label
    
    def entropy(self):
        return self._entropy
    
    def label_dist(self):
        return self._label_dist
    
    def type(self):
        return self._type
    
    def annotator_labels(self):
        return self._annotator_labels
    
    def to_dict(self):
        return {
            'uid': self.uid(),
            'sentence_1': self._sentence_1,
            'sentence_2': self._sentence_2,
            'label': self._label,
            'entropy': self._entropy,
            'label_dist': self._label_dist,
            'annotator_labels': self._annotator_labels,
            'type': self._type,
        }
    
    @staticmethod
    def from_dict(datum_dict):
        datum = MNLIDatum()
        datum.set_uid(datum_dict['uid'])
        datum.set_sentence_1(datum_dict['sentence_1'])
        datum.set_sentence_2(datum_dict['sentence_2'])
        datum.set_label(datum_dict['label'])
        datum.set_entropy(datum_dict['entropy'])
        datum.set_label_dist(datum_dict['label_dist'])
        datum.set_type(datum_dict['type'])
        datum.set_annotator_labels(datum_dict['annotator_labels'])
        return datum
