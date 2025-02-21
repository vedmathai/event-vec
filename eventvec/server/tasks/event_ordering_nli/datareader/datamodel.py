class TemporalNLIRow:
    def __init__(self):
        self._uid = None
        self._possible = None
        self._relationship = None
        self._event_number = 0
        self._relationship_number = 0
        self._hops = 0
        self._premise = ''
        self._hypothesis = ''
        self._label = ''
        self._type = ''

    def uid(self):
        return self._uid
    
    def possible(self):
        return self._possible
    
    def relationship(self):
        return self._relationship
    
    def event_number(self):
        return self._event_number
    
    def relationship_number(self):
        return self._relationship_number
    
    def hops(self):
        return self._hops
    
    def premise(self):
        return self._premise
    
    def hypothesis(self):
        return self._hypothesis
    
    def label(self):
        return self._label
    
    def type(self):
        return '{}_{}_{}_{}'.format(self.event_number(), self.relationship_number(), self.hops(), self.possible())
    
    def set_uid(self, uid):
        self._uid = uid


    @classmethod
    def from_array(self, datum):
        row = TemporalNLIRow()
        row._uid = datum[0]
        row._possible = datum[1]
        row._relationship = datum[2]
        row._event_number = int(datum[3])
        row._relationship_number = int(datum[4])
        row._hops = int(datum[5])
        row._premise = datum[6]
        row._hypothesis = datum[7]
        row._label = datum[8]
        return row
    
    def to_array(self):
        return [self._count, self._possible, self._relationship, self._event_number, self._relationship_number, self._hops, self._premise, self._hypothesis, self._label]