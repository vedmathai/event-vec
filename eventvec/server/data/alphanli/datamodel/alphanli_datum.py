
class AlphaNLIDatum():
    def __init__(self):
        self._uid = None
        self._obs_1 = None
        self._obs_2 = None
        self._hyp_1 = None
        self._hyp_2 = None
        self._label = None

    def set_uid(self, uid):
        self._uid = uid

    def set_obs_1(self, obs_1):
        self._obs_1 = obs_1

    def set_obs_2(self, obs_2):
        self._obs_2 = obs_2

    def set_hyp_1(self, hyp_1):
        self._hyp_1 = hyp_1

    def set_hyp_2(self, hyp_2):
        self._hyp_2 = hyp_2

    def set_label(self, label):
        self._label = label

    def uid(self):
        return self._uid

    def obs_1(self):
        return self._obs_1
    
    def obs_2(self):
        return self._obs_2
    
    def hyp_1(self):
        return self._hyp_1
    
    def hyp_2(self):
        return self._hyp_2
    
    def label(self):
        return self._label
    
    def to_dict(self):
        return {
            'uid': self.uid(),
            'obs_1': self._obs_1,
            'obs_2': self._obs_2,
            'hyp_1': self._hyp_1,
            'hyp_2': self._hyp_2,
            'label': self._label
        }
    
    @staticmethod
    def from_dict(datum_dict):
        datum = AlphaNLIDatum()
        datum.set_uid(datum_dict['uid'])
        datum.set_obs_1(datum_dict['obs1'])
        datum.set_obs_2(datum_dict['obs2'])
        datum.set_hyp_1(datum_dict['hyp1'])
        datum.set_hyp_2(datum_dict['hyp2'])
        datum.set_label(datum_dict['label'])
        return datum
