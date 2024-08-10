class ConnectorDatum:
    def __init__(self):
        self._uid = None
        self._para = None
        self._label = None

    def set_uid(self, uid):
        self._uid = uid

    def set_para(self, para):
        self._para = para

    def set_label(self, label):
        self._label = label

    def uid(self):
        return self._uid

    def para(self):
        return self._para
    
    def label(self):
        return self._label
    
