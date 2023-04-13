from eventvec.server.config import Config

class AbstractDatareader:
    def __init__(self):
        self._config = Config.instance()