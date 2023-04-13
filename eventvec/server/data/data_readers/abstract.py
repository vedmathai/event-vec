from eventvec.server.config import Config


class AbstractDataReader:
    def __init__(self):
        self.config = Config.instance()
