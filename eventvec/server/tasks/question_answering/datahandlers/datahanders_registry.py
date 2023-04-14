from eventvec.server.data.torque.datahandlers.torque_datahandler import TorqueDatahandler


class DatahandlersRegistry:
    _registry = {
        "torque": TorqueDatahandler,
    }

    def get_datahandler(self, datahandler):
        return self._registry[datahandler]
