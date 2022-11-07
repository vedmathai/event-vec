
class DataHandlerRegistry:
    _data_handlers = {
        'timebank_data_handler': TimeBankDataHandler,
        'matres_data_handler': MatresDataHandler,
    }

    def data_handler(self, data_handler_name):
        data_handler = self._data_handlers[data_handler_name]
        data_handler()
        data_handler.load()
        return data_handler
