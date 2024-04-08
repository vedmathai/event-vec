from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_data_reader import ChaosMNLIDatareader  # noqa


class ChaosNLIStats():
    def __init__(self):
        self._data_handler = ChaosMNLIDatareader()

    def calculate(self):
        data = self._data_handler.read_file('test').data()
        for datum in data:
            print(datum.annotations())


if __name__ == '__main__':
    stats = ChaosNLIStats()
    stats.calculate()