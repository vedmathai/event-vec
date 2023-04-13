from eventvec.server.datamodels.run_configs_datamodel.run_config import RunConfig


class RunConfigs:
    def __init__(self):
        self._run_configs = []

    def run_configs(self):
        return self._run_configs

    def set_run_configs(self, run_configs):
        self._run_configs = run_configs

    def add_run_config(self, run_config):
        self._run_configs.append(run_config)

    @staticmethod
    def from_dict(run_configs_dict):
        run_configs = RunConfigs()
        for run_config_dict in run_configs_dict['run_configs']:
            run_config = RunConfig.from_dict(run_config_dict)
            run_configs.add_run_config(run_config)
        return run_configs

    def to_dict(self):
        return {
            'run_configs': [i.to_dict() for i in self._run_configs()]
        }
