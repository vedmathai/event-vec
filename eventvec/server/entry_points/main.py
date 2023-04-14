from eventvec.server.config import Config
from eventvec.server.common.run_configs.run_config_loader.run_config_loader import RunConfigsLoader
from eventvec.server.entry_points.registries.trainer_registry import TrainerRegistry


class Main:
    def __init__(self):
        self._config = Config.instance()
        self._trainer_registry = TrainerRegistry()
        self._run_config_loader = RunConfigsLoader()

    def load(self):
        self._run_config_loader.load()

    def run(self):
        run_configs = self._run_config_loader.run_configs()
        for run_config in run_configs.run_configs():
            if run_config.is_train():
                trainer = self._trainer_registry.get_trainer(run_config.trainer())
                trainer = trainer()
                trainer.load(run_config)
                trainer.train(run_config)


if __name__ == '__main__':
    main = Main()
    main.load()
    main.run()