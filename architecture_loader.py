import importlib.util
from pathlib import Path


class ArchitectureLoader:
    archi_path = Path.cwd() / 'architectures'

    # ------------------- For classification only, can set number of output class in the DM from the config file -------------------------------------
    @classmethod
    def get_classification_dm(cls, architecture, dropout, n_classes):
        assert not architecture.endswith('_dm'), "Remove '_dm' from the architecture you passed"
        # Load the given decision maker
        module_path = cls.archi_path / (architecture + "_dm.py")
        spec = importlib.util.spec_from_file_location(architecture, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_classification_block(dropout, n_classes)


    # ------------------- For other tasks -------------------------------------
    @classmethod
    def get_fe(cls, architecture, dropout):
        assert not architecture.endswith('_fe'), "Remove '_fe' from the architecture you passed"
        # Load the given feature extractor
        module_path = cls.archi_path / (architecture + "_fe.py")
        spec = importlib.util.spec_from_file_location(architecture, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_block(dropout)

    @classmethod
    def get_dm(cls, architecture, dropout):
        assert not architecture.endswith('_dm'), "Remove '_dm' from the architecture you passed"
        # Load the given decision maker
        module_path = cls.archi_path / (architecture + "_dm.py")
        spec = importlib.util.spec_from_file_location(architecture, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_block(dropout)
