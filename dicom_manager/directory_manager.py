import os

from dataclasses import dataclass

@dataclass
class DirManager:
    
    def __init__(self, **kwargs):
        for var, path in kwargs.items():
            setattr(self, var, path)
        self.build_dirs()
    
    def build_dirs(self) -> None:
        for process_dir in vars(self).values():
            if not os.path.exists(process_dir): os.makedirs(process_dir)

    def print(self) -> None:
        for var, process_path in vars(self).items():
            print(f'{var} : {process_path}')
