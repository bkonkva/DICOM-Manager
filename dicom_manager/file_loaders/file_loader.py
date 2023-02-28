import os
from abc import ABC, abstractmethod

from glob import glob
from natsort import natsorted

from ballir_dicom_manager.exception_handling import ArgErrorType


class FileLoader(ABC):
    """Load medical imaging data file."""

    def get_file_paths(self, target_path) -> list:
        try:
            if isinstance(target_path, str) and os.path.isdir(target_path):
                # target directory
                file_paths = glob(os.path.join(target_path, "**/*"), recursive=True)
                file_paths = [file for file in file_paths if not os.path.isdir(file)]
                return natsorted(file_paths)
            elif isinstance(target_path, str) and not os.path.isdir(target_path):
                # single target file
                return [target_path]
            elif isinstance(target_path, list):
                # list of target files
                return natsorted(target_path)
            else:
                raise ArgErrorType(
                    f"MUST ENTER target_path VARIABLE OF STRING TYPE (directory) or LIST TYPE (full paths), not {type(target_path)}"
                )
        except ArgErrorType as e:
            print(e)

    @abstractmethod
    def load_file(self, file_path):
        """try/except block to read individual files."""

    def load_all_files(self, target_path: str):
        file_paths: list = self.get_file_paths(target_path)
        return [self.load_file(file_path) for file_path in file_paths]
