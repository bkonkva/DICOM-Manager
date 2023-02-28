import os
import pathlib
from typing import List

from glob import glob
from tqdm import tqdm
import pydicom as dcm


class DicomFinder:
    def attempt_dicom_read(self, possible_dicom_path: pathlib.Path):
        if (
            "DICOMDIR" in os.path.basename(possible_dicom_path)
            or os.path.splitext(possible_dicom_path)[-1] == ".txt"
        ):
            return False
        try:
            with open(possible_dicom_path, "rb") as fp:
                dcm.filereader.read_preamble(fp, False)
            return possible_dicom_path
        except dcm.errors.InvalidDicomError:
            return False

    def get_all_file_paths(self, target_directory: pathlib.Path) -> List[pathlib.Path]:
        """Return paths to all non-dir files."""
        all_paths = glob(os.path.join(target_directory, "**/*"), recursive=True)
        return [file_path for file_path in all_paths if os.path.isfile(file_path)]

    def get_test_dicom_path(self, target_directory: pathlib.Path) -> pathlib.Path:
        """Return single DICOM file path for code testing."""
        possible_dicom_paths = self.get_all_file_paths(target_directory)
        for dicom_path in possible_dicom_paths:
            if self.attempt_dicom_read(dicom_path):
                return dicom_path

    def get_test_dicom_dir(self, target_directory: pathlib.Path) -> pathlib.Path:
        """Return single DICOM file path for code testing."""
        return os.path.dirname(self.get_test_dicom_path(target_directory))

    def get_dicom_paths(self, target_directory: pathlib.Path) -> List[pathlib.Path]:
        """Return all DICOM file paths"""
        possible_dicom_paths = self.get_all_file_paths(target_directory)
        return [
            dicom_path
            for dicom_path in tqdm(
                possible_dicom_paths,
                desc="locating all DICOM containing directories...",
            )
            if self.attempt_dicom_read(dicom_path)
        ]

    def get_dicom_dirs(self, target_directory: pathlib.Path) -> List[pathlib.Path]:
        """Return all subdirectories containing DICOM files."""
        dicom_paths = self.get_dicom_paths(target_directory)
        dicom_dirs = [os.path.dirname(dicom_path) for dicom_path in dicom_paths]
        dicom_dirs = list(set(dicom_dirs))
        print(
            f"found {len(dicom_dirs)} DICOM containing directories in {target_directory}, example: {dicom_dirs[0]}"
        )
        return dicom_dirs
