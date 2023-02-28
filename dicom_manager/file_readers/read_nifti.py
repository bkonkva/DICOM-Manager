import pathlib

import numpy as np

from dicom_manager.file_readers.read_image_volume import ReadImageVolume
from dicom_manager.file_loaders.nifti_loader import NiftiLoader
from dicom_manager.file_viewers.array_viewer import ArrayViewer


class ReadNifti(ReadImageVolume):

    loader = NiftiLoader()

    def __init__(self, target_path: pathlib.Path, value_clip=False):
        super().__init__(target_path)
        #         self.files = self.sorter.sort_dicom_files(self.files)
        #         self.validator.validate(self.files)
        self.value_clip = value_clip
        self.spacing = self.files[0].header.get_zooms()
        self.set_arr()

    def set_arr(self):
        self.arr = np.rot90(self.files[0].get_fdata(), k=1, axes=(0, 1))  #
        if self.value_clip:
            self.arr = np.clip(self.arr, self.value_clip[0], self.value_clip[1])
        self.viewer = ArrayViewer(self.arr, self.spacing)
