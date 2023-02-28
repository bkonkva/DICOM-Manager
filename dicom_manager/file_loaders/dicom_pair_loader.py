"""Loader preprocessed DICOM files from postprocessed DICOM meta data origin path."""

import pathlib

import pydicom as dcm

from dicom_manager.file_loaders.file_loader import FileLoader


class DicomPairLoader(FileLoader):
    """Load DICOM files."""

    def get_path_from_meta(self, target_path: pathlib.Path):
        preprocessed_meta_data = dcm.dcmread(target_path, stop_before_pixels=True)
        return preprocessed_meta_data[0x000B, 0x1001].value

    def load_file(self, target_path: str):
        try:
            return dcm.dcmread(self.get_path_from_meta(target_path))
        except dcm.errors.InvalidDicomError as e:
            print(f"{target_path} is unreadable: {e}")
            pass


# Convert Nifti to DICOM... make sure we're using the correct DICOM files as templates.
# Calculate volume by... (voxels with Spacing based on z range of mask?)...
# assert first slice pos minus last slice pos/slices == SpacingBetweenSlices
