import logging
from typing import List

import cv2
import numpy as np
import pydicom as dcm

from dicom_manager.preprocess.dicom_tag_parser import DicomTagParser
from dicom_manager.preprocess.slice_manager import SliceManager

log = logging.getLogger(__name__)


class DicomVolumeValidator(DicomTagParser):

    slice_manager = SliceManager()

    def __init__(self, allow: list):
        super().__init__(allow=allow)

    def get_best_shape(self, all_shapes: List[tuple]) -> tuple:
        """Returns largest shape in shape list...maybe better than most common?"""
        # return self.slice_manager.most_common(all_shapes)
        shape_sizes = [np.product(shape) for shape in all_shapes]
        return all_shapes[shape_sizes.index(np.amax(shape_sizes))]

    def conform_array_shape(
        self, dicom_files: List[dcm.dataset.Dataset]
    ) -> List[dcm.dataset.Dataset]:
        """Return pixel array resized to conform with dims of largest slice."""
        all_shapes = self.get_all_subtag(dicom_files, "pixel_array", "shape")
        best_shape = self.get_best_shape(all_shapes)
        log.warning(f"Resizing all arrays to shape: {best_shape}")
        pixel_data = []
        for file in dicom_files:
            pixel_data.append(
                cv2.resize(
                    file.pixel_array,
                    dsize=(best_shape[1], best_shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            )
            file.Rows = best_shape[0]
            file.Columns = best_shape[1]
        return dicom_files, pixel_data

    def check_tag_consistent(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str
    ) -> bool:
        """Return True if tag value is consistent across all DICOM files in set."""
        return len(self.get_all_tag_unique(dicom_files, tag)) <= 1

    def check_tag_consistent_idx(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str, idx: int
    ) -> bool:
        """Return True if tag[idx] value is consistent across all DICOM files in set."""
        return len(self.get_all_tag_unique_idx(dicom_files, tag, idx)) <= 1

    def check_subtag_consistent(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str, subtag: str
    ) -> bool:
        """Return True if tag value is consistent across all DICOM files in set."""
        return len(self.get_all_subtag_unique(dicom_files, tag, subtag)) <= 1

    def check_tag_unique(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str
    ) -> bool:
        """Return True if tag is consistent across all DICOM files in set."""
        return len(self.get_all_tag_unique(dicom_files, tag)) == len(
            self.get_all_tag(dicom_files, tag)
        )

    def check_tag_unique_idx(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str, idx: int
    ) -> bool:
        """Return True if tag[idx] value is unique across all DICOM files in set."""
        return len(self.get_all_tag_unique_idx(dicom_files, tag, idx)) == len(
            self.get_all_tag(dicom_files, tag)
        )

    def get_instance_count(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str
    ) -> dict:
        """Return dict of how many times each tag value appears across all DICOM files in set."""
        all_tags = self.get_all_tag(dicom_files, tag)
        return {tag: all_tags.count(tag) for tag in all_tags}

    def get_instance_count_idx(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str, idx: int
    ) -> dict:
        """Return dict of how many times each tag[idx] value appears across all DICOM files in set."""
        all_tags = self.get_all_tag_idx(dicom_files, tag, idx)
        return {tag: all_tags.count(tag) for tag in all_tags}

    def get_instance_count_sub(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str, subtag: str
    ) -> dict:
        """Return dict of how many times each tag value appears across all DICOM files in set."""
        all_tags = self.get_all_subtag(dicom_files, tag, subtag)
        return {tag: all_tags.count(tag) for tag in all_tags}

    def handle_failure(self, tag: str, warning_message: str) -> None:
        """Log unexpected tag error."""
        log.warning(warning_message)
        if not tag in self.allow:
            assert False, warning_message

    def validate_arr(self, dicom_files: List[dcm.dataset.Dataset]):
        if self.check_subtag_consistent(dicom_files, "pixel_array", "shape"):
            return dicom_files, np.dstack([file.pixel_array for file in dicom_files])
        else:
            warning_message = f'pixel_array.shape is non-unique: {self.get_instance_count_sub(dicom_files, "pixel_array", "shape")}'
            self.handle_failure("pixel_array.shape", warning_message)
            dicom_files, pixel_array = self.conform_array_shape(dicom_files)
            return dicom_files, np.dstack(pixel_array)

    def validate(self, dicom_files: List[dcm.dataset.Dataset]) -> None:
        """
        Validate tags expected to have consistent values (e.g., SeriesNumber) across set are consistent.
        Validate tags expected to have unique values (e.g., InstanceNumber) across set are consistent.
        """

        if not self.check_tag_consistent(dicom_files, "SeriesNumber"):
            warning_message = f'mulitple SeriesNumber values found: {self.get_all_tag_unique(dicom_files, "SeriesNumber")}'
            self.handle_failure("SeriesNumber", warning_message)

        if not self.check_tag_consistent_idx(dicom_files, "PixelSpacing", 0):
            warning_message = f'PixelSpacing in Y-dim inconsistent: {self.get_instance_count_idx(dicom_files, "PixelSpacing", 0)}'
            self.handle_failure("PixelSpacing", warning_message)

        if not self.check_tag_consistent_idx(dicom_files, "PixelSpacing", 1):
            warning_message = f'PixelSpacing in X-dim inconsistent: {self.get_instance_count_idx(dicom_files, "PixelSpacing", 1)}'
            self.handle_failure("PixelSpacing", warning_message)

        if not self.check_tag_consistent_idx(dicom_files, "ImagePositionPatient", 0):
            warning_message = f'ImagePositionPatient in Y-dim inconsistent: {self.get_instance_count_idx(dicom_files, "ImagePositionPatient", 0)}'
            self.handle_failure("ImagePositionPatient", warning_message)

        if not self.check_tag_consistent_idx(dicom_files, "ImagePositionPatient", 1):
            warning_message = f'ImagePositionPatient in X-dim inconsistent: {self.get_instance_count_idx(dicom_files, "ImagePositionPatient", 1)}'
            self.handle_failure("ImagePositionPatient", warning_message)

        if not self.check_tag_unique_idx(dicom_files, "ImagePositionPatient", 2):
            warning_message = f'ImagePositionPatient Z position is non-unique: {self.get_instance_count_idx(dicom_files, "ImagePositionPatient", 2)}'
            self.handle_failure("ImagePositionPatient", warning_message)

        if not self.check_tag_consistent(dicom_files, "SpacingBetweenSlices"):
            warning_message = f'SpacingBetweenSlices is non-unique: {self.get_instance_count(dicom_files, "SpacingBetweenSlices")}'
            self.handle_failure("SpacingBetweenSlices", warning_message)

        if not self.check_tag_consistent(dicom_files, "Modality"):
            warning_message = f'Modality is non-unique: {self.get_instance_count(dicom_files, "SpacingBetweenSlices")}'
            self.handle_failure("Modality", warning_message)

        if not self.check_tag_consistent(dicom_files, "RescaleIntercept"):
            warning_message = f'RescaleIntercept is non-unique: {self.get_instance_count(dicom_files, "RescaleIntercept")}'
            self.handle_failure("RescaleIntercept", warning_message)
