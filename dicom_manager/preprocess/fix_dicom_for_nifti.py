import copy
import logging
from typing import List

import cv2
import dicom2nifti
import pydicom as dcm
import numpy as np

from dicom_manager.preprocess.dicom_validator import DicomVolumeValidator

log = logging.getLogger(__name__)


class FixDicomForNifti(DicomVolumeValidator):
    def __init__(self, fill_missing_with_adjacent: bool = False, allow: list = []):
        super().__init__(allow=allow)
        self.fill_missing_with_adjacent = fill_missing_with_adjacent

    def validate_for_nifti(self, dicom_files: List[dcm.dataset.Dataset], is_label: bool):
        dicom_files = self.validate_slice_increment(dicom_files)
        dicom_files = self.validate_orthogonal(dicom_files)
        dicom_files = self.validate_required_tags(dicom_files)
        dicom_files = self.validate_label_scaling(dicom_files, is_label)

        return dicom_files

        # for file in dicom_files:
        #     if hasattr(file, "Manufacturer"):
        #         del file.Manufacturer
        # return dicom_files
        #     except ValueError as e:
        # if (
        #     "all the input array dimensions for the concatenation axis must match exactly"
        #     in e
        # ):
        #     for file in dicom_files:
        #         if hasattr(file, "Manufacturer"):
        #             del file.Manufacturer
        #     return dicom_files
        # print(f"VALUE ERROR DETECTED: {e}")
        # raise ValueError

    def validate_label_scaling(self, dicom_files: List[dcm.dataset.Dataset], is_label: bool):
        if is_label:
            for file in dicom_files:
                min_pixel_val = np.min(file.pixel_array)
                file.RescaleIntercept = min_pixel_val
                file.RescaleSlope = 1
            return dicom_files
        else:
            return dicom_files

    def validate_required_tags(self, dicom_files: List[dcm.dataset.Dataset]):
        # add Modality if missing (required for dicom2nifti)
        for dicom_file in dicom_files:
            if not "Modality" in dicom_file:
                dicom_file.add_new([0x0008, 0x0060], "CS", "")
        return dicom_files

    def validate_slice_increment(
        self, dicom_files: List[dcm.dataset.Dataset]
    ) -> List[dcm.dataset.Dataset]:
        try:
            dicom2nifti.common.validate_slice_increment(dicom_files)
            return dicom_files
        except dicom2nifti.exceptions.ConversionValidationError as e:
            log.exception(e)
            return self.correct_slice_increment(dicom_files)

    def validate_orthogonal(
        self, dicom_files: List[dcm.dataset.Dataset]
    ) -> List[dcm.dataset.Dataset]:
        print("is working here")
        try:
            dicom2nifti.common.validate_orthogonal(dicom_files)
            print("valid orthogonality")
            return dicom_files
        except dicom2nifti.exceptions.ConversionValidationError as e:
            print("invalid orthogonality")
            log.exception(e)
            return self.correct_orthogonality(dicom_files)

    def correct_orthogonality(self, dicom_files: List[dcm.dataset.Dataset]):
        print("Rounding ImagePositionPatient tag values.")
        for file in dicom_files:
            print(f"before:{file.ImageOrientationPatient}")
            file.ImageOrientationPatient = [
                round(pos, 0) for pos in file.ImageOrientationPatient
            ]
            print(f"after: {file.ImageOrientationPatient}")
        log.warning("Rounding ImagePositionPatient tag values.")
        return dicom_files

    def correct_slice_increment(self, dicom_files: List[dcm.dataset.Dataset]):

        # return next_best dicom slices (with key thing...)
        # rewrite slice locations as best
        dicom_slice_positions = self.get_all_tag_idx(
            dicom_files, tag="ImagePositionPatient", idx=2
        )
        step_size = self.get_step_size(dicom_files)
        assert step_size != 0, "step size cannot be equal to zero"
        best_slice_positions = self.slice_manager.get_best_positions(
            dicom_slice_positions, step_size
        )
        next_best_slice_positions = self.slice_manager.get_next_best_positions(
            dicom_slice_positions, best_slice_positions
        )
        if not self.fill_missing_with_adjacent:
            next_best_slice_positions = sorted(list(set(next_best_slice_positions)))
        dicom_files = self.get_next_best_slices(
            dicom_files, dicom_slice_positions, next_best_slice_positions
        )
        log.warning(
            f"reconfiguring slices positions for conversion to NIFTI with step size {step_size}"
        )
        dicom_files = self.reset_slice_positions(dicom_files, best_slice_positions)
        assert (
            dicom2nifti.common.validate_slice_increment(dicom_files) == None
        ), f'increment still broken: {self.get_all_tag(dicom_files, tag = "ImagePositionPatient")}'
        return dicom_files

    def get_next_best_slices(
        self,
        dicom_files: List[dcm.dataset.Dataset],
        dicom_slice_positions: list,
        next_best_slice_positions: list,
    ):
        return [
            copy.deepcopy(dicom_files[dicom_slice_positions.index(pos)])
            for pos in next_best_slice_positions
        ]

    def reset_slice_positions(
        self, dicom_files: List[dcm.dataset.Dataset], best_slice_positions: list
    ):
        for num, dicom_file in enumerate(dicom_files):
            dicom_file.ImagePositionPatient = [
                0,
                0,
                round(best_slice_positions[num], 5),
            ]
            dicom_file.InstanceNumber = num + 1
        return dicom_files
