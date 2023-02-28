from typing import List
import logging

import dicom2nifti
import pydicom as dcm

from dicom_manager.preprocess.slice_manager import SliceManager

log = logging.getLogger(__name__)


class DicomSorter:
    def sort_dicom_files(self, dicom_files):
        return dicom2nifti.common.sort_dicoms(dicom_files)


class DicomTagParser:

    slice_manager = SliceManager()

    def __init__(self, allow=[]):
        self.allow = allow

    def get_all_tag(self, dicom_files, tag: str) -> list:
        """Return list of all tag values across volume for provided tag."""
        return [getattr(file, tag) for file in dicom_files if hasattr(file, tag)]

    def get_all_subtag(
        self, dicom_files: List[dcm.dataset.Dataset], tag: str, subtag: str
    ) -> list:
        """Return list of nested tag values across volume for provided tag.subtag"""
        all_tags = [getattr(file, tag) for file in dicom_files if hasattr(file, tag)]
        return [
            getattr(file_tag, subtag)
            for file_tag in all_tags
            if hasattr(file_tag, subtag)
        ]

    def get_all_tag_idx(self, dicom_files, tag: str, idx: int) -> list:
        """Return list of all tag values at specified index across volume for provided tag."""
        return [getattr(file, tag)[idx] for file in dicom_files if hasattr(file, tag)]

    def get_all_tag_unique(self, dicom_files, tag: str) -> list:
        """Return all unique values across volume for provided tag."""
        return list(set(self.get_all_tag(dicom_files, tag)))

    def get_all_tag_unique_idx(self, dicom_files, tag: str, idx: int) -> list:
        """Return all unique values at specified index across volume for provided tag."""
        return list(set(self.get_all_tag_idx(dicom_files, tag, idx)))

    def get_all_subtag_unique(self, dicom_files, tag: str, subtag: str) -> list:
        """Return all unique values across volume for provided tag."""
        return list(set(self.get_all_subtag(dicom_files, tag, subtag)))

    def get_common_tag(self, dicom_files, tag: str):
        """Return the most common tag value across volume for provided tag."""
        return self.slice_manager.most_common(self.get_all_tag(dicom_files, tag))

    def get_dicom_pixel_spacing(self, dicom_files):
        """Return X and Y dim pixel spacing."""
        assert all([hasattr(file, "PixelSpacing") for file in dicom_files])
        dicom_pixel_spacing = [
            list(x)
            for x in set(
                tuple(x) for x in [file.PixelSpacing[:2] for file in dicom_files]
            )
        ]
        if not "PixelSpacing" in self.allow:
            assert (
                len(dicom_pixel_spacing) == 1
            ), f"{len(dicom_pixel_spacing)} YX spacings detected: {dicom_pixel_spacing}"
        return dicom_pixel_spacing[0]

    def get_dicom_spacing(self, dicom_files: List[dcm.dataset.Dataset]):
        """Return voxel size as (Y-spacing, X-spacing, step-size)."""
        return self.get_dicom_pixel_spacing(dicom_files) + [
            self.get_step_size(dicom_files)
        ]

    def get_step_size(self, dicom_files: List[dcm.dataset.Dataset]) -> float:
        """Return step size for voxel size."""
        if all([hasattr(file, "SpacingBetweenSlices") for file in dicom_files]):
            return self.slice_manager.most_common(
                self.get_all_tag(dicom_files, tag="SpacingBetweenSlices")
            )
        else:
            if "SpacingBetweenSlices" in self.allow and len(dicom_files) > 1:
                # log.warning('SpacingBetweenSlices missing from DICOM file data, using distance between ImagePositionPatient[2] values')
                steps_between_slice_position = self.get_all_tag_idx(
                    dicom_files, tag="ImagePositionPatient", idx=2
                )
                steps_between_slice_position = [
                    abs(
                        round(
                            (
                                steps_between_slice_position[num]
                                - steps_between_slice_position[num - 1]
                            ),
                            5,
                        )
                    )
                    for num in range(1, len(steps_between_slice_position))
                ]
                steps_between_slice_position = [
                    step for step in steps_between_slice_position if step != 0
                ]
                return self.slice_manager.most_common(steps_between_slice_position)
            else:
                if "SpacingBetweenSlices" in self.allow and len(dicom_files) == 1:
                    return self.slice_manager.most_common(
                        self.get_all_tag(dicom_files, tag="SliceThickness"))
                else:
                    assert (
                        False
                    ), "ReadDicom requires some measure of spacing between slices"
