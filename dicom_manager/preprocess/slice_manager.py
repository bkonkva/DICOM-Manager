"""Slice operations for DICOM position sorting/etc."""

import numpy as np


class SliceManager:
    def most_common(self, lst: list):
        return max(set(lst), key=lst.count)

    def closest(self, lst, K):
        return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

    def get_best_positions(self, dicom_slice_positions: list, step_size: float) -> list:
        position_offset = self.get_position_offset(dicom_slice_positions, step_size)
        best_range = self.get_best_range(
            dicom_slice_positions, position_offset, step_size
        )
        return [
            best_range[0] + step_size * slice_num
            for slice_num in range(1 + int((best_range[1] - best_range[0]) / step_size))
        ]

    def get_missing_positions(
        self, dicom_slice_positions: list, step_size: float
    ) -> list:
        best_positions = self.get_best_positions(dicom_slice_positions, step_size)
        return [loc for loc in best_positions if not loc in dicom_slice_positions]

    def get_extra_positions(
        self, dicom_slice_positions: list, step_size: float
    ) -> list:
        best_positions = self.get_best_positions(dicom_slice_positions, step_size)
        return [loc for loc in dicom_slice_positions if not loc in best_positions]

    def get_position_offset(
        self, dicom_slice_positions: list, step_size: float
    ) -> float:
        offsets = [
            dicom_slice_location % step_size
            for dicom_slice_location in dicom_slice_positions
        ]
        offset_counts = {offset: offsets.count(offset) for offset in offsets}
        return list(offset_counts.keys())[
            list(offset_counts.values()).index(max(list(offset_counts.values())))
        ]

    def get_best_range(
        self, dicom_slice_positions: list, position_offset: float, step_size: float
    ) -> tuple:
        start_position = np.amin(dicom_slice_positions) - (
            (np.amin(dicom_slice_positions) % step_size) - position_offset
        )
        stop_position = np.amax(dicom_slice_positions) - (
            (np.amax(dicom_slice_positions) % step_size) - position_offset
        )
        if stop_position < np.amax(dicom_slice_positions):
            stop_position += step_size
        return (start_position, stop_position)

    def get_next_best_positions(
        self, dicom_slice_positions: list, best_positions: list
    ) -> list:
        return [self.closest(dicom_slice_positions, loc) for loc in best_positions]
