from typing import List, Tuple

import cv2 as cv
import scipy
import numpy as np

from dicom_manager.file_viewers.rgb_viewer import RGBViewer
from dicom_manager.file_readers.read_image_label_pair import ReadImageLabelPair


class ReadMultiImageLabelPair(ReadImageLabelPair):
    """
    If aligning, will reformat read_pair_2 to align with read_pair_1.
    (so order input accordingly)
    """

    def __init__(
        self,
        read_pair_1: ReadImageLabelPair,
        read_pair_2: ReadImageLabelPair,
        transparency: float = 0.3,
        allow: list = [],
        align: bool = False,
        align_by_roi: bool = False,
        contours: bool = False,
        contour_thickness: int = 2,
        **kwargs
    ):
        self.allow = allow
        self.validate(read_pair_1, read_pair_2)

        if align:
            self.read_pair_1, self.read_pair_2 = self.align_volumes(
                read_pair_1, read_pair_2, align_by_roi
            )
        else:
            self.read_pair_1, self.read_pair_2 = read_pair_1, read_pair_2

        self.transparency = transparency
        self.contours = contours
        self.contour_thickness = contour_thickness
        self.colors = self.get_color_tuples_from_hex(**kwargs)
        self.dsc = self.get_dsc()
        self.arr = self.build_rgb_overlay(**kwargs)
        self.viewer = RGBViewer(
            self.arr, self.foreground, read_pair_1.read_dicom_image.spacing
        )

    def validate(
        self,
        read_pair_1: ReadImageLabelPair,
        read_pair_2: ReadImageLabelPair,
    ):
        if not "PixelSpacing" in self.allow:
            assert (
                read_pair_1.spacing == read_pair_2.spacing
            ), f"spacing between pairs does not match: {read_pair_1.spacing}:{read_pair_2.spacing}"
        if not "pixel_array.shape" in self.allow:
            assert (
                read_pair_1.read_dicom_image.arr.shape[-2]
                == read_pair_2.read_dicom_image.arr.shape[-2]
            ), f"array shape mismatch: {read_pair_1.read_dicom_image.arr.shape[-2]}:{read_pair_1.read_dicom_image.arr.shape[-2]}"  # double check
        assert (
            len(self.get_label_values(read_pair_1.read_dicom_label.arr)) == 1
            and self.get_label_values(read_pair_1.read_dicom_label.arr)[0] == 1
        )
        assert (
            len(self.get_label_values(read_pair_2.read_dicom_label.arr)) == 1
            and self.get_label_values(read_pair_2.read_dicom_label.arr)[0] == 1
        )

    # assert array slices are same XY dims
    # assert slice positions align

    def align_volumes(
        self,
        read_pair_1: ReadImageLabelPair,
        read_pair_2: ReadImageLabelPair,
        align_by_roi: bool,
    ) -> Tuple[ReadImageLabelPair]:
        """
        Build empty array of read_pair_1.arr.shape and fill with aligned read_pair_2 values.
        """
        # if self.check_volumes_aligned(read_pair_1, read_pair_2):
        #     return read_pair_1, read_pair_2
        read_pair_2 = self.match_spacing(read_pair_1, read_pair_2)
        pair_1_centroid, pair_2_centroid = self.get_centroids(
            read_pair_1, read_pair_2, align_by_roi
        )
        shift_positions = self.get_shift_positions(
            read_pair_1, read_pair_2, pair_1_centroid, pair_2_centroid
        )
        read_pair_2 = self.write_shifted(read_pair_1, read_pair_2, shift_positions)
        # reset arr?
        return read_pair_1, read_pair_2

    def rewrite_spacing_tags(
        self, read_pair_1: ReadImageLabelPair, read_pair_2: ReadImageLabelPair
    ) -> ReadImageLabelPair:
        for pair_1_file, pair_2_file in zip(
            read_pair_1.read_dicom_image.files, read_pair_2.read_dicom_image.files
        ):
            pair_2_file.PixelSpacing = pair_1_file.PixelSpacing
            pair_2_file.SpacingBetweenSlices = pair_1_file.SpacingBetweenSlices
        for pair_1_file, pair_2_file in zip(
            read_pair_1.read_dicom_label.files, read_pair_2.read_dicom_label.files
        ):
            pair_2_file.PixelSpacing = pair_1_file.PixelSpacing
            pair_2_file.SpacingBetweenSlices = pair_1_file.SpacingBetweenSlices
        return read_pair_2

    def match_spacing(
        self, read_pair_1: ReadImageLabelPair, read_pair_2: ReadImageLabelPair
    ) -> ReadImageLabelPair:
        """Return read_pair_2 resized to align with read_pair_1 voxel size."""
        if read_pair_1.spacing == read_pair_2.spacing:
            return read_pair_2
        resize_dims = [j / i for i, j in zip(read_pair_1.spacing, read_pair_2.spacing)]
        read_pair_2 = self.rewrite_spacing_tags(read_pair_1, read_pair_2)

        read_pair_2.read_dicom_image.arr = scipy.ndimage.zoom(
            read_pair_2.read_dicom_image.arr, resize_dims
        )
        read_pair_2.read_dicom_label.arr = scipy.ndimage.zoom(
            read_pair_2.read_dicom_label.arr, resize_dims
        )
        return read_pair_2

    def check_volumes_aligned(
        self, read_pair_1: ReadImageLabelPair, read_pair_2: ReadImageLabelPair
    ) -> bool:
        if not read_pair_1.spacing == read_pair_2.spacing:
            return False
        if not read_pair_1.arr.shape == read_pair_2.arr.shape:
            return False
        return True

    def get_centroids(
        self,
        read_pair_1: ReadImageLabelPair,
        read_pair_2: ReadImageLabelPair,
        align_by_roi: bool,
    ):
        if align_by_roi:
            return [
                np.floor(val)
                for val in self.get_centroids_by_roi(read_pair_1, read_pair_2)
            ]
        else:
            return [
                np.floor(val)
                for val in self.get_centroids_by_position_patient(
                    read_pair_1, read_pair_2
                )
            ]

    def get_centroids_by_position_patient(self) -> List[int]:
        """Return coordinate shift to align read_pair_2 array with read_pair_1 by ImagePositionPatient."""
        assert False, "build out to return center of ImagePositionPatient... "

    def get_centroids_by_roi(
        self, read_pair_1: ReadImageLabelPair, read_pair_2: ReadImageLabelPair
    ) -> Tuple[float]:
        """
        Return coordinate shift to align read_pair_2 array with read_pair_1 by ROI center of mass.
        Return
        """
        pair_1_centroid = read_pair_1.viewer.get_center_of_mass(
            read_pair_1.read_dicom_label.arr
        )
        pair_2_centroid = read_pair_2.viewer.get_center_of_mass(
            read_pair_2.read_dicom_label.arr
        )
        return pair_1_centroid, pair_2_centroid

    def write_shifted(
        self,
        read_pair_1: ReadImageLabelPair,
        read_pair_2: ReadImageLabelPair,
        shift_positions: tuple,
    ) -> ReadImageLabelPair:
        empty_array = np.zeros(read_pair_1.read_dicom_image.arr.shape)
        read_pair_2.read_dicom_image.arr = self.write_shifted_pixel_data(
            np.copy(empty_array), read_pair_2.read_dicom_image.arr, shift_positions
        )
        read_pair_2.read_dicom_label.arr = self.write_shifted_pixel_data(
            np.copy(empty_array), read_pair_2.read_dicom_label.arr, shift_positions
        )
        return read_pair_2

    def write_shifted_pixel_data(
        self,
        cropped_and_padded: np.array,
        input_array: np.array,
        shift_positions: tuple,
    ) -> np.array:
        (
            output_arr_start,
            output_arr_stop,
            input_arr_start,
            input_arr_stop,
        ) = shift_positions

        cropped_and_padded[
            input_arr_start[0] : input_arr_stop[0],
            input_arr_start[1] : input_arr_stop[1],
            input_arr_start[2] : input_arr_stop[2],
        ] = input_array[
            output_arr_start[0] : output_arr_stop[0],
            output_arr_start[1] : output_arr_stop[1],
            output_arr_start[2] : output_arr_stop[2],
        ]
        return cropped_and_padded

    def get_shift_positions(
        self,
        read_pair_1: ReadImageLabelPair,
        read_pair_2: ReadImageLabelPair,
        pair_1_centroid: Tuple[int],
        pair_2_centroid: Tuple[int],
    ) -> Tuple[int]:
        input_arr_start = list(
            np.array(
                [int(b - a) for a, b in zip(pair_1_centroid, pair_2_centroid)]
            ).clip(0)
        )
        input_arr_stop = [
            int(b - a + c) if (b - a + c) < d else d
            for a, b, c, d in zip(
                pair_1_centroid,
                pair_2_centroid,
                list(read_pair_1.read_dicom_image.arr.shape),
                list(read_pair_2.read_dicom_image.arr.shape),
            )
        ]

        output_arr_start = list(
            np.array(
                [int(a - b) for a, b in zip(pair_1_centroid, pair_2_centroid)]
            ).clip(0)
        )
        output_arr_stop = [
            int(a - b + d) if (a - b + d) < c else c
            for a, b, c, d in zip(
                pair_1_centroid,
                pair_2_centroid,
                list(read_pair_1.read_dicom_image.arr.shape),
                list(read_pair_2.read_dicom_image.arr.shape),
            )
        ]
        return input_arr_start, input_arr_stop, output_arr_start, output_arr_stop

    def get_dsc(self) -> float:
        """Return DSC between 2 labels."""
        overlap = (
            self.read_pair_1.read_dicom_label.arr
            * self.read_pair_2.read_dicom_label.arr
        )
        return (2 * np.sum(overlap)) / (
            np.sum(self.read_pair_1.read_dicom_label.arr)
            + np.sum(self.read_pair_2.read_dicom_label.arr)
        )

    # SET ARR STUFFS --separate out

    def draw_multi_label_array_contours(self, multi_label_array: np.array, overlap: np.array, pixel_data_1: np.array, pixel_data_2: np.array) -> np.array:
        """Draw contour of 1,2,3 values on multi_label_array."""

        #multi_label_array = cv.cvtColor(multi_label_array, cv.COLOR_BGR2RGB)
        temp_array = np.zeros((overlap.shape[0],overlap.shape[1]))
        print(temp_array.shape)
        #temp_array = cv.cvtColor(temp_array, cv.COLOR_BGR2RGB)
        print(overlap.shape)

        for sl in range(overlap.shape[2]):
            #overlap_temp = overlap.astype(np.uint8)[:,:,sl]
            pixel_data_1_temp = pixel_data_1.astype(np.uint8)[:,:,sl]
            pixel_data_2_temp = pixel_data_2.astype(np.uint8)[:,:,sl]
            #ret,thresh = cv.threshold(pixel_data_2_temp,0.5,255,cv.THRESH_BINARY)

            #contour_overlap, _ = cv.findContours(overlap_temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour_pixel_data_1, _ = cv.findContours(pixel_data_1_temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour_pixel_data_2, _ = cv.findContours(pixel_data_2_temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # need to consider what order we're drawing for overwriting of values 
            # might want to play around with line thickness - last arg in drawContours(), or incorporate this as a passable arg 
            #cv.drawContours(temp_array, contour_overlap, -1, 1, self.contour_thickness)
            #cv.drawContours(temp_array, contour_pixel_data_1, -1, 2, self.contour_thickness)
            cv.drawContours(temp_array, contour_pixel_data_2, 0, 3, self.contour_thickness)

            multi_label_array[:,:,sl] = temp_array

        return multi_label_array
     

    def draw_multi_label_array_overlay(self, multi_label_array: np.array, overlap: np.array, pixel_data_1: np.array, pixel_data_2: np.array) -> np.array:
        """Fill overlay of 1,2,3 values on multi_label_array."""
        multi_label_array[overlap > 0] = 1
        multi_label_array[pixel_data_1 > pixel_data_2] = 2
        multi_label_array[pixel_data_1 < pixel_data_2] = 3
        return multi_label_array

    def get_multi_label_array(
        self, pixel_data_1: np.array, pixel_data_2: np.array, contour: bool = False
    ) -> np.array:
        """Return 2d array with overlap=1, pixel_data_1=2, pixel_data_2=3."""
        assert pixel_data_1.shape == pixel_data_2.shape
        multi_label_array = np.zeros(pixel_data_1.shape)
        overlap = pixel_data_1 * pixel_data_2
        if contour:
            return self.draw_multi_label_array_contours(multi_label_array, overlap, pixel_data_1, pixel_data_2)
        else:
            return self.draw_multi_label_array_overlay(multi_label_array, overlap, pixel_data_1, pixel_data_2)

    # might want to retitle this function for clarity now that it can handle contours in addition to overlay
    def build_rgb_overlay(self, **kwargs) -> np.array:

        rgb_array = self.convert_grayscale_to_rgb(self.read_pair_1.read_dicom_image.arr)
        multi_label_array = self.get_multi_label_array(
            self.read_pair_1.read_dicom_label.arr, self.read_pair_2.read_dicom_label.arr, self.contours
        )
        self.foreground = np.copy(multi_label_array)
        self.foreground[multi_label_array > 0] = 1

        rgb_array = self.dampen_mask_regions(rgb_array, self.foreground) # might want to only do this for overlay (not contour)
        rgb_array = self.color_in_labels(rgb_array, multi_label_array)
        if "side_by_side" in kwargs and kwargs["side_by_side"]:
            gray_array = self.convert_grayscale_to_rgb(self.read_dicom_image.arr)
            rgb_array = np.concatenate((gray_array, rgb_array), axis=1)
        rgb_array = np.swapaxes(rgb_array, 0, -1)
        return rgb_array