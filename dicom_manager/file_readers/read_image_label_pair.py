from typing import List
import pathlib

import numpy as np
import pydicom as dcm

from dicom_manager.file_readers.read_dicom import ReadDicom
from dicom_manager.file_viewers.rgb_viewer import RGBViewer


class ReadImageLabelPair(ReadDicom):
    def __init__(
        self,
        read_dicom_image: ReadDicom,
        read_dicom_label: ReadDicom,
        transparency: float = 0.3,
        **kwargs,
    ):

        self.read_dicom_image = read_dicom_image
        self.read_dicom_label = read_dicom_label

        self.transparency = transparency
        self.colors = self.get_color_tuples_from_hex(**kwargs)
        assert (
            #read_dicom_image.spacing == read_dicom_label.spacing
            [round(float(x),2) for x in read_dicom_image.spacing] == [round(float(x),2) for x in read_dicom_label.spacing]
        ), f"image and label spacing is inconsistent {read_dicom_image.spacing}:{read_dicom_label.spacing}"
        self.spacing = read_dicom_image.spacing
        self.arr = self.build_rgb_overlay(**kwargs)
        self.viewer = RGBViewer(
            self.arr, self.read_dicom_label.arr, read_dicom_image.spacing
        )

    def get_color_tuples_from_hex(self, **kwargs) -> List[tuple]:
        """Return hex color inputs as (r,g,b) tuples. User can enter colors in hex format inside of a list, mask elements 1,2,3... will be colored with same colors[idx] color."""
        if "colors" in kwargs:
            return [
                tuple(int(color.strip("#")[i : i + 2], 16) for i in (0, 2, 4))
                for color in kwargs["colors"]
            ]
        else:
            colors = (
                "#06b70c",
                "#2c2cc9",
                "#eaf915",
                "#ef4a53",
            )  # green, blue, yellow, red
            return [
                tuple(int(color.strip("#")[i : i + 2], 16) for i in (0, 2, 4))
                for color in colors
            ]

    def get_label_values(self, dicom_label: np.array) -> List[int]:
        assert np.amin(dicom_label) >= 0, "negative values present in mask"
        return np.unique(dicom_label[dicom_label > 0])

    def normalize_image(self, grayscale_array: np.array) -> np.array:
        normalized_array = (grayscale_array - np.amin(grayscale_array)) / (
            np.amax(grayscale_array) - np.amin(grayscale_array)
        )
        normalized_array *= 255
        normalized_array = normalized_array.astype("uint8")
        return normalized_array

    def convert_grayscale_to_rgb(self, grayscale_array: np.array) -> np.array:
        return np.array(
            [np.copy(self.normalize_image(grayscale_array)) for i in range(3)]
        )

    def dampen_mask_regions(
        self, rgb_array: np.array, dicom_label: np.array
    ) -> np.array:
        """Reduce grayscale values in masked areas to allow for transparency stuffs."""
        for i in range(3):
            rgb_array[i, ...][dicom_label > 0] = (
                rgb_array[i, ...][dicom_label > 0]
                - rgb_array[i, ...][dicom_label > 0] * self.transparency
            )
        return rgb_array

    def color_in_labels(self, rgb_array: np.array, label_array: np.array) -> np.array:
        """Color in mask regions with designated color scheme."""
        for i in range(3):
            for label_value in self.get_label_values(label_array):
                rgb_array[i, ...][label_array == label_value] = rgb_array[i, ...][
                    label_array == label_value
                ] + (self.colors[int(label_value - 1)][i] * self.transparency)
        return rgb_array

    def build_rgb_overlay(self, **kwargs) -> np.array:
        rgb_array = self.convert_grayscale_to_rgb(self.read_dicom_image.arr)
        rgb_array = self.dampen_mask_regions(rgb_array, self.read_dicom_label.arr)
        rgb_array = self.color_in_labels(rgb_array, self.read_dicom_label.arr)
        if "side_by_side" in kwargs and kwargs["side_by_side"]:
            gray_array = self.convert_grayscale_to_rgb(self.read_dicom_image.arr)
            rgb_array = np.concatenate((gray_array, rgb_array), axis=1)
        rgb_array = np.swapaxes(rgb_array, 0, -1)
        return rgb_array

    def get_voxel_size(
        self, read_dicom: ReadDicom, dicom_file: dcm.dataset.Dataset
    ) -> float:
        if hasattr(dicom_file, "SpacingBetweenSlices"):
            return np.product(dicom_file.PixelSpacing[:2]) * abs(
                dicom_file.SpacingBetweenSlices
            )
        else:
            return np.product(dicom_file.PixelSpacing[:2]) * abs(
                read_dicom.parser.get_step_size(read_dicom.files)
            )

    def get_volume(self, read_dicom_label: ReadDicom, label_value: int = 1) -> float:
        """ "Return volume measurement in cm^3 for label int value passed."""
        return (np.sum(
            [
                self.get_voxel_size(read_dicom_label, file)
                * np.sum(file.pixel_array == label_value)
                for file in read_dicom_label.files
            ]
        ) / 1000)

    def get_pixel_size(
        self, dicom_file: dcm.dataset.Dataset
    ) -> float:
        return np.product(
            dicom_file.PixelSpacing[:2]
            )
        
    def get_area(self, read_dicom_label: ReadDicom, label_value: int = 1):
        """ "Return area measurement in cm^2 for label int value passed."""
        if len(read_dicom_label.files) == 1:
            for file in read_dicom_label.files:
                return ((
                        self.get_pixel_size(file)
                        * np.sum(file.pixel_array == label_value)
                ) / 100)
        elif len(read_dicom_label.files) > 1:
            area_list = []
            for file in read_dicom_label.files:
                slice_area = ((
                    self.get_pixel_size(file)
                    * np.sum(file.pixel_array == label_value))
                     / 100)
                area_list.append(slice_area)
            return area_list


# ADD ASSERT FOR EMPTY SLICES
# ADD ASSERT FOR MISSING LOCATIONS
