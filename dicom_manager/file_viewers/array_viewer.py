from typing import List, Tuple

import cv2
import numpy as np

from dicom_manager.file_viewers.array_plotter import ArrayPlotter


class ArrayViewer:

    plotter = ArrayPlotter()

    def __init__(self, arr, spacing):
        self.arr = arr
        self.spacing = spacing
        return

    def print_range(self) -> None:
        print(f"RANGE: {np.amin(self.arr)}:{np.amax(self.arr)}")

    def get_transverse(self, arr: np.array, resize_dims: List[int]) -> np.array:
        """Return transverse slice through center for orthogonal preview."""
        transverse = arr[..., int(arr.shape[2] / 2)]
        return cv2.resize(
            transverse,
            dsize=(resize_dims[1], resize_dims[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    def get_sagittal(self, arr: np.array, resize_dims: List[int]) -> np.array:
        """Return sagittal slice through center for orthogonal preview."""
        sagittal = np.rot90(arr[:, int(arr.shape[0] / 2), :], k=1, axes=(0, 1))
        return cv2.resize(
            sagittal,
            dsize=(resize_dims[1], resize_dims[2]),
            interpolation=cv2.INTER_CUBIC,
        )

    def get_coronal(self, arr: np.array, resize_dims: List[int]) -> np.array:
        """Return coronal slice through center for orthogonal preview."""
        coronal = np.rot90(arr[int(arr.shape[1] / 2), ...], k=1, axes=(0, 1))
        return cv2.resize(
            coronal,
            dsize=(resize_dims[0], resize_dims[2]),
            interpolation=cv2.INTER_CUBIC,
        )

    def get_resize_dimensions(self) -> List[int]:
        resize_dims = np.multiply(self.arr.shape, self.spacing)
        return [abs(int(dim)) for dim in resize_dims]

    def get_orthogonal_slices(self) -> Tuple[np.array]:
        """Return orthogonal slices for orthoview preview."""
        resize_dims = self.get_resize_dimensions()
        transverse = self.get_transverse(self.arr, resize_dims)
        sagittal = self.get_sagittal(self.arr, resize_dims)
        coronal = self.get_coronal(self.arr, resize_dims)
        return transverse, sagittal, coronal

    def orthoview(self, **kwargs) -> None:
        """Plots orthognal slice preview of image volume."""
        if "print_range" in kwargs and kwargs["print_range"]:
            self.print_range()
        transverse, sagittal, coronal = self.get_orthogonal_slices()
        self.plotter.plot_images([transverse, coronal, sagittal], **kwargs)
