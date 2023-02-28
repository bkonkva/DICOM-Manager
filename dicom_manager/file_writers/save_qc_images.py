import pathlib
import os

import imageio
import cv2
import numpy as np
from matplotlib import pyplot as plt

from dicom_manager.file_readers.read_dicom import ReadDicom
from dicom_manager.file_viewers.array_plotter import ArrayPlotter


class QCSaver(ArrayPlotter):
    def __init__(self):
        pass

    def save(
        self, QC_PATH: pathlib.Path, qc_file: ReadDicom, orthoview: bool, **kwargs
    ) -> None:
        if orthoview:
            self.save_orthoview(QC_PATH, qc_file, **kwargs)
        else:
            self.save_full_volume(QC_PATH, qc_file, **kwargs)

    def save_image(self, write_path: pathlib.Path, image: np.array, **kwargs) -> None:
        """Write single RGB QC image."""
        self.plot_images([image], **kwargs)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.tight_layout()
        plt.savefig(write_path)
        plt.close()

    def concatenate(self, image_list) -> np.array:
        image_shapes = [image.shape for image in image_list]
        resize_dims = [image_shape[0] for image_shape in image_shapes]
        resize_dims = [
            (int(float(dim[1]) * (max(resize_dims) / ratio)), max(resize_dims))
            for dim, ratio in zip(image_shapes, resize_dims)
        ]
        image_list = [
            cv2.resize(image, dsize=dims, interpolation=cv2.INTER_CUBIC)
            for dims, image in zip(resize_dims, image_list)
        ]
        image_shapes = [image.shape for image in image_list]
        return np.concatenate(image_list, axis=1)

    def save_orthoview(self, QC_PATH, qc_file, **kwargs) -> None:
        transverse, sagittal, coronal = qc_file.viewer.get_orthogonal_slices()
        orthoview = self.concatenate([transverse, sagittal, coronal])
        write_path = os.path.join(
            os.path.dirname(QC_PATH), f"{os.path.basename(QC_PATH)}.png"
        )
        self.save_image(write_path, orthoview, **kwargs)

    def save_full_volume(self, QC_PATH, qc_file, **kwargs) -> None:
        if not os.path.exists(QC_PATH):
            os.makedirs(QC_PATH)
        for image_num in range(qc_file.arr.shape[0]):
            write_path = os.path.join(QC_PATH, f"{str(image_num).zfill(4)}.png")
            self.save_image(
                write_path,
                np.rot90(qc_file.arr[image_num, ...], k=1, axes=(0, 1)),
                **kwargs,
            )
