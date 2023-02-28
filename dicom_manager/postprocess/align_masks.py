import os
import pathlib
from copy import deepcopy

from natsort import natsorted
from tqdm import tqdm
import numpy as np

from dicom_manager.file_readers.read_multi_image_label_pair import (
    ReadMultiImageLabelPair,
)
from dicom_manager.file_readers.read_dicom import ReadDicom
from dicom_manager.file_readers.read_image_label_pair import ReadImageLabelPair
from dicom_manager.directory_manager import DirManager

from matplotlib import pyplot as plt


class AlignMasks:
    def __init__(self, DIR_1: pathlib.Path, DIR_2: pathlib.Path, allow: list = []):
        self.DIRS = DirManager(
            DIR_IMAGE_1=os.path.join(DIR_1, "images"),
            DIR_LABEL_1=os.path.join(DIR_1, "labels"),
            DIR_IMAGE_2=os.path.join(DIR_2, "images"),
            DIR_LABEL_2=os.path.join(DIR_2, "labels"),
            DIR_IMAGE_2_SHIFTED=os.path.join(DIR_2, "images_shifted"),
            DIR_LABEL_2_SHIFTED=os.path.join(DIR_2, "labels_shifted"),
        )
        self.allow = allow

    def write_pixel_data_to_dicom(
        self, read_dicom_1: ReadDicom, read_dicom_2: ReadDicom, slice_num: int
    ):
        dicom_file = deepcopy(read_dicom_1.files[slice_num])
        dicom_file = read_dicom_1.writer.decompress_dicom(dicom_file)
        dicom_file = read_dicom_1.writer.write_array_to_dicom(
            read_dicom_2.arr[..., slice_num], dicom_file
        )
        return dicom_file

    def save_shifted(
        self, case: str, pair_1=ReadImageLabelPair, pair_2=ReadImageLabelPair
    ) -> None:
        pair_2.read_dicom_image.files = [
            self.write_pixel_data_to_dicom(
                pair_1.read_dicom_image,
                pair_2.read_dicom_image,
                slice_num,
            )
            for slice_num in range(pair_2.read_dicom_image.arr.shape[-1])
        ]
        pair_2.read_dicom_image.writer.save_all(
            pair_2.read_dicom_image.files,
            os.path.join(self.DIRS.DIR_IMAGE_2_SHIFTED, case),
        )

        pair_2.read_dicom_label.files = [
            self.write_pixel_data_to_dicom(
                pair_1.read_dicom_label,
                pair_2.read_dicom_label,
                slice_num,
            )
            for slice_num in range(pair_2.read_dicom_label.arr.shape[-1])
        ]
        pair_2.read_dicom_label.writer.save_all(
            pair_2.read_dicom_label.files,
            os.path.join(self.DIRS.DIR_LABEL_2_SHIFTED, case),
        )

        # then load and preview with multi-label code...

    def align(self) -> None:
        for case in tqdm(
            natsorted(os.listdir(self.DIRS.DIR_IMAGE_1)),
            desc="aligning masks...",
        ):
            try:
                read_dicom_image_1 = ReadDicom(
                    os.path.join(self.DIRS.DIR_IMAGE_1, case), allow=self.allow
                )
                read_dicom_label_1 = ReadDicom(
                    os.path.join(self.DIRS.DIR_LABEL_1, case), allow=self.allow
                )
                pair_1 = ReadImageLabelPair(
                    read_dicom_image_1, read_dicom_label_1, allow=self.allow
                )

                read_dicom_image_2 = ReadDicom(
                    os.path.join(self.DIRS.DIR_IMAGE_2, case), allow=self.allow
                )
                read_dicom_label_2 = ReadDicom(
                    os.path.join(self.DIRS.DIR_LABEL_2, case), allow=self.allow
                )
                pair_2 = ReadImageLabelPair(
                    read_dicom_image_2, read_dicom_label_2, allow=self.allow
                )

                multi_pair = ReadMultiImageLabelPair(
                    pair_1, pair_2, align=True, align_by_roi=True, allow=self.allow
                )

                self.save_shifted(case, multi_pair.read_pair_1, multi_pair.read_pair_2)
            # break
            except FileNotFoundError as e:
                if not "unpaired" in self.allow:
                    assert False, f"MISSING MATCH: {e}"
                print(f"MISSING MATCH: {e}")
