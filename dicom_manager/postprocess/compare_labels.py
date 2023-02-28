import os
from typing import Tuple

from natsort import natsorted
from tqdm import tqdm
from copy import deepcopy

from dicom_manager.file_readers.read_multi_image_label_pair import (
    ReadMultiImageLabelPair,
)
from dicom_manager.directory_manager import DirManager
from dicom_manager.postprocess.postprocess import PostProcess
from dicom_manager.file_readers.read_multi_image_label_pair import (
    ReadMultiImageLabelPair,
)
from dicom_manager.file_readers.read_image_label_pair import (
    ReadImageLabelPair,
)
from dicom_manager.file_readers.read_dicom import ReadDicom


class CompareLabels(PostProcess):

    dsc = {}

    def __init__(
        self,
        DIR_IMAGES_1,
        DIR_LABELS_1,
        DIR_LABELS_2,
        DIR_POSTPROCESS,
        allow=[],
    ):

        DIR_QC = os.path.join(DIR_LABELS_2, "QC")
        DIR_MEASUREMENTS = os.path.join(DIR_POSTPROCESS, "MEASUREMENTS")
        self.DIRS = DirManager(
            DIR_IMAGES_1=DIR_IMAGES_1,
            DIR_LABELS_1=DIR_LABELS_1,
            DIR_LABELS_2=DIR_LABELS_2,
            DIR_QC=DIR_QC,
            DIR_MEASUREMENTS=DIR_MEASUREMENTS,
        )
        self.allow = allow

    def calculate_dsc(self):
        for case in tqdm()

    def read_all_data(self, case: str) -> ReadImageLabelPair:
        read_dicom_image_1 = ReadDicom(
        os.path.join(self.DIRS.DIR_IMAGES_1, case), allow=self.allow
        )
        read_dicom_label_1 = ReadDicom(
        os.path.join(self.DIRS.DIR_LABELS_1, case), allow=self.allow
        )
        pair_1 = ReadImageLabelPair(
        read_dicom_image_1, read_dicom_label_1, allow=self.allow
        )

        read_dicom_image_2 = ReadDicom(
        os.path.join(self.DIRS.DIR_IMAGES_1, case), allow=self.allow
        )
        read_dicom_label_2 = ReadDicom(
        os.path.join(self.DIRS.DIR_LABELS_2, case), allow=self.allow
        )
        pair_2 = ReadImageLabelPair(
        read_dicom_image_2, read_dicom_label_2, allow=self.allow
        )
        return ReadMultiImageLabelPair(pair_1, pair_2, allow=self.allow)

    def preview_combined_dicoms(self, **kwargs):
        for case in tqdm(
            natsorted(os.listdir(self.DIRS.DIR_IMAGES_1)), desc="generating previews..."
        ):
            try:
                multi_pair = self.read_all_data(case)
                multi_pair.viewer.orthoview(
                    **self.build_legend(case, **deepcopy(kwargs))
                )
            except FileNotFoundError as e:
                print(e)

    def build_legend(self, case: str, **kwargs) -> tuple:
        if "legend" in kwargs and not kwargs["legend"]:
            return kwargs
        if not "legend_position" in kwargs:
            kwargs["legend_position"] = "lower right"
        if not "legend_size" in kwargs:
            kwargs["legend_size"] = 10
        if not "legend" in kwargs:
            kwargs["legend"] = (
                [
                    ["overlap", "#06b70c"],
                    ["label 1", "#2c2cc9"],
                    ["label 2", "#eaf915"],
                    [case, "#808080"],
                ],
            )
        else:
            kwargs["legend"].append([case, "#808080"])
        return kwargs
