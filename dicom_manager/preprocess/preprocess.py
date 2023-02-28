import pathlib
import os
import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_orthogonal()
settings.disable_validate_slicecount()
settings.disable_validate_orientation()

from tqdm import tqdm
from glob import glob
from natsort import natsorted

from dicom_manager.directory_manager import DirManager
from dicom_manager.file_readers.read_dicom import ReadDicom
from dicom_manager.file_readers.read_nifti import ReadNifti
from dicom_manager.file_readers.read_image_label_pair import ReadImageLabelPair
from dicom_manager.preprocess.dicom_finder import DicomFinder

log = logging.getLogger(__name__)


class PreProcess:

    dicom_finder = DicomFinder()

    def __init__(self, DIR_RAW, add_subgroup=False, value_clip=False, allow=[]):
        self.RAW_DICOM_DIRS = self.dicom_finder.get_dicom_dirs(DIR_RAW)
        DIR_PREPROCESSED = (
            "raw".join(DIR_RAW.split("raw")[:-1]) + "preprocessed"
        )  # join in case of 2nd "raw" dir somewhere in directory structure
        # option for images and labels?
        DIR_PRE_DICOM_IMAGES, DIR_PRE_DICOM_LABELS = self.get_preprocessed_dir(
            DIR_PREPROCESSED, image_type="dicom", add_subgroup=add_subgroup
        )
        DIR_PRE_NIFTI_IMAGES, DIR_PRE_NIFTI_LABELS = self.get_preprocessed_dir(
            DIR_PREPROCESSED, image_type="nifti", add_subgroup=add_subgroup
        )
        self.DIRS = DirManager(
            DIR_RAW=DIR_RAW,
            DIR_PREPROCESSED=DIR_PREPROCESSED,
            DIR_PRE_DICOM_IMAGES=DIR_PRE_DICOM_IMAGES,
            DIR_PRE_DICOM_LABELS=DIR_PRE_DICOM_LABELS,
            DIR_PRE_NIFTI_IMAGES=DIR_PRE_NIFTI_IMAGES,
            DIR_PRE_NIFTI_LABELS=DIR_PRE_NIFTI_LABELS,
        )
        self.configure_logger(DIR_PREPROCESSED, add_subgroup)
        self.value_clip = value_clip
        self.allow = allow

    def configure_logger(self, log_directory: pathlib.Path, add_subgroup) -> None:
        log_date = datetime.now()
        log_date = "_".join(
            [
                str(log_date.year),
                str(log_date.month),
                str(log_date.day),
                str(log_date.hour),
                str(log_date.minute),
            ]
        )
        if add_subgroup:
            log_date = "_".join([add_subgroup, log_date])
        # set up log config
        logging.basicConfig(
            filename=os.path.join(
                log_directory,
                f"preprocess_{log_date}.log",
            ),
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def get_preprocessed_dir(
        self, DIR_PREPROCESSED: pathlib.Path, image_type: str, add_subgroup
    ) -> Tuple[pathlib.Path]:
        if add_subgroup:
            return os.path.join(
                DIR_PREPROCESSED, image_type, "images", add_subgroup
            ), os.path.join(DIR_PREPROCESSED, image_type, "labels", add_subgroup)
        else:
            return os.path.join(DIR_PREPROCESSED, image_type, "images"), os.path.join(
                DIR_PREPROCESSED, image_type, "labels"
            )

    def clean_dicom(
        self, raw_dicom_dir: pathlib.Path, clean_dicom_dir: pathlib.Path, is_label: bool
    ) -> None:
        raw = ReadDicom(raw_dicom_dir, value_clip=self.value_clip, allow=self.allow)
        raw.prep_for_nifti(raw.files, is_label)
        raw.writer.save_all(raw.files, clean_dicom_dir)
        log.info(f"{raw_dicom_dir} preprocessed as DICOM to {clean_dicom_dir}")

    def get_clean_dicom_dir(self, case_name: str, is_label=bool) -> pathlib.Path:
        if is_label:
            return os.path.join(self.DIRS.DIR_PRE_DICOM_LABELS, case_name)
        else:
            return os.path.join(self.DIRS.DIR_PRE_DICOM_IMAGES, case_name)

    def get_case_name(self, raw_dicom_dir, case_name_fn) -> str:
        if not case_name_fn:
            return os.path.basename(raw_dicom_dir)
        else:
            return case_name_fn(raw_dicom_dir)

    def get_nifti_write_path(self, case_name: str, is_label: bool) -> pathlib.Path:
        if is_label:
            return os.path.join(
                self.DIRS.DIR_PRE_NIFTI_LABELS, f"{case_name}.nii.gz"
                #self.DIRS.DIR_PRE_NIFTI_LABELS, f"{case_name}_0000.nii.gz"
            )
        else:
            return os.path.join(
                self.DIRS.DIR_PRE_NIFTI_IMAGES, f"{case_name}_0000.nii.gz"
            )

    def write_nifti(
        self, clean_dicom_dir: pathlib.Path, case_name: str, is_label=bool
    ) -> None:

        nifti_write_path = self.get_nifti_write_path(case_name, is_label)
        print(f"{clean_dicom_dir} preprocessing as NIFTI to {nifti_write_path}")
        print(len(os.listdir(clean_dicom_dir)))
        if len(os.listdir(clean_dicom_dir)) > 1:
            dicom2nifti.dicom_series_to_nifti(clean_dicom_dir, nifti_write_path)
        else:
            dicom2nifti.settings.disable_validate_slicecount()
            dicom2nifti.dicom_series_to_nifti(clean_dicom_dir, nifti_write_path)
        log.info(f"{clean_dicom_dir} preprocessed as NIFTI to {nifti_write_path}")

    def build_legend(self, dicom_dir: pathlib.Path, pixel_array: np.array, **kwargs):
        if "legend" in kwargs and not kwargs["legend"]:
            return kwargs
        if not "legend_size" in kwargs:
            kwargs["legend_size"] = 12
        if not "legend" in kwargs:
            kwargs["legend"] = [
                [
                    f"{os.path.basename(dicom_dir.rstrip('/'))}: {np.amin(pixel_array)}:{np.amax(pixel_array)}",
                    "#808080",
                ]
            ]
            return kwargs
        return kwargs

    def preview_raw_dicom(self, value_clip=False, **kwargs) -> None:
        """Generate orthoview previews of original/raw DICOM files."""
        for raw_dicom_dir in tqdm(
            natsorted(glob(os.path.join(self.DIRS.DIR_RAW, "*/"))),
            desc="generating previews of raw data...",
        ):
            raw_dicom = ReadDicom(
                raw_dicom_dir, allow=self.allow, value_clip=value_clip
            )
            raw_dicom.viewer.orthoview(
                **self.build_legend(raw_dicom_dir, raw_dicom.arr, **kwargs)
            )

    def preview_preprocessed_dicom(self, value_clip=False, **kwargs) -> None:
        """Generate orthoview previews of preprocessed DICOM files."""
        for clean_dicom_dir in tqdm(
            natsorted(glob(os.path.join(self.DIRS.DIR_PRE_DICOM_IMAGES, "*/")))
            + natsorted(glob(os.path.join(self.DIRS.DIR_PRE_DICOM_LABELS, "*/"))),
            desc="generating previews of preprocessed DICOM data...",
        ):
            clean_dicom = ReadDicom(
                clean_dicom_dir, allow=self.allow, value_clip=value_clip
            )
            clean_dicom.viewer.orthoview(
                **self.build_legend(clean_dicom_dir, clean_dicom.arr, **kwargs)
            )

    def preview_preprocessed_dicom_pair(self, value_clip=False, **kwargs) -> None:
        """Generate orthoview previews of preprocessed DICOM files."""
        for clean_dicom_dir in tqdm(
            natsorted(glob(os.path.join(self.DIRS.DIR_PRE_DICOM_IMAGES, "*/"))),
            desc="generating previews of preprocessed DICOM data...",
        ):
            clean_dicom_image = ReadDicom(
                clean_dicom_dir, allow=self.allow, value_clip=value_clip
            )
            clean_dicom_label = ReadDicom(
                "labels".join(clean_dicom_dir.split("images")),
                allow=self.allow,
                value_clip=value_clip,
            )
            clean_dicom_pair = ReadImageLabelPair(clean_dicom_image, clean_dicom_label)
            clean_dicom_pair.viewer.orthoview(
                **self.build_legend(clean_dicom_dir, clean_dicom_image.arr, **kwargs)
            )

    def preview_preprocessed_nifti(self, value_clip=False, **kwargs) -> None:
        """Generate orthoview previews of preprocessed NIFTI files."""
        for nifti_path in tqdm(
            natsorted(glob(os.path.join(self.DIRS.DIR_PRE_NIFTI_IMAGES, "*.nii.gz")))
            + natsorted(glob(os.path.join(self.DIRS.DIR_PRE_NIFTI_LABELS, "*.nii.gz"))),
            desc="generating previews of preprocessed NIFTI data...",
        ):
            nifti_file = ReadNifti(nifti_path, value_clip=value_clip)
            nifti_file.viewer.orthoview(
                **self.build_legend(nifti_path, nifti_file.arr, **kwargs)
            )

    def preview_preprocessed_dicom_pair(self, value_clip=False, **kwargs) -> None:
        """Generate orthoview previews of preprocessed DICOM files."""
        for nifti_path in tqdm(
            natsorted(glob(os.path.join(self.DIRS.DIR_PRE_NIFTI_IMAGES, "*.nii.gz"))),
            desc="generating previews of preprocessed DICOM data...",
        ):
            nifti_file_image = ReadNifti(nifti_path, value_clip=value_clip)
            nifti_file_label = ReadNifti(
                "labels".join(nifti_path.split("images")), value_clip=value_clip
            )
            nifti_pair = ReadImageLabelPair(nifti_file_image, nifti_file_label)
            nifti_pair.viewer.orthoview(
                **self.build_legend(nifti_path, nifti_file_image.arr, **kwargs)
            )

    ##this is extra code (not being called)
    def check_path_is_label(
        self, raw_dicom_dir: pathlib.Path, label_identifier
    ) -> bool:
        if not label_identifier:
            return False
        return label_identifier(raw_dicom_dir)

    def preprocess(self, case_name_fn=False, label_identifier=False) -> None:
        for raw_dicom_dir in tqdm(
            natsorted(self.RAW_DICOM_DIRS), desc="preprocessing..."
        ):
            # try:
            is_label = label_identifier #self.check_path_is_label(raw_dicom_dir, label_identifier)
            case_name = self.get_case_name(raw_dicom_dir, case_name_fn)
            print(case_name)
            clean_dicom_dir = self.get_clean_dicom_dir(
                case_name,
                is_label,
            )
            self.clean_dicom(raw_dicom_dir, clean_dicom_dir, is_label)
            self.write_nifti(clean_dicom_dir, case_name, is_label)
            # except Exception as e:
            #     log.exception(e)
            #     print(f'ERROR converting {raw_dicom_dir}: {e}')


if __name__ == "__main__":
    main()
