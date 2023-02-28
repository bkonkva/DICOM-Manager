import pathlib
import os
import copy
import csv
import json
from typing import List, Dict

import numpy as np
from glob import glob
from tqdm import tqdm
import pydicom as dcm

from dicom_manager.directory_manager import DirManager
from dicom_manager.file_readers.read_nifti import ReadNifti
from dicom_manager.file_readers.read_dicom import ReadDicom, ReadRawDicom
from dicom_manager.file_readers.read_image_label_pair import ReadImageLabelPair
from dicom_manager.file_writers.save_measurements_to_csv import MeasurementSaver
from dicom_manager.file_writers.save_qc_images import QCSaver

# both this and preprocess image/label combo should inheret from image_label_reader
class PostProcess:

    qc_saver = QCSaver()
    measurements = MeasurementSaver()
    volume = {}

    def __init__(self, DIR_PRE_DICOM, DIR_PRE_NIFTI, DIR_INFERENCE, allow=[]):
        missing_inference_files = self.verify_inference_complete(DIR_PRE_NIFTI, DIR_INFERENCE, allow)
        DIR_POSTPROCESS = "postprocessed".join(DIR_INFERENCE.split("inference"))
        DIR_QC = os.path.join(DIR_POSTPROCESS, "QC")
        DIR_MEASUREMENTS = os.path.join(DIR_POSTPROCESS, "MEASUREMENTS")
        self.DIRS = DirManager(
            DIR_PRE_DICOM=DIR_PRE_DICOM,
            DIR_PRE_NIFTI=DIR_PRE_NIFTI,
            DIR_INFERENCE=DIR_INFERENCE,
            DIR_POSTPROCESS=DIR_POSTPROCESS,
            DIR_QC=DIR_QC,
            DIR_MEASUREMENTS=DIR_MEASUREMENTS,
        )
        self.allow = allow
        self.missing_inference_files = missing_inference_files

    def verify_inference_complete(
        self, DIR_PRE_NIFTI: pathlib.Path, DIR_INFERENCE: pathlib.Path, allow
    ) -> None:
        preprocessed_paths = set(
            os.path.basename(case).split("_0000.nii.gz")[0]
            for case in glob(os.path.join(DIR_PRE_NIFTI, "*.nii.gz"))
        )
        inference_paths = set(
            os.path.basename(case).split(".nii.gz")[0]
            for case in glob(os.path.join(DIR_INFERENCE, "*.nii.gz"))
        )
        missing_inference_files = preprocessed_paths.difference(inference_paths)
        if not "missing_inference" in allow:
            assert (
                len(missing_inference_files) == 0
            ), f'{len(missing_inference_files)} preprocessed files in {DIR_PRE_NIFTI} are not accounted for in {DIR_INFERENCE}: {missing_inference_files}, continue inference or pass "missing_inference" in allow list'
        return missing_inference_files


    def get_dicom_path(self, nifti_path: pathlib.Path) -> pathlib.Path:
        """Return write path for postprocessed DICOM file."""
        return os.path.join(
            self.DIRS.DIR_PRE_DICOM,
            os.path.basename(nifti_path).split("_0000.nii.gz")[0],
        )

    def get_label_path(self, nifti_path: pathlib.Path) -> pathlib.Path:
        """Return write path for postprocessed DICOM label file."""
        return os.path.join(
            self.DIRS.DIR_INFERENCE,
            ".nii.gz".join(os.path.basename(nifti_path).split("_0000.nii.gz")),
        )

    def read_files(self, nifti_path: pathlib.Path):
        nifti_read_image = ReadNifti(nifti_path)
        nifti_read_label = ReadNifti(self.get_label_path(nifti_path))
        dicom_read_image = ReadRawDicom(
            self.get_dicom_path(nifti_path), allow=self.allow
        )
        dicom_read_label = copy.deepcopy(dicom_read_image)
        return nifti_read_image, nifti_read_label, dicom_read_image, dicom_read_label

    def undo_dicom2nifti_rescale(
        self, pixel_data: np.array, dicom_files: List[dcm.dataset.Dataset]
    ) -> np.array:
        for slice_num in range(pixel_data.shape[-1]):
            if (
                hasattr(dicom_files[slice_num], "Modality")
                and dicom_files[slice_num].Modality.upper() == "CT"
            ):
                if hasattr(dicom_files[slice_num], "RescaleIntercept"):
                    pixel_data[..., slice_num] += dicom_files[
                        slice_num
                    ].RescaleIntercept
        return pixel_data

    def copy_nifti_to_dicom(
        self, nifti_read: ReadNifti, dicom_read: ReadDicom, rescale: bool = False
    ) -> ReadDicom:
        """Copy pixel data from segmentation output NIFTI file to original (raw path) DICOM meta data."""
        nifti_pixel_array = np.rot90(nifti_read.files[0].get_fdata(), k=1, axes=(0, 1))
        if rescale:
            nifti_pixel_array = self.undo_dicom2nifti_rescale(
                pixel_data=nifti_pixel_array, dicom_files=dicom_read.files
            )
        dicom_read.files = dicom_read.writer.write_array_volume_to_dicom(
            nifti_pixel_array, dicom_read.files
        )
        return dicom_read

    def postprocess(self) -> None:
        for nifti_path in tqdm(
            glob(os.path.join(self.DIRS.DIR_PRE_NIFTI, "*.nii.gz")),
            desc="postprocessing...",
        ):
            case = nifti_path.rsplit('/',1)[1].rsplit('_',1)[0]
            if ((case in self.missing_inference_files) and ("missing_inference" in self.allow)):
                continue

            nifti_image, nifti_label, dicom_image, dicom_label = self.read_files(
                nifti_path
            )
            # dicom_image = self.copy_nifti_to_dicom(nifti_image, dicom_image, rescale=True)
            # dicom_image.files = dicom_image.writer.write_array_volume_to_dicom(
            #     np.flip(dicom_image.arr, 1), dicom_image.files
            # )
            dicom_image.files = dicom_image.writer.write_array_volume_to_dicom(
                dicom_image.arr, dicom_image.files
            )

            dicom_label = self.copy_nifti_to_dicom(nifti_label, dicom_label)
            dicom_image_write_dir = os.path.join(
                self.DIRS.DIR_POSTPROCESS,
                "images",
                os.path.basename(nifti_path.split("_0000.nii.gz")[0]),
            )
            dicom_image.writer.save_all(dicom_image.files, dicom_image_write_dir)
            dicom_label_write_dir = os.path.join(
                self.DIRS.DIR_POSTPROCESS,
                "labels",
                os.path.basename(nifti_path.split("_0000.nii.gz")[0]),
            )
            dicom_label.writer.save_all(dicom_label.files, dicom_label_write_dir)

    def preview_postprocessed_dicom(self, value_clip=False, **kwargs) -> None:
        """Display segmentation mask overlay of postprocessed DICOM data as RGB."""
        if "value_clip" in kwargs:
            value_clip = kwargs["value_clip"]

        for postprocessed_dir in tqdm(
            glob(os.path.join(self.DIRS.DIR_POSTPROCESS, "images", "*/")),
            desc="generating previews of postprocessed DICOM data...",
        ):
            case_kwargs = copy.deepcopy(kwargs)
            if "legend" in case_kwargs:
                case_kwargs["legend"].append(
                    [os.path.basename(postprocessed_dir.rstrip("/")), "#808080"]
                )
            image = ReadDicom(
                postprocessed_dir, value_clip=value_clip, allow=self.allow
            )
            label = ReadDicom(
                "labels".join(postprocessed_dir.split("images")), allow=self.allow
            )
            pair = ReadImageLabelPair(image, label, **kwargs)
            pair.viewer.orthoview(**case_kwargs)

    def save_qc(self, orthoview: bool = True, value_clip=False, **kwargs) -> None:
        """Save QC images with RGB overlay of segmentation masks."""
        if "value_clip" in kwargs:
            value_clip = kwargs["value_clip"]
        for postprocessed_dir in tqdm(
            glob(os.path.join(self.DIRS.DIR_POSTPROCESS, "images", "*/")),
            desc=f"writing QC images to {self.DIRS.DIR_QC}",
        ):
            image = ReadDicom(
                postprocessed_dir, allow=self.allow, value_clip=value_clip
            )
            label = ReadDicom(
                "labels".join(postprocessed_dir.split("images")), allow=self.allow
            )
            pair = ReadImageLabelPair(image, label, **kwargs)
            self.qc_saver.save(
                os.path.join(
                    self.DIRS.DIR_QC, os.path.basename(postprocessed_dir.strip("/"))
                ),
                pair,
                orthoview=orthoview,
                **kwargs,
            )

    def get_csv_path(self, csv_name: str, single_slices: bool) -> pathlib.Path:
        """Return csv path for measurements results file."""
        if csv_name:
            return os.path.join(self.DIRS.DIR_MEASUREMENTS, csv_name)
        else:
            if single_slices==True:
                return os.path.join(self.DIRS.DIR_MEASUREMENTS, "mask_area_cm2_measurements.csv")
            else:
                return os.path.join(self.DIRS.DIR_MEASUREMENTS, "mask_volume_cm3_measurements.csv")

    def write_csv(self, csv_name: str, volume: dict, label_key: Dict[str, int], single_slices: bool) -> None:
        """Write volume measurements as csv file."""
        fieldnames = ["case"] + list(label_key.keys())
        with open(self.get_csv_path(csv_name, single_slices), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for case, label_volume_key in self.volume.items():
                writer.writerow({"case": case, **label_volume_key})

    def calculate_volume_for_all_labels(
        self, pair: ReadImageLabelPair, label_key: Dict[str, int], single_slices: bool
    ) -> Dict[str, int]:
        if single_slices==True:
            return {
                label_name: pair.get_area(
                    read_dicom_label=pair.read_dicom_label, label_value=label_value
                )
                for label_name, label_value in label_key.items()
            }    
        if single_slices==False:
            return {
                label_name: pair.get_volume(
                    read_dicom_label=pair.read_dicom_label, label_value=label_value
                )
                for label_name, label_value in label_key.items()
            }

    def parse_patient_key(self, patient_key_path: pathlib.Path) -> Dict[str, str]:
        """
        Return patienty key with original:anonymized.
        For use if processing anonymized files and require original-name results file.
        """
        with open(patient_key_path) as f:
            patient_key: dict = json.load(f)
        # return patient_key["Original PatientID: Anonymized PatientID"]
        return patient_key["Original DICOM File Path: Anonymized DICOM File Path"]

    def get_patient_id(
        self, postprocessed_dir: pathlib.Path, patient_key: Dict[str, str]
    ) -> str:
        """Return patient_id for results file."""
        if not patient_key:
            return os.path.basename(postprocessed_dir.strip("/"))
        for original_path, anonymized_path in patient_key.items():
            if (
                os.path.basename(postprocessed_dir.strip("/")).split("-")[0]
                in anonymized_path
            ):
                return original_path.split(os.path.sep)[-2]
    """
    def get_label_key_units(
        self, single_slices: bool, label_key: Dict[str, int]
    ) -> Dict:
        for key in label_key.keys():
            if single_slices==True:
                key = key + " cm^2"
            else:
                key = key + " cm^3"
        return label_key
    """
    def autocalculate(
        self,
        single_slices: bool,
        csv_name: str = False,
        label_key: Dict[str, int] = {"Segmentation Mask": 1},
        patient_key: pathlib.Path = False,
    ) -> None:
        """
        Calculate volume as cm^3 or area as cm^2 for all patients. e.g., label_values = {liver: 1, tumor: 2}.
        Builds nested dict with {case_name: {liver: 1500cm^3, tumor: 300cm^3}}....
        """
        if patient_key:
            patient_key = self.parse_patient_key(patient_key)
        for postprocessed_dir in tqdm(
            glob(os.path.join(self.DIRS.DIR_POSTPROCESS, "images", "*/")),
            desc="calculating...",
        ):
            read_dicom_image = ReadDicom(postprocessed_dir, allow=self.allow)
            read_dicom_label = ReadDicom(
                "labels".join(postprocessed_dir.split("images")), allow=self.allow
            )
            pair = ReadImageLabelPair(read_dicom_image, read_dicom_label)
            #label_key = self.get_label_key_units(single_slices, label_key)
            self.volume[
                self.get_patient_id(postprocessed_dir, patient_key)
            ] = self.calculate_volume_for_all_labels(pair=pair, label_key=label_key, single_slices=single_slices)
        self.write_csv(csv_name=csv_name, volume=self.volume, label_key=label_key, single_slices=single_slices)


# WHY ARE WE RESETTING ORIGINAL DICOM FILES WITH NIFTI ARRAY VALUES FOR POSTPROCESSED?
