import pydicom as dcm

from dicom_manager.file_loaders.file_loader import FileLoader


class DicomLoader(FileLoader):
    """Load DICOM files."""

    def add_path_to_meta(self, dicom_file, target_path):
        block = dicom_file.private_block(0x000B, "CustomTags", create=True)
        block.add_new(0x01, "SH", target_path)
        return dicom_file

    def load_file(self, target_path: str):
        try:
            return self.add_path_to_meta(dcm.dcmread(target_path), target_path)
        except dcm.errors.InvalidDicomError as e:
            print(f"{target_path} is unreadable: {e}")
            pass
