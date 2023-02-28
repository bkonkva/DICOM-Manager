import nibabel as nib

from dicom_manager.file_loaders.file_loader import FileLoader


class NiftiLoader(FileLoader):
    """Load DICOM files."""

    def load_file(self, target_path: str):
        try:
            return nib.load(target_path)
        except nib.filebasedimages.ImageFileError as e:
            print(f"{target_path} is unreadable: {e}")
            pass
