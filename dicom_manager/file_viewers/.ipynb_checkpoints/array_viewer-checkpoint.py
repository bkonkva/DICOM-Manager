import cv2
import numpy as np

from ballir_dicom_manager.file_viewers.array_plotter import ArrayPlotter

class ArrayViewer:
    
    plotter = ArrayPlotter()
    
    def __init__(self, arr, spacing):
        self.arr = arr
        self.spacing = spacing
        return
    
    def get_transverse(self, arr, resize_dims) -> np.array:
        transverse =  np.flip(np.rot90(arr[...,int(arr.shape[2]/2)], axes = (1, 0)), 1)
        return cv2.resize(transverse, dsize = (resize_dims[1], resize_dims[0]), interpolation = cv2.INTER_CUBIC)
    
    def get_sagittal(self, arr, resize_dims) -> np.array:
        sagittal = np.flip(np.rot90(arr[int(arr.shape[0]/2),...], axes = (0, 1)), 1)
        return cv2.resize(sagittal, dsize = (resize_dims[1], resize_dims[2]), interpolation = cv2.INTER_CUBIC)
    
    def get_coronal(self, arr, resize_dims) -> np.array:
        coronal = np.rot90(arr[:, int(arr.shape[1]/2),:], axes = (0, 1))
        return cv2.resize(coronal, dsize = (resize_dims[0], resize_dims[2]), interpolation = cv2.INTER_CUBIC)
    
    def get_resize_dimensions(self):
        resize_dims = np.multiply(self.arr.shape, self.spacing)
        return [int(dim) for dim in resize_dims]
    
    def get_orthogonal_slices(self):
        resize_dims = self.get_resize_dimensions()
        transverse = self.get_transverse(self.arr, resize_dims)
        sagittal = self.get_sagittal(self.arr, resize_dims)
        coronal = self.get_coronal(self.arr, resize_dims)
        return transverse, sagittal, coronal
    
    def orthoview(self, **kwargs):
        transverse, sagittal, coronal = self.get_orthogonal_slices()
        self.plotter.plot_images([transverse, sagittal, coronal], **kwargs)
