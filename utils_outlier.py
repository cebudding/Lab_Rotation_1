import os
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage.interpolation import zoom



def load_nifti(file_path, dtype=np.float32, z_factor=None, incl_nifti=False):
    '''
        Function to extract a numpy matrix from a nifti file.

        params:
            file_path: (str) path to file location
            dtype: (numpy datatype) default np.float32
            z_factor: (float) zooming factor; if 0.5 image is reduced to half
            incl_nifti: (boolean) if true function will also return the full 
                         nifti, including header information

        returns:
            struct_arr: (np.ndarray) image matrix
            img: (Nifti1Image) nifti data type
    '''
    # Define datatype
    dt = dtype
    # Load image
    img = nib.load(file_path)
    struct_arr = img.get_data().astype(dt)
    # Downsample if required
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)
    # Return result
    if incl_nifti:
        return struct_arr, img
    else:
        return struct_arr

def load_masked_nifti(file_path, mask, sess, scan_pl, mask_pl, dtype=np.float32, z_factor=None):
    '''
        Function to extract a numpy matrix from a nifti file and apply
        a filter mask. Using a binary mask will put zeros in the image
        wherever the mask contains a zero.
        

        Uses tensorflow for speed up.
        TODO: Needs to be implemented in torch as well.

        params:
            file_path: (str) path to file location
            mask: (np.ndarray) mask matrix; same number of dimensions as the scan
            sess: tensorflow session object
            scan_pl: (tf.placeholder) placeholder; same shape as scan
            mask_pl: (tf.placeholder) placeholder; same shape as mask
            dtype: (numpy datatype) default np.float32
            z_factor: (float) zooming factor; if 0.5 image is reduced to half

        returns:
            res: (np.ndarray) image matrix with mask applied
    '''
    # Import tensorflow to use local GPU memory only; prevents overflow
    import tensorflow as tf

    # Load image
    struct_arr = load_nifti(file_path, dtype=dtype, z_factor=z_factor)
    struct_arr[np.where(np.isnan(struct_arr))] = 0

    # Downsample if required
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)

    # Multiply on GPU
    c = tf.multiply(scan_pl, mask_pl)
    res = sess.run(c, feed_dict={scan_pl: struct_arr,
                                mask_pl: mask})
    return res

def normalize_float(x, min=-1):
    '''
     Function to normalize a matrix of floats.
     Can also deal with Pytorch dictionaries where the data matrix
     key is 'image'.
    '''
    was_dict = False
    if type(x) == dict:
        was_dict = True
        mat = x['image']
    else:
        mat = x
        mat = torch.from_numpy(mat)
    if min == -1:
        norm = 2 * (mat - torch.min(mat)) / (torch.max(mat) - torch.min(mat)) - 1
    elif min == 0:
        norm = (mat - torch.min(mat)) / (torch.max(mat) - torch.min(mat))
    if was_dict:
        x['image'] = norm
        return x
    else:
        return norm

def preprocess_img(x, mean, std):
    x = (x - mean) / (std + e)
    return x

def deprocess_img(x, mean, std):
    x = x * std + mean
    return x

class Normalize(object):
    """Normalize tensor with first and second moments."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor, eps=1e-12):
        tensor['image'] = (tensor['image'] - self.mean) / (self.std + eps)
        return tensor
    
    def denormalize(self, tensor, eps=1e-12):
        tensor['image'] = tensor['image']  * (self.std + eps) + self.mean
        return tensor

class TorchSummarize():
    def __init__(self):
        import torch
        from torch.nn.modules.module import _addindent

    def torch_summarize(self, model, show_weights=True, show_parameters=True):
        """Summarizes torch model by showing trainable parameters and weights.
        Taken from wassname on Stackoverflow: https://stackoverflow.com/a/45528544
        """
        tmpstr = model.__class__.__name__ + ' (\n'
        for key, module in model._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = torch_summarize(module)
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr 
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr +=  ', parameters={}'.format(params)
            tmpstr += '\n'   

        tmpstr = tmpstr + ')'
        return tmpstr
