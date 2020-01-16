"""rttypes.py

Datatypes for general dicom processing including masking, rescaling, and fusion
"""

import sys
import os
import logging
import warnings
import math
import numpy as np
import pickle
import struct
from scipy.ndimage import interpolation
from PIL import Image

from . import dcmio, misc
from .frame import FrameOfReference
from .misc import ensure_extension
from .fileio.strutils import getFileType, isFileByExt

# initialize module logger
logger = logging.getLogger(__name__)



class Volume:
    """Defines basic storage for volumetric voxel intensities within a dicom FrameOfReference
    """
    def __init__(self):
        """Entrypoint to class, initializes members
        """
        self.data = None
        self.init_object = None
        self.frameofreference = None
        self.modality = None
        self.feature_label = None
        self.valid_exts = set()

    def __repr__(self):
        return '{!s}:\n'.format(self.__class__) + \
               '  modality: {!s}\n'.format(self.modality) + \
               '  feature_label: {!s}\n'.format(self.feature_label) + \
               '  {!s}\n'.format(self.frameofreference)

    @property
    def nslices(self):
        if len(self.frameofreference.size)>=3:
            return self.frameofreference.size[-1]
        else:
            return 1

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        self._data = v

    @property
    def array(self):
        warnings.warn('use of Volume.array property is deprecated. use Volume.data instead')
        return self.data

    @array.setter
    def array(self, v):
        warnings.warn('use of Volume.array property is deprecated. use Volume.data instead')
        self.data = v

    @property
    def frame(self):
        return self.frameofreference

    @frame.setter
    def frame(self, v):
        self.frameofreference = v

    def astype(self, type):
        self.data = self.data.astype(type)
        return self

    def _getDataDict(self):
        xstr = misc.xstr  # shorter call-name for use in function
        return {'arraydata':     self.data,
                'size':          self.frameofreference.size[::-1],
                'start':         self.frameofreference.start[::-1],
                'spacing':       self.frameofreference.spacing[::-1],
                'for_uid':       xstr(self.frameofreference.UID),
                'modality':      xstr(self.modality),
                'feature_label': xstr(self.feature_label),
                'order':         'ZYX'
                }

    @classmethod
    def load(cls, fname, frameofreference=None, recursive=False):
        if os.path.isfile(fname):
            constructorByType = {'.nii':    cls.fromNII,
                                 '.nii.gz': cls.fromNII,
                                 '.dcm':    cls.fromDicom,
                                 '.mag':    cls.fromDicom,
                                 '.mat':    cls.fromMatlab,
                                 '.pickle': cls.fromPickle,
                                 '.raw':    cls.fromBinary,
                                 '.png':    cls.fromImage,
                                 '.jpg':    cls.fromImage,
                                 '.jpeg':   cls.fromImage,
                                 None:        cls.fromBinary,
                                 '.h5':     cls.fromHDF5}
            return constructorByType[getFileType(fname)](fname)
        elif os.path.isdir(fname):
            vols = []
            # collect all full paths to dirs containing medical image files
            for dirpath, dirnames, filenames in os.walk(fname, followlinks=True):
                for f in filenames:
                    if isFileByExt(f, '.dcm') or isFileByExt(f, '.mag'):
                        try:
                            vols.append(cls.fromDir(dirpath))
                            break
                        except Exception as e:
                            logger.warning('failed to open dicom directory: "{}"\n{}'.format(dirpath, e))
                if not recursive: break
            if len(vols) > 1: return vols
            elif len(vols)==1: return vols[0]
            else: raise RuntimeError('Failed to load')

    @classmethod
    def fromArray(cls, array, frameofreference=None):
        """Constructor: from a numpy array and FrameOfReference object

        Args:
            array             -- numpy array
            frameofreference  -- FrameOfReference object
        """
        # ensure array matches size in frameofreference
        self = cls()
        if array.ndim == 2:
            array = np.atleast_3d(array)
        if frameofreference is not None:
            self.data = array.reshape(frameofreference.size[::-1])
            self.frameofreference = frameofreference
        else:
            self.data = array
            self.frameofreference = FrameOfReference((0,0,0), (1,1,1), (*array.shape[::-1], 1))

        return self

    @classmethod
    def fromImage(cls, fname, frameofreference=None, normalize=True):
        with open(fname, 'rb') as fd:
            im = Image.open(fd, 'r')
            if im.mode in ['1', 'L', 'P']:
                dim = 1
                if im.mode=='P':
                    im = im.convert('L')
            elif im.mode in ['RGB', 'YCbCr']:
                dim = 3
            elif im.mode in ['RGBA', 'CMYK']:
                dim = 4
            else:
                raise RuntimeError("Couldn't determine dimensionality of image with mode=\"{!s}\"".format(im.mode))
            maxint = 255 # assume all 8-bit per channel
            arr = np.asarray(im).transpose([2,0,1])
            if normalize:
                # normalize to [0,1]
                arr = arr.astype('float32')
                arr /= maxint

        #  def plotChannels(arr):
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(9,3))
            titles = ['red', 'green', 'blue']
            for i in range(arr.shape[0]):
                ax = fig.add_subplot(1,3,i+1)
                ax.imshow(arr[i,:,:], cmap="Greys")
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                ax.set_title(titles[i])
            plt.show()

            if frameofreference is None:
                frame = FrameOfReference((0,0,0), (1,1,1), arr.shape)
            return cls.fromArray(arr, frame)

    #  def toImage(self, fname, mode='L', resize=None, cmap='Set3'):
    #      array = self.data
    #      array = np.squeeze(array)
    #      if array.ndim != 2:
    #          raise RuntimeError('Saving image with ndim={} is not supported'.format(array.ndim))

    #      if mode in ['RGB', 'RGBA']:
    #          # convert integer class ids to rgb colors according to cmap
    #          rng = abs(np.max(array)-np.min(array))
    #          if rng == 0: rng = 1
    #          normarray = (array - np.min(array)) / rng
    #          im = Image.fromarray(np.uint8(plt.cm.get_cmap(cmap)(normarray)*255))
    #      elif mode in ['P']:
    #          # separates gray values so they can be distinguished
    #          array*=math.floor((255 / len(np.unique(array))))
    #          im = Image.fromarray(array.astype('uint8'))
    #      elif mode in ['1', 'L', 'P']:
    #          im = Image.fromarray(array.astype('uint8'))
    #      else: raise RuntimeError

    #      # restore image to original dims
    #      if isinstance(resize, numbers.Number) and resize>0 and not resize==1:
    #          im = im.resize( [int(resize*s) for s in im.size], resample=Image.NEAREST)

    #      fname = ensure_extension(fname, '.png')
    #      im.save(fname)
    #      logger.debug('file saved to {}'.format(fname))


    @classmethod
    def fromDir(cls, path, recursive=False):
        """constructor: takes path to directory containing dicom files and builds a sorted array

        Args:
            recursive -- find dicom files in all subdirectories?
        """
        # get the datasets from files
        dataset_list = dcmio.read_dicom_dir(path, recursive=recursive)

        # pass dataset list to constructor
        self = cls.fromDatasetList(dataset_list)

        return self

    @classmethod
    def fromBinary(cls, path, frameofreference):
        """constructor: takes path to binary file (neylon .raw)
        data is organized as binary float array in row-major order

        Args:
            path (str): path to .raw file in binary format
            frameofreference (FOR): most importantly defines mapping from 1d to 3d array
        """
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in ['.raw', '.bin', None, '']:
            raise Exception('data is not formatted properly. must be one of [.raw, .bin]')

        if not isinstance(frameofreference, FrameOfReference):
            if not isinstance(frameofreference, tuple):
                raise TypeError('frameofreference must be a valid FrameOfReference or tuple of dimensions')
            frameofreference = FrameOfReference(start=(0,0,0), spacing=(1,1,1), size=frameofreference)

        with open(path, mode='rb') as f:
            flat = f.read()
        _shape = frameofreference.size[::-1]
        _expected_n = np.product(_shape)
        thetype = None
        for type in ['f', 'd']:
            _n = int(os.path.getsize(path)/struct.calcsize(type))
            if _n != _expected_n:
                logger.debug('filesize ({:f}) doesn\'t match expected ({:f}) size'.format(
                    os.path.getsize((path)), struct.calcsize(type)*_expected_n
                ))
            else:
                thetype = type
                break
        if thetype is None:
            raise RuntimeError("filesize ({:f}) doesn't match expected size ({:f})".format(
                    os.path.getsize((path)), struct.calcsize('f')*_expected_n
            ))
        s = struct.unpack(thetype*_n, flat)
        vol = np.array(s).reshape(_shape)
        #  vol[vol>1e10] = 0
        #  vol[vol<-1e10] = 0
        return cls.fromArray(vol, frameofreference)

    @classmethod
    def fromDicom(cls, fname):
        return cls.fromDatasetList([dcmio.read_dicom(fname)])

    def toDicom(self, dname, fprefix=''):
        import pydicom  # pydicom
        SeriesInstanceUID   = pydicom.uid.generate_uid()
        StudyInstanceUID    = pydicom.uid.generate_uid()
        FrameOfReferenceUID = pydicom.uid.generate_uid()
        min_val = np.min(self.data)
        for i in range(self.frameofreference.size[2]):
            ds = dcmio.make_dicom_boilerplate(SeriesInstanceUID, StudyInstanceUID, FrameOfReferenceUID)
            ds.SliceThickness = self.frameofreference.spacing[2]
            ds.PixelSpacing = list(self.frameofreference.spacing[:2])
            ds.SliceLocation = self.frameofreference.start[2] + i*self.frameofreference.spacing[2]
            ds.ImagePositionPatient = [*self.frameofreference.start[:2], ds.SliceLocation]
            ds.Columns = self.frameofreference.size[0]
            ds.Rows = self.frameofreference.size[1]
            ds.AcquisitionNumber = i+1
            ds.Modality = self.modality if self.modality is not None else ''
            ds.DerivationDescription = self.feature_label if self.feature_label is not None else ''
            ds.PixelData = ((self.data[i, :, :]-min_val).flatten().astype(np.uint16)).tostring()
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = math.floor(min_val)
            ds.PixelRepresentation = 0 # unsigned integers
            os.makedirs(dname, exist_ok=True)
            ds.save_as(os.path.join(dname, '{}{:04d}.dcm'.format(fprefix, i)))

    @classmethod
    def fromDatasetList(cls, dataset_list):
        """constructor: takes a list of dicom slice datasets and builds a Volume array
        Args:
            slices
        """
        import pydicom  # pydicom
        self = cls()
        if (dataset_list is None):
            raise ValueError('no valid dataset_list provided')

        # check that all elements are valid slices, if not remove and continue
        nRemoved = 0
        for i, slice in enumerate(list(dataset_list)):
            if (not isinstance(slice, pydicom.dataset.Dataset)):
                logger.debug('invalid type ({t:s}) at idx {i:d}. removing.'.format(
                    t=str(type(slice)),
                    i=i ) )
                dataset_list.remove(slice)
                nRemoved += 1
            elif (slice.get('SOPClassUID', None)!='1.2.840.10008.5.1.4.1.1.2'):
                logger.debug('invalid .dcm image at idx {:d}. removing.'.format(i))
                dataset_list.remove(slice)
                nRemoved += 1
        if (nRemoved > 0):
            logger.info('# slices removed with invalid types: {:d}'.format(nRemoved))

        # sort datasets by increasing slicePosition (inferior -> superior)
        dataset_list.sort(key=lambda dataset: dataset.ImagePositionPatient[2], reverse=False)

        # build object properties
        start = dataset_list[0].ImagePositionPatient
        spacing = (*dataset_list[0].PixelSpacing, dataset_list[1].ImagePositionPatient[2]-dataset_list[0].ImagePositionPatient[2])
        try:
            # some modalities don't provide NumberOfSlices attribute
            size = (dataset_list[0].Columns, dataset_list[0].Rows, dataset_list[0].NumberOfSlices)
        except:
            # use length of list instead
            size = (dataset_list[0].Columns, dataset_list[0].Rows, len(dataset_list))

        UID = dataset_list[0].FrameOfReferenceUID
        self.frameofreference = FrameOfReference(start, spacing, size, UID)

        # standardize modality labels
        mod = dataset_list[0].Modality
        if (mod == 'PT'):
            mod = 'PET'
        self.modality = mod

        # construct 3dArray
        array_list = []
        for dataset in dataset_list:
            array = dataset.pixel_array.astype(np.int16)
            factor = dataset.RescaleSlope
            offset = dataset.RescaleIntercept
            array = array * factor + offset
            array = array.reshape((1, array.shape[0], array.shape[1]))
            array_list.append(array)

        # stack arrays
        self.data = np.concatenate(array_list, axis=0)
        #  self.data = self.data.astype(int)
        return self

    @classmethod
    def fromPickle(cls, path):
        """initialize Volume from unchanging format so features can be stored and recalled long term
        """
        warnings.warn('{!s}.fromPickle() will be deprecated soon in favor of other serialization methods.'.format(cls.__name__), DeprecationWarning)
        path = ensure_extension(path, '.pickle')
        if (not os.path.exists(path)):
            logger.info('file at path: {:s} doesn\'t exists'.format(path))
        with open(path, 'rb') as p:
            # added to fix broken module refs in old pickles
            sys.modules['utils'] = sys.modules[__name__]
            sys.modules['utils.rttypes'] = sys.modules[__name__]
            basevolumeserial = pickle.load(p)
            del sys.modules['utils.rttypes']
            del sys.modules['utils']

        # import data to this object
        try:
            self = cls()
            self.data = basevolumeserial.dataarray
            self.frameofreference = FrameOfReference(basevolumeserial.startposition,
                                                     basevolumeserial.spacing,
                                                     basevolumeserial.size)
            self.modality = basevolumeserial.modality
            self.feature_label = basevolumeserial.feature_label
        except:
            raise SerialOutdatedError()
        return self

    def toPickle(self, path):
        """store critical data to unchanging format that can be pickled long term
        """
        warnings.warn('{!s}.toPickle() will be deprecated soon in favor of other serialization methods.'.format(self.__class__), DeprecationWarning)
        basevolumeserial = VolumeSerial()
        basevolumeserial.startposition = self.frameofreference.start
        basevolumeserial.spacing = self.frameofreference.spacing
        basevolumeserial.size = self.frameofreference.size
        basevolumeserial.dataarray = self.data
        basevolumeserial.modality = self.modality
        basevolumeserial.feature_label = self.feature_label

        path = ensure_extension(path, '.pickle')
        _dirname = os.path.dirname(path)
        if (_dirname and _dirname is not ''):
            os.makedirs(_dirname, exist_ok=True)
        with open(path, 'wb') as p:
            pickle.dump(basevolumeserial, p)

    @classmethod
    def fromMatlab(cls, path):
        """restore Volume from .mat file that was created using Volume.toMatlab() """
        import scipy.io  # savemat -> save to .mat
        path = ensure_extension(path, '.mat')
        extract_str = misc.numpy_safe_string_from_array
        data = scipy.io.loadmat(path, appendmat=True)
        #  for key, obj in data.items():
        #      print('{!s}({!s}: {!s}'.format(key, type(obj), obj))
        converted_data = {
            'arraydata': data['arraydata'],
            'size': tuple(data['size'][0,:])[::-1],
            'start': tuple(data['start'][0,:])[::-1],
            'spacing': tuple(data['spacing'][0,:])[::-1],
            'for_uid': extract_str(data['for_uid']),
            'modality': extract_str(data['modality']),
            'feature_label': extract_str(data['feature_label']),
            'order': extract_str(data['order'])
        }

        # construct new volume
        self = cls()
        self.data = converted_data['arraydata']
        self.frameofreference = FrameOfReference(converted_data['start'],
                                                 converted_data['spacing'],
                                                 converted_data['size'],
                                                 converted_data['for_uid'])
        self.modality = converted_data['modality']
        self.feature_label = converted_data['feature_label']

        return self


    def toMatlab(self, path, compress=False):
        """store critical data to .mat file compatible with matlab loading
        This is essentially .toPickle() with compat. for matlab reading

        Optional Args:
            compress (bool): compress dataarray at the cost of write speed
        """
        import scipy.io  # savemat -> save to .mat
        # first represent as dictionary for savemat()
        data = self._getDataDict()
        data['order'] = 'ZYX'
        path = ensure_extension(path, '.mat')

        # write to .mat
        scipy.io.savemat(path, data, appendmat=False, format='5', long_field_names=False,
                         do_compression=compress, oned_as='row')

    def toHDF5(self, path, compress=False):
        """store object to hdf5 file with image data stored as dataset and metadata as attributes"""
        import h5py
        data = self._getDataDict()
        arraydata = data.pop('arraydata')
        path = ensure_extension(path, '.h5')
        with h5py.File(path, 'w') as f:
            for k, v in data.items():
                f.attrs.__setitem__(k, v)
            f.create_dataset('arraydata', data=arraydata)
            f.attrs['fileversion'] = '1.0'

    def _fromDoseH5(self, path):
        """load from dosecalc defined h5 file"""
        import h5py
        with h5py.File(path, 'r') as f:
            ad = f['dose']
            self.data = np.empty(ad.shape)
            ad.read_direct(self.data)
            self.data = np.array(self.data)
            self.frameofreference = FrameOfReference(
                tuple(ad.attrs['dicom_start_cm'])[::-1],
                tuple(ad.attrs['voxel_size_cm'])[::-1],
                tuple(ad.shape)[::-1]
            )
        return self

    def _fromH5(self, path):
        """load from pymedimage defined h5 file"""
        import h5py
        extract_str = misc.numpy_safe_string_from_array
        with h5py.File(path, 'r') as f:
            ad = f['arraydata']
            self.data = np.empty(ad.shape)
            ad.read_direct(self.data)
            self.data = np.array(self.data)
            self.frameofreference = FrameOfReference(
                tuple(f.attrs['start'])[::-1],
                tuple(f.attrs['spacing'])[::-1],
                tuple(f.attrs['size'])[::-1],
                extract_str(f.attrs['for_uid'])
            )
            self.modality = f.attrs['modality']
            self.feature_label = f.attrs['feature_label']

    @classmethod
    def fromHDF5(cls, path):
        """restore objects from hdf5 file with image data stored as dataset and metadata as attributes"""
        # construct new volume
        self = cls()
        path = ensure_extension(path, '.h5')
        loaded = False
        except_msgs = []
        for meth in [self._fromDoseH5, self._fromH5]:
            try:
                meth(path)
                loaded = True
                break
            except Exception as e: except_msgs.append(str(e))
        if not loaded:
            raise RuntimeError('failed to load "{!s}"\n{!s}'.format(path, '\n'.join(except_msgs)))
        return self

    def toImage(self, fname):
        if self.nslices > 1:
            ext = os.path.splitext(fname)[1]
            for i in range(self.nslices):
                fname = fname.replace(ext, '_{:0.4d}.{}'.format(i, ext))
                arr = self.data[i,:,:].reshape(self.frameofreference.size[0:2:-1])
                arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr)) * 255
                im = Image.fromarray(arr).convert('L')
                im.save(fname)
        else:
            arr = self.data[0,:,:].reshape(self.frameofreference.size[-2:-4:-1])
            arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr)) * 255
            im = Image.fromarray(arr).convert('L')
            im.save(fname)

    def toNII(self, fname, affine=None):
        import nibabel as nib
        if affine is None:
            logger.warning('No information about global coordinate system provided')
            affine = np.diag([1,1,1,1])

        fname = ensure_extension(fname, '.nii')
        img = nib.Nifti1Image(self.data, affine)
        img.to_filename(fname)
        return fname

    @classmethod
    def fromNII(cls, fname):
        import nibabel as nib
        img = nib.load(fname)
        h = img.header
        # TODO: add support for non-axially oriented slices (with affine view transformation)
        data = np.transpose(img.get_data(), (2,1,0))
        frame = FrameOfReference((0,0,0), h.get_zooms(), h.get_data_shape())
        self = cls.fromArray(data, frame)
        self.init_object = img
        return self


    # PUBLIC METHODS
    def conformTo(self, frameofreference):
        """Resamples the current Volume to the supplied FrameOfReference

        Args:
            frameofreference   -- FrameOfReference object to resample the Basevolume to

        Returns:
            Volume
        """
        # conform volume to alternate FrameOfReference
        if (frameofreference is None):
            logger.exception('no FrameOfReference provided')
            raise ValueError
        elif (FrameOfReference.__name__ not in str(type(frameofreference))):  # This is an ugly way of type-checking but cant get isinstance to see both as the same
            logger.exception(('supplied frameofreference of type: "{:s}" must be of the type: "FrameOfReference"'.format(
                str(type(frameofreference)))))
            raise TypeError

        if self.frameofreference == frameofreference:
            return self

        # first match self resolution to requested resolution
        zoomarray, zoomFOR = self._resample(frameofreference.spacing)

        # crop to active volume of requested FrameOfReference in frameofreference
        xstart_idx, ystart_idx, zstart_idx = zoomFOR.getIndices(frameofreference.start)
        # xend_idx, yend_idx, zend_idx = zoomFOR.getIndices(frameofreference.end())
        # force new size to match requested FOR size
        xend_idx, yend_idx, zend_idx = tuple((np.array((xstart_idx, ystart_idx, zstart_idx)) + np.array(frameofreference.size)).tolist())
        try:
            cropped = zoomarray[zstart_idx:zend_idx, ystart_idx:yend_idx, xstart_idx:xend_idx]
            zoomFOR.start = frameofreference.start
            zoomFOR.size = cropped.shape[::-1]
        except:
            logger.exception('request to conform to frame outside of volume\'s frame of reference failed')
            raise Exception()

        # reconstruct volume from resampled array
        resampled_volume = Volume.fromArray(cropped, zoomFOR)
        resampled_volume.modality = self.modality
        resampled_volume.feature_label = self.feature_label
        return resampled_volume

    def _resample(self, new_voxelsize=None, mode='nearest', order=3, zoom_factors=None):
        if zoom_factors is None and new_voxelsize is None: raise RuntimeError('must set either factor or new_voxelsize')
        if zoom_factors is not None and not isinstance(zoom_factors, list) and not isinstance(zoom_factors, tuple):
                zoom_factors = tuple([zoom_factors]*self.data.ndim)

        if new_voxelsize is not None and zoom_factors is None:
            if new_voxelsize == self.frameofreference.spacing:
                # no need to resample
                return (self.data, self.frameofreference)
            # voxelsize spec is in order (X,Y,Z) but array is kept in order (Z, Y, X)
            zoom_factors = np.true_divide(self.frameofreference.spacing, new_voxelsize)

        logger.debug('resizing volume with factors (xyz): {!s}'.format(zoom_factors))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zoomarray = interpolation.zoom(self.data, zoom_factors[::-1], order=order, mode=mode)
        zoomFOR = FrameOfReference(self.frameofreference.start, new_voxelsize, zoomarray.shape[::-1])
        return (zoomarray, zoomFOR)

    def resample(self, *args, **kwargs):
        """resamples volume to new voxelsize

        Args:
            new_voxelsize: 3 tuple of voxel size in mm in the order (X, Y, Z)

        """
        zoomarray, zoomFOR = self._resample(*args, **kwargs)
        new_vol = Volume.fromArray(zoomarray, zoomFOR)
        new_vol.modality = self.modality
        new_vol.feature_label = self.feature_label
        return new_vol

    def getSlice(self, idx, axis=0,  flatten=False):
        """Extracts 2dArray of idx along the axis.
        Args:
            idx       -- idx identifying the slice along axis

        Optional Args:
            axis      -- specifies axis along which to extract
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            flatten   -- return a 1darray in depth-stacked row-major order
        """
        cols, rows, depth = self.frameofreference.size

        # perform index bounding
        if (axis==0):
            if (idx < 0 or idx >= depth):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(depth-1))
                raise IndexError
            thisslice = self.data[idx, :, :]
        elif (axis==1):
            if (idx < 0 or idx >= rows):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(rows-1))
                raise IndexError
            thisslice = self.data[:, idx, :]
        elif (axis==2):
            if (idx < 0 or idx >= cols):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(cols-1))
                raise IndexError
            thisslice = self.data[:, :, idx]
        else:
            logger.exception('invalid axis supplied. must be between 0 -> 2')
            raise ValueError

        # RESHAPE
        if (flatten):
            thisslice = thisslice.flatten(order='C').reshape((-1, 1))

        return thisslice

    def vectorize(self):
        """flatten self.data in stacked-depth row-major order
        """
        return self.data.flatten(order='C').reshape((-1, 1))

    def get_val(self, z, y, x):
        """take xyz indices and return the value in array at that location
        """
        frameofreference = self.frameofreference
        # get volume size
        (cols, rows, depth) = frameofreference.size

        # perform index bounding
        if (x < 0 or x >= cols):
            logger.exception('x index ({:d}) out of bounds. must be between 0 -> {:d}'.format(x, cols-1))
            raise IndexError
        if (y < 0 or y >= rows):
            logger.exception('y index ({:d}) out of bounds. must be between 0 -> {:d}'.format(y, rows-1))
            raise IndexError
        if (z < 0 or z >= depth):
            logger.exception('z index ({:d}) out of bounds. must be between 0 -> {:d}'.format(z, depth-1))
            raise IndexError

        return self.data[z, y, x]

    def set_val(self, z, y, x, value):
        """take xyz indices and value and reassing the value in array at that location
        """
        frameofreference = self.frameofreference
        # get volume size
        (cols, rows, depth) = frameofreference.size

        # perform index bounding
        if (x < 0 or x >= cols):
            logger.exception('x index ({:d}) out of bounds. must be between 0 -> {:d}'.format(x, cols-1))
            raise IndexError
        if (y < 0 or y >= rows):
            logger.exception('y index ({:d}) out of bounds. must be between 0 -> {:d}'.format(y, rows-1))
            raise IndexError
        if (z < 0 or z >= depth):
            logger.exception('z index ({:d}) out of bounds. must be between 0 -> {:d}'.format(z, depth-1))
            raise IndexError

        # reassign value
        self.data[z, y, x] = value


class SerialOutdatedError(Exception):
    def __init__(self):
        super().__init__('a missing value was requested from a VolumeSerial object')


class VolumeSerial:
    """Defines common object that can store feature data for long term I/O
    """
    def __init__(self):
        self.dataarray     = None  # numpy ndarray
        self.startposition = None  # (x, y, z)<float>
        self.spacing       = None  # (x, y, z)<float>
        self.size          = None  # (x, y, z)<integer>
        self.modality      = None  # string
        self.feature_label = None  # string
