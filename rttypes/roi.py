import os
import logging
import warnings
import pickle
import math

import numpy as np
from PIL import Image, ImageDraw

from . import dcmio, misc
from .volume import Volume
from .misc import ensure_extension
from .frame import FrameOfReference

logger = logging.getLogger(__name__)

class ROI:
    """Defines a labeled RTStruct ROI for use in masking and visualization of Radiotherapy contours
    """
    def __init__(self, roicontour=None, structuresetroi=None):
        self.roinumber = None
        self.refforuid = None
        self.frameofreference = None
        self.roiname = None
        self.coordslices = []
        # Cached variables
        self.__cache_densemask = None   # storage for Volume when consecutive calls to
                                        # makeDenseMask are made
                                        # with the same frameofreference object

        if roicontour and structuresetroi:
            self._fromDicomDataset(roicontour, structuresetroi)

    def __repr__(self):
        return '{!s}:\n'.format(self.__class__) + \
               '  roiname: {!s}\n'.format(self.roiname) + \
               '  {!s}\n'.format(self.frameofreference)

    @staticmethod
    def _loadRtstructDicom(rtstruct_path):
        """load rtstruct dicom data from a direct path or containing directory"""
        if (not os.path.exists(rtstruct_path)):
            logger.debug('invalid path provided: "{:s}"'.format(rtstruct_path))
            raise FileNotFoundError

        # check if path is file or dir
        if (os.path.isdir(rtstruct_path)):
            # search recursively for a valid rtstruct file
            ds_list = dcmio.read_dicom_dir(rtstruct_path, recursive=True)
            if (ds_list is None or len(ds_list) == 0):
                logger.debug('no rtstruct datasets found at "{:s}"'.format(rtstruct_path))
                raise Exception
            ds = ds_list[0]
        elif (os.path.isfile(rtstruct_path)):
            ds = dcmio.read_dicom(rtstruct_path)
        return ds

    def _fromDicomDataset(self, roicontour, structuresetroi):
        """takes FrameOfReference object and roicontour/structuresetroi dicom dataset objects and stores
        sorted contour data

        Args:
            roicontour         -- dicom dataset containing contour point coords for all slices
            structuresetroi    -- dicom dataset containing additional information about contour
        """
        self.roinumber = int(structuresetroi.ROINumber)
        self.refforuid = str(structuresetroi.ReferencedFrameOfReferenceUID)
        self.roiname = str(structuresetroi.ROIName)

        # Populate list of coordslices, each containing a list of ordered coordinate points
        contoursequence = roicontour.ContourSequence
        if (len(contoursequence) <= 0):
            logger.debug('no coordinates found in roi: {:s}'.format(self.roiname))
        else:
            logger.debug('loading roi: {:s} with {:d} slices'.format(self.roiname, len(roicontour.ContourSequence)))
            for coordslice in roicontour.ContourSequence:
                points_list = []
                for x, y, z in misc.grouper(3, coordslice.ContourData):
                    points_list.append( (x, y, z) )
                self.coordslices.append(points_list)

            # sort by slice position in ascending order (inferior -> superior)
            self.coordslices.sort(key=lambda coordslice: coordslice[0][2], reverse=False)

            #  # create frameofreference based on the extents of the roi and apparent spacing
            #  self.frameofreference = self.getROIExtents()

    @classmethod
    def roiFromFile(cls, rtstruct_path, name, casesensitive=True):
        ds = cls._loadRtstructDicom(rtstruct_path)
        if (ds is not None):
            # get structuresetROI sequence
            StructureSetROI_list = ds.StructureSetROISequence
            nContours = len(StructureSetROI_list)
            if (nContours <= 0):
                logger.exception('no contours were found')

            for StructureSetROI in StructureSetROI_list:
                if (casesensitive and StructureSetROI.ROIName == name) or (not casesensitive and str(StructureSetROI.ROIName).lower() == name.lower()):
                    ROIContour = None
                    for ROIContour in ds.ROIContourSequence:
                        if ROIContour.ReferencedROINumber == StructureSetROI.ROINumber:
                            return cls(ROIContour, StructureSetROI)
            return None

        else:
            logger.exception('no dataset was found')

    @classmethod
    def collectionFromFile(cls, rtstruct_path, keep_empty=False):
        """loads an rtstruct specified by path and returns a dict of ROI objects

        Args:
            rtstruct_path    -- path to rtstruct.dcm file

        Returns:
            dict<key='contour name', val=ROI>
        """
        ds = cls._loadRtstructDicom(rtstruct_path)

        # parse rtstruct file and instantiate maskvolume for each contour located
        # add each maskvolume to dict with key set to contour name and number?
        if (ds is not None):
            # get structuresetROI sequence
            StructureSetROI_list = ds.StructureSetROISequence
            nContours = len(StructureSetROI_list)
            if (nContours <= 0):
                logger.exception('no contours were found')

            # Add structuresetROI to dict
            StructureSetROI_dict = {StructureSetROI.ROINumber: StructureSetROI
                                    for StructureSetROI
                                    in StructureSetROI_list }

            # get dict containing a contour dataset for each StructureSetROI with a paired key=ROINumber
            ROIContour_dict = {ROIContour.ReferencedROINumber: ROIContour
                               for ROIContour
                               in ds.ROIContourSequence }

            # construct a dict of ROI objects where contour name is key
            roi_dict = {}
            for ROINumber, structuresetroi in StructureSetROI_dict.items():
                roi_dict[structuresetroi.ROIName] = (cls(roicontour=ROIContour_dict[ROINumber],
                                                         structuresetroi=structuresetroi))
            # prune empty ROIs from dict
            if not keep_empty:
                for roiname, roi in dict(roi_dict).items():
                    if (roi.coordslices is None or len(roi.coordslices) <= 0):
                        logger.debug('pruning empty ROI: {:s} from loaded ROIs'.format(roiname))
                        del roi_dict[roiname]

            logger.debug('loaded {:d} ROIs succesfully'.format(len(roi_dict)))
            return roi_dict
        else:
            logger.exception('no dataset was found')

    @staticmethod
    def getROINames(rtstruct_path):
        ds = ROI._loadRtstructDicom(rtstruct_path)

        if (ds is not None):
            # get structuresetROI sequence
            StructureSetROI_list = ds.StructureSetROISequence
            nContours = len(StructureSetROI_list)
            if (nContours <= 0):
                logger.exception('no contours were found')

            roi_names = []
            for structuresetroi in StructureSetROI_list:
                roi_names.append(structuresetroi.ROIName)

            return roi_names
        else:
            logger.exception('no dataset was found')

    def makeDenseMaskSlice(self, position, frameofreference=None):
        """Takes a FrameOfReference and constructs a dense binary mask for the ROI (1 inside ROI, 0 outside)
        as a numpy 2dArray

        Args:
            position           -- position of the desired slice (mm) within the frameofreference along z-axis
            frameofreference   -- FrameOfReference that defines the position of ROI and size of dense volume

        Returns:
            numpy 2dArray
        """
        # get FrameOfReference params
        if (frameofreference is None):
            if (self.frameofreference is not None):
                frameofreference = self.frameofreference
            else:
                logger.exception('no frame of reference provided')
                raise Exception
        xstart, ystart, zstart = frameofreference.start
        xspace, yspace, zspace = frameofreference.spacing
        cols, rows, depth = frameofreference.size

        # get nearest coordslice
        minerror = 5000
        coordslice = None
        ### REVISIT THE CORRECT SETTING OF TOLERANCE TODO
        tolerance = frameofreference.spacing[2]*0.95 - 1e-9  # if upsampling too much then throw error
        for slice in self.coordslices:
            # for each list of coordinate tuples - check the slice for distance from position
            error = abs(position - slice[0][2])
            if error <= minerror:
                #  if minerror != 5000:
                #     logger.info('position:{:0.3f} | slicepos:{:0.3f}'.format(position, slice[0][2]))
                #     logger.info('improved with error {:f}'.format(error))
                minerror = error
                coordslice = slice
                # logger.debug('updating slice')
            else:
                # we've already passed the nearest slice, break
                break

        # check if our result is actually valid or we just hit the end of the array
        if coordslice and minerror >= tolerance:
            logger.debug('No slice found within {:f} mm of position {:f}'.format(tolerance, position))
            # print(minerror, tolerance)
            # print(position)
            # print(zstart, zspace*depth)
            # for slice in self.coordslices:
            #     if abs(slice[0][2]-position) < 100:
            #         print(slice[0][2])
            return np.zeros((rows, cols))
            # raise Exception('Attempt to upsample ROI to densearray beyond 5x')
        logger.debug('slice found at {:f} for position query at {:f}'.format(coordslice[0][2], position))

        # get coordinate values
        index_coords = []
        for x, y, z in coordslice:
            # shift x and y and scale appropriately
            x_idx = int(round((x-xstart)/xspace))
            y_idx = int(round((y-ystart)/yspace))
            index_coords.append( (x_idx, y_idx) )

        # use PIL to draw the polygon as a dense image (PIL uses shape: (width, height))
        im = Image.new('1', (cols, rows), color=0)
        imdraw = ImageDraw.Draw(im)
        imdraw.polygon(index_coords, fill=1, outline=None)
        del imdraw

        # convert from PIL image to np.ndarray and threshold to binary
        return np.array(im.getdata()).reshape((rows, cols))

    def makeDenseMask(self, frameofreference=None):
        """Takes a FrameOfReference and constructs a dense binary mask for the ROI (1 inside ROI, 0 outside)
        as a Volume

        Args:
            frameofreference   -- FrameOfReference that defines the position of ROI and size of dense volume

        Returns:
            Volume
        """
        # get FrameOfReference params
        if (frameofreference is None):
            if (self.frameofreference is not None):
                frameofreference = self.frameofreference
            else:
                logger.exception('no frame of reference provided')
                raise Exception

        # check cache for similarity between previously and currently supplied frameofreference objects
        if (self.__cache_densemask is not None
                and frameofreference == self.__cache_densemask.frameofreference):
            # cached mask frameofreference is similar to current, return cached densemask volume
            # logger.debug('using cached dense mask volume')
            return self.__cache_densemask
        else:
            xstart, ystart, zstart = frameofreference.start
            xspace, yspace, zspace = frameofreference.spacing
            cols, rows, depth = frameofreference.size

            # generate binary mask for each slice in frameofreference
            maskslicearray_list = []
            # logger.debug('making dense mask volume from z coordinates: {:f} to {:f}'.format(
            #              zstart, (zspace * (depth+1) + zstart)))
            for i in range(depth):
                position = zstart + i * zspace
                # get a slice at every position within the current frameofreference
                densemaskslice = self.makeDenseMaskSlice(position, frameofreference)
                maskslicearray_list.append(densemaskslice.reshape((1, *densemaskslice.shape)))

            # construct Volume from dense slice arrays
            densemask = Volume.fromArray(np.concatenate(maskslicearray_list, axis=0), frameofreference)
            self.__cache_densemask = densemask
            return densemask

    def getROIExtents(self, spacing=None):
        """Creates a tightly bound frame of reference around the ROI which allows visualization in a cropped
        frame
        """
        # guess at spacing and assign arbitrarily where necessary
        # get list of points first
        point_list = []
        for slice in self.coordslices:
            for point3d in slice:
                point_list.append(point3d)

        # set actually z spacing estimated from separation of coordslice point lists
        min_z_space = 9999
        prev_z = point_list[0][2]
        for point3d in point_list[1:]:
            z = point3d[2]
            this_z_space = abs(z-prev_z)
            if (this_z_space > 0 and this_z_space < min_z_space):
                min_z_space = this_z_space
            prev_z = z

        if (min_z_space <= 0 or min_z_space > 10):
            # unreasonable result found, arbitrarily set
            new_z_space = 1
            logger.debug('unreasonable z_spacing found: {:0.3f}, setting to {:0.3f}'.format(
                min_z_space, new_z_space))
            min_z_space = new_z_space
        else:
            logger.debug('estimated z_spacing: {:0.3f}'.format(min_z_space))

        # arbitrarily set spacing
        if spacing is None:
            spacing = (1, 1, min_z_space)
            warnings.warn('Inferred spacing is deprecated in favor of manual specification. Please change code immediately to ensure correctness', DeprecationWarning)
        else:
            if min_z_space != spacing[2]:
                warnings.warn('Inferred slice thickness from rtstruct ({0:g}) not equal to user specified ({:g}). Using user specification ({1:g})'.format(min_z_space, spacing[2]))

        # get start and end of roi volume extents
        global_limits = {'xmax': -5000,
                         'ymax': -5000,
                         'zmax': -5000,
                         'xmin': 5000,
                         'ymin': 5000,
                         'zmin': 5000 }
        for slice in self.coordslices:
            # convert coords list to ndarray
            coords = np.array(slice)
            (xmin, ymin, zmin) = tuple(coords.min(axis=0, keepdims=False))
            (xmax, ymax, zmax) = tuple(coords.max(axis=0, keepdims=False))

            # update limits
            if xmin < global_limits['xmin']:
                global_limits['xmin'] = xmin
            if ymin < global_limits['ymin']:
                global_limits['ymin'] = ymin
            if zmin < global_limits['zmin']:
                global_limits['zmin'] = zmin
            if xmax > global_limits['xmax']:
                global_limits['xmax'] = xmax
            if ymax > global_limits['ymax']:
                global_limits['ymax'] = ymax
            if zmax > global_limits['zmax']:
                global_limits['zmax'] = zmax

        # build FrameOfReference
        start = (global_limits['xmin'],
                 global_limits['ymin'],
                 global_limits['zmin'] )
        size = (int(math.ceil((global_limits['xmax'] - global_limits['xmin']) / spacing[0])),
                int(math.ceil((global_limits['ymax'] - global_limits['ymin']) / spacing[1])),
                int(math.ceil((global_limits['zmax'] - global_limits['zmin']) / spacing[2])) )

        logger.debug('ROIExtents:\n'
                     '    start:   {:s}\n'
                     '    spacing: {:s}\n'
                     '    size:    {:s}'.format(str(start), str(spacing), str(size)))
        frameofreference = FrameOfReference(start, spacing, size, UID=None)
        return frameofreference

    def toPickle(self, path):
        """convenience function for storing ROI to pickle file"""
        warnings.warn('ROI.toPickle() will be deprecated soon in favor of other serialization methods.', DeprecationWarning)
        _dirname = os.path.dirname(path)
        if (_dirname and _dirname is not ''):
            os.makedirs(_dirname, exist_ok=True)
        with open(path, 'wb') as p:
            pickle.dump(self, p)

    @staticmethod
    def fromPickle(path):
        """convenience function for restoring ROI from pickle file"""
        warnings.warn('ROI.fromPickle() will be deprecated soon in favor of other serialization methods.', DeprecationWarning)
        with open(path, 'rb') as p:
            return pickle.load(p)

    def toHDF5(self, path):
        """serialize object to file in h5 format"""
        import h5py
        path = ensure_extension(path, '.h5')
        with h5py.File(path, 'w') as f:
            # store attributes
            f.attrs['roinumber'] = self.roinumber
            f.attrs['roiname'] = self.roiname
            f.attrs['refforuid'] = self.refforuid
            f.attrs['FrameOfReference.start'] = self.frameofreference.start
            f.attrs['FrameOfReference.spacing'] = self.frameofreference.spacing
            f.attrs['FrameOfReference.size'] = self.frameofreference.size
            f.attrs['fileversion'] = '1.0'

            # store datasets
            g = f.create_group('coordslices')
            g.attrs['Nslices'] = len(self.coordslices)
            for i, slice in enumerate(self.coordslices):
                arr = np.array(slice)
                g.create_dataset('{:04d}'.format(i), data=arr)

    @classmethod
    def fromHDF5(cls, path):
        """reconstruct object from serialized data in h5 format"""
        import h5py
        self = cls()
        path = ensure_extension(path, '.h5')
        with h5py.File(path, 'r') as f:
            self.roinumber = int(f.attrs['roinumber'])
            self.roiname = str(f.attrs['roiname'])
            self.refforuid = str(f.attrs['refforuid'])
            self.frameofreference = FrameOfReference(
                tuple(f.attrs['FrameOfReference.start']),
                tuple(f.attrs['FrameOfReference.spacing']),
                tuple(f.attrs['FrameOfReference.size'])
            )
            self.coordslices = []
            for k in sorted(f['coordslices'].keys()):
                points = []
                data = f['coordslices'][k]
                npdata = np.empty(data.shape, dtype=data.dtype)
                data.read_direct(npdata)
                for i in range(data.shape[0]):
                    points.append(tuple(npdata[i, :]))
                self.coordslices.append(points)
        return self
