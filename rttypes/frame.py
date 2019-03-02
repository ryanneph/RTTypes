import copy
import math
import logging

import numpy as np
import pydicom

from . import dcmio

# initialize module logger
logger = logging.getLogger(__name__)


class FrameOfReference:
    """Defines a dicom frame of reference to which BaseVolumes can be conformed for fusion of pre-registered
    image data
    """
    def __init__(self, start=None, spacing=None, size=None, UID=None):
        """Define a dicom frame of reference

        Args:
            start    -- (x,y,z) describing the start of the FOR (mm)
            spacing  -- (x,y,z) describing the spacing of voxels in each direction (mm)
            size     -- (x,y,z) describing the number of voxels in each direction (integer)
            UID      -- dicom FrameOfReferenceUID can be supplied to support caching in BaseVolume

        Standard Anatomical Directions Apply:
            x -> increasing from patient right to left
            y -> increasing from patient anterior to posterior
            z -> increasing from patient inferior to superior
        """
        self.start = start
        self.spacing = spacing
        self.size = size
        self.UID = UID

    @classmethod
    def fromDatasetList(cls, dataset_list):
        # check that all elements are valid slices, if not remove and continue
        nRemoved = 0
        for i, slice in enumerate(dataset_list):
            if (not isinstance(slice, pydicom.dataset.Dataset)):
                logger.debug('invalid type ({t:s}) at idx {i:d}. removing.'.format(
                    t=str(type(slice)),
                    i=i ) )
                dataset_list.remove(slice)
                nRemoved += 1
            elif (len(slice.dir('ImagePositionPatient')) == 0):
                logger.debug('invalid .dcm image at idx {:d}. removing.'.format(i))
                dataset_list.remove(slice)
                nRemoved += 1
        if (nRemoved > 0):
            logger.info('# slices removed with invalid types: {:d}'.format(nRemoved))

        # sort datasets by increasing slicePosition (inferior -> superior)
        dataset_list.sort(key=lambda dataset: dataset.ImagePositionPatient[2], reverse=False)

        # build object properties
        start = dataset_list[0].ImagePositionPatient
        spacing = (*dataset_list[0].PixelSpacing, dataset_list[0].SliceThickness)
        try:
            # some modalities don't provide NumberOfSlices attribute
            size = (dataset_list[0].Columns, dataset_list[0].Rows, dataset_list[0].NumberOfSlices)
        except:
            # use length of list instead
            size = (dataset_list[0].Columns, dataset_list[0].Rows, len(dataset_list))

        UID = dataset_list[0].FrameOfReferenceUID
        return cls(start, spacing, size, UID)

    @classmethod
    def fromDir(cls, path, recursive=False):
        dataset_list = dcmio.read_dicom_dir(path, recursive=recursive, only_headers=True)
        return cls.fromDatasetList(dataset_list)

    def copy(self):
        new = FrameOfReference()
        new.start = copy.deepcopy(self.start)
        new.size = copy.deepcopy(self.size)
        new.spacing = copy.deepcopy(self.spacing)
        new.UID = copy.deepcopy(self.UID)
        return new

    def __repr__(self):
        return '{!s}:\n'.format(self.__class__) + \
               '  start   <mm> (x,y,z): ({:0.3f}, {:0.3f}, {:0.3f})\n'.format(*self.start) + \
               '  spacing <mm> (x,y,z): ({:0.3f}, {:0.3f}, {:0.3f})\n'.format(*self.spacing) + \
               '  size    <mm> (x,y,z): ({:d}, {:d}, {:d})\n'.format(*self.size)

    def __eq__(self, compare):
        if (self.start   == compare.start and
            self.spacing == compare.spacing and
            self.size    == compare.size):
            return True
        else: return False

    def changeSpacing(self, new_spacing):
        """change frameofreference resolution while maintaining same bounding box
        Changes occur in place, self is returned
            Args:
                new_spacing (3-tuple<float>): spacing expressed as (X, Y, Z)
        """
        old_spacing = self.spacing
        old_size = self.size
        self.spacing = new_spacing
        self.size = tuple((np.array(old_size) * np.array(old_spacing) / np.array(self.spacing)).astype(int).tolist())
        return self

    def end(self):
        """Calculates the (x,y,z) coordinates of the end of the frame of reference (mm)
        """
        # compute ends
        end = []
        for i in range(3):
            end.insert(i, self.spacing[i] * self.size[i] + self.start[i])

        return tuple(end)

    def volume(self):
        """Calculates the volume of the frame of reference (mm^3)
        """
        length = []
        end = self.end()
        vol = 1
        for i in range(3):
            length.insert(i, end[i] - self.start[i])
            vol *= length[i]

        return vol

    def getIndices(self, position):
        """Takes a position (x, y, z) and returns the indices at that location for this FrameOfReference

        Args:
            position  -- 3-tuple of position coordinates (mm) in the format: (x, y, z)
        """
        indices = []
        for i in range(3):
            indices.insert(i, math.floor(int(round((position[i] - self.start[i]) / self.spacing[i] ))))

        return tuple(indices)

