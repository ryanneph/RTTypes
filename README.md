# RTTypes
This is a library written in Python 3.x for common tasks when dealing with the DICOM medical image format in a research setting. rttypes is a shorthand for Radiotherapy-types, class definitions for common data formats used in radiation therapy treatment planning and delivery, such as CT/MR images, and regions of interest (ROIs).
## Overview
Key features include:
* Easy I/O of dicom images/volumes commonly used in storage of CT, MR, and PET images.
* Reading and conversion of Radiotherapy contours/masks from .rtstruct file to 2D/3D binary masks.

## Upcoming Changes/Enhancements
rttypes will be updated periodically when time permits to become more functional and robust. Please stay tuned.
1. Unit Tests
2. Documentation Page

## Installing & Updating
Open a terminal window and enter:
``` bash
pip3 install --upgrade git+git://github.com/ryanneph/rttypes.git
```

## Development
Open a terminal window and enter:
``` bash
git clone https://github.com/ryanneph/rttypes.git
cd rttypes
pip3 install -e .
```

---
## Example Usage
```python
from rttypes.volume import Volume
from rttypes.roi import ROI

# load a set of axial slices into a Volume object with its coordinate system information
ctvol = Volume.fromDir('./ct_directory')
print(ctvol)
print(ctvol.frame)
# get access to raw numpy array of Hounsfield Unit values
print(ctvol.data.dtype, ctvol.data.shape)

# load a named organ structure from an rtstruct file
#   and convert it into a binary volume mask
roi = ROI.roiFromFile('./rtstruct.dcm', 'O_HEART)
roimask = roi.makeDenseMask(ctvol.frame)
```

## Contributing
If you'd like to get involved in contributing to this project, contact Ryan Neph at neph320@gmail.com.
