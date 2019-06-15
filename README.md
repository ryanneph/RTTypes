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
3. Examples and Getting Started Guide

## Installing
Open a terminal window and enter:
``` bash
pip3 install git+git://github.com/ryanneph/rttypes.git#egg=rttypes
```

## Updating
Open a terminal window and enter:
``` bash
pip3 install --upgrade git+git://github.com/ryanneph/rttypes.git#egg=rttypes
```

## Development
Open a terminal window and enter:
``` bash
git clone https://github.com/ryanneph/rttypes.git
cd rttypes
pip3 install -e .
```

## Contributing
If you'd like to get involved in contributing to this project, contact Ryan Neph at neph320@gmail.com.
