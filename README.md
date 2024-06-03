# 3D slicer AC3T dataloader

* Load HDF files in 3D Slicer
* Load data from a custom REST Server (load from remote files, run inference on GPU server and display in slicer...)

## HDF
* Loads volumes from ``Volume`` dataset. Volumes can be 3D or 4D ([N],X,Y,Z]). For 4D Volumes a slicer Sequence is created.
* Loads projections from ``Projection*`` datasets
* Loads segmentation from ``TotalSegmentator`` datasets

Flexibility is limited and only files following a specific format(TM) are supported.

## Installation
Don't know...