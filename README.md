# Improved-Mask-R-CNN-for-nuclei-segmentation

This repository contains the implementational parts of our COMPAY2021 submission as part of the MICCAI conference. It contains the notebooks for the described experiments. All implementations are based on the [matterport implementation of Mask R-CNN](https://github.com/matterport/Mask_RCNN). 

## Set-up and environment
### Requirements
```
pip install -r req.txt
```
### Datasets
As part of this release we used the [MoNuSeg2018](https://monuseg.grand-challenge.org/) and [PanNuke](https://jgamper.github.io/PanNukeDataset/) datasets.
The additionally used hematoxylin-stain intensities and estimated distance maps can be found [HERE](LINK)

## Data and Folder structure
Data is loaded using the MoNuSegDataset class defined in `MoNuSeg.py`. It extends the Dataset class of the matterport implementation using the `load_image` function. It contains 2 modes based on whether the ground truth (GT), or estimated (U-Net) data should be loaded. Examples of using the Dataset class can be found in our `Dataset_Notebook`.


All data is placed into the `datasets/MoNuSeg/[train/val]/` folder and is then further split into:
- `./distance_maps/`: estimated distance maps for the test set and ground truth distance maps for the training set.
- `./H-stain/`: extracted hematoxylin stain intensities
- `./mask_binary/`: ground truth masks (saved as binary HxWxN matrices)
- `./tissue_images/`: RGB-images in .tif format

Trained models get saved automatically into a `./logs` folder. 
!!! NOTE: I have uploaded a model [HERE](LINK) if you want to try running the code without having to train a model from sratch - we could provide the Pannuke pretrained one. !!!


# Running the code

After all setup is complete, the code can be run in the Jupyter Notebooks. Further information can be found in the respective notebooks:
- `Dataset_Notebook`: how to use the dataset class to add, load and display data within the Dataset class.
- `Detection_Notebook`: how to run inference using the provided Mask R-CNN. It also showcases how to visualize detections in Mask R-CNN.
- `Ensemble_Notebook`: ensembling inference method step-by-step.
- `Merge_Notebook`: our replacement of the traditional non-maximum suppression used in Mask R-CNN.
- `TTA_Notebook`: our implementation of test-time augmentation.
- `Train_Notebook`: train Mask R-CNN+ from COCO or PanNuke pretrained weights, and its ensemble variants.
- `Validation_Notebook`: get validation scores on the MoNuSeg test-set using the `stats_utils.py` from [HoVerNet](https://github.com/vqdang/hover_net).

# Config differences from matterport
The configs containing model-setup parameters are part of `MoNuSeg.py`. The major differences from the matterport implementations include
- Added support for 5 channel image inputs.
- Slightly increased `RPN_NMS_THRESHOLD` to generate more proposals.
- Adapted `DETECTION_MAX_INSTANCES`, `MAX_GT_INSTANCES`, and `TRAIN_ROIS_PER_IMAGE` to better reflect the high density data.
- Reduced `RPN_ANCHOR_SCALES` to reflect the probable sizes of candidate nuclei regions.

# Credits: 
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN): Original Mask R-CNN implementation
- [HoVerNet](https://github.com/vqdang/hover_net): stats_utils.py

