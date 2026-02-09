## 3D Cell Segmentation Workflow

### Overview

3D cell segmentation is a core module of the **CT3Dcellas** framework, which enables cell-level segmentation of rice embryo CT images based on the **Cellpose**, supporting automated output from raw CT images to cell contours/labels.

### Workflow Pipeline

#### 1. Input Data

The input is a 3D CT image sequence (example data: reconstructed images of rice embryo 24H_AntherS1 with 2.07μm resolution), and the data format must be tif (supporting multi-channel/single-channel 3D stacks).

#### 2. Segmentation Command

Execute 3D segmentation via Cellpose using a pre-trained rice embryo cell segmentation model:

```bash
python -m cellpose \
  --dir "G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/avizoProj/10originalAngle-3Dtif/final" \  # Directory of input CT images
  --diameter 0 \  # Automatically calculate cell diameter
  --use_gpu \  # Enable GPU acceleration (recommended)
  --save_png \  # Save segmentation results as PNG
  --save_tif \  # Save segmentation labels as 3D stacked TIF
  --save_outlines \  # Save cell outlines
  --save_flows \  # Save Cellpose flow field maps
  --save_txt \  # Save cell coordinate information
  --do_3D \  # Enable 3D segmentation mode
  --verbose \  # Output detailed logs
  --pretrained_model "models/cellpose_residual_on_style_on_concatenation_off_traindata7.25-originalPng-4_2023_07_25_01_58_50.651179"  # Path to pre-trained model

```

#### 3. Datasets

● **Train Dataset**: Open-sourced in the GitHub repository, including rice embryo CT images and corresponding manually annotated labels:

● **Test Dataset**: Example test sets are provided (including raw CT images, Ground Truth, and segmentation results), with visualization effects as follows:

| Input CT Image | Ground Truth | Segmentation Output |
|----------------|--------------|---------------------|
| <img src="origin-236_row3_col2.png" width="200"> | <img src="origin-236_row3_col2_cp_masks.png" width="200"> | <img src="origin-236_row3_col2_cp_masks.png" width="200"> |
> *Table 1: Visual comparison of input CT images, expert-annotated ground truth, and 3D cell segmentation outputs for the 24H_AntherS1 sample.*

#### 4. Output Files

After segmentation is completed, the following files will be generated in the output directory:

 _seg.npy: 3D array of cell labels (Numpy format)

 _seg.tif: 3D segmentation label image (can be directly imported into 
ImageJ/Avizo for visualization)

#### Notes

1.If GPU is unavailable, remove the --use_gpu parameter to automatically switch to CPU (slower speed);
2.To retrain the model, adjust parameters via Cellpose's train mode based on the open-source train dataset;
3.Segmentation results can be directly used for subsequent modules of 3D cell atlas construction and multi-omics data mapping.

