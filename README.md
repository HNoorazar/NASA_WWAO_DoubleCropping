### NASA WWAO Double-Cropping
Classification of single- vs. double-cropped fields via satellite imagery.

This repo contains the codes used in the paper **Identifying Double-cropped Fields with Remote Sensing in Areas with High Crop Diversity**; **DOI:** 

Here is a step-by-step demo from Google Earth Engine (GEE) to the final product. 

________________________
 ## 1. Compute and export vegetation indices.
 At this point, this step is done in GEE in JavaScript. We will have a Python version soon.
  - [Compute NDVI and EVI using Landsat 7 on GEE (Aug 31, 2023)](https://code.earthengine.google.com/821e1704014d498954fa91d6eda8e5b4)
  - [Compute NDVI and EVI using Landsat 8 on GEE (Aug 31, 2023)](https://code.earthengine.google.com/f48b0081245b535f47a6c6b96558a938)

## 2. Denoise and predict
The Jupyter notebook (```smoothing_and_classification_demo.ipynb```) demo:

* reads ```.csv``` files of vegetation indices that are outputs of GEE (that are in the ```data``` folder),
* applies the smoothing steps to ```NDVI```,
* reads the trained deep learning model (that is in ```model.zip```. Must unzip.),
* plots ```NDVI``` time-series of each field, one at a time, writes them on the disk (in the ```figures``` folder), reads them, and makes predictions.
* ________________________
* The modules called ```NASA_WWAO_core.py``` and ```NASA_WWAO_plot_core.py``` contain the functions needed for the process. 
* The ```data``` folder contains a shapefile (that includes 4 fields in Grant County, WA, USA) and two ```.csv``` files. These files are outputs of GEE which include ```NDVI``` and ```EVI``` time series for the four aforementioned fields. One of the ```.csv``` files is from Landsat 7 and the other is from Landsat 8.
________________________
**Remark.** 
When I used GEE I was able to process all of my Vegetation Index (VI) 
computations (thousands of fields and several years) on JAVA platform, however, by the end of the project I 
had a lot of memory issues and was forced to use ```for-loop``` 
in CoLab (Python) to fetch and export the data.
________________________
## Disclaimer
... coming soon ...
