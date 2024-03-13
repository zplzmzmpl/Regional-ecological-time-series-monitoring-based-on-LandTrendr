# Regional-ecological-time-series-monitoring-based-on-LandsatTrendr
*Use long-term series remote sensing image data obtained by LLR and LT for time series clustering and classification based on deep learning.*

## STEP 1: Get Your Date

**Using [`ee-LandsatLinkr`](https://github.com/gee-community/ee-LandsatLinkr) tools in GEE platform or python scrips developed by *Justin Braaten* & *Annie Taylor*.**

- ### OPTION 1: Gather your data with `google colab`

  Click me to get it!⚓ [![Click here to use it](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gee-community/ee-LandsatLinkr/blob/main/colab_template.ipynb)

- ### OPTION 2: Gather your data in [`GEE`](https://code.earthengine.google.com/2ec3c28efc3ecf15504979a9698a8b0d?noload=true)
<p align="center">
  <img width="600" height="300" src="https://i.postimg.cc/ZYQphMtL/01.png">
</p>

*then you can get these asset in your `GEE` account*
<p align="center">
    <img src="https://i.postimg.cc/bvKjQnZb/02.jpg">
</p>

*after this you can get data after executing `LandasatLinkr`*
<p align="center">
    <img width='600' height='300' src="https://i.postimg.cc/d1QB1HBV/03.png">
</p>

*now you need to storage data in `Google Drive` before you download it to local(becouse of gee doesn't support to dowanload to local directly)*
<p align="center">
    <img width='600' height='300' src="https://i.postimg.cc/ZnRNN00G/04.png">
</p>

---
> [!NOTE]  
> These steps will cost lots of time, keep patient🛏️.
---

*and check data in ENVI*
<p align="center">
    <img width='600' height='300' src="https://i.postimg.cc/vZxGBnLJ/05.png">
</p>

Congratulations!㊗️ As now you have got the original data!🎆

## STEP 2: Fit Change Curve(e.g. TCG index)
- **use IDL to execute [`LandsatTrendr`](https://github.com/jdbcode/LLR-LandTrendr)**[^1]
  - *we develop a GUI to help users to use it.*
  <p align="center">
    <img width='600' height='500' src="https://i.postimg.cc/hPTkFLLv/13.png">
  </p>
  
  - *you'd better know what's meanings of these parameters before run it, following below intro:*
    | **Parameter**          | **Type**          | **Default** | **Definition** |
    |------------------------|-------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
    | maxSegments            | _Integer_         |             | Maximum number of segments to be fitted on the time series|
    | spikeThreshold         | _Float_           | _0.9_       | Threshold for dampening the spikes (1.0 means no dampening)|
    | vertexCountOvershoot   | _Integer_         | _3_         | The inital model can overshoot the maxSegments + 1 vertices by this amount. Later, it will be prunned down to maxSegments + 1                      |
    | preventOneYearRecovery | _Boolean_         | _false_     | Prevent segments that represent one year recoveries                                                                                                |
    | recoveryThreshold      | _Float_           | _0.25_      | If a segment has a recovery rate faster than 1/recoveryThreshold (in years), then the segment is disallowed                                        |
    | pvalThreshold          | _Float_           | _0.1_       | If the p-value of the fitted model exceeds this threshold, then the current model is discarded and another one is fitted using the Levenberg-Marquardt optimizer|
    | bestModelProportion    | _Float_           | _1.25_      | Takes the model with most vertices that has a p-value that is at most this proportion away from the model with lowest p-value                      |
    | minObservationsNeeded  | _Integer_         | _6_         | Min observations needed to perform output fitting                                                                                                  |
    | timeSeries             | _ImageCollection_ |             | Collection from which to extract trends (it’s assumed that each image in the collection represents one year). The first band is used to find breakpoints, and all subsequent bands are fitted using those breakpoints |

  - *now see what we get*
  <p align="center">
    <img width='600' height='300' src="https://i.postimg.cc/28T5WL6H/14.png">
    <img src="https://i.postimg.cc/pT8fVJPc/17.png">
  </p>

  - *apply LT to all image, we get time series in 3 bands(ndvi/tcg/tcw), Salty Lake City as example*
  <p align="center">
    <img width='500' height='500' src="https://i.postimg.cc/J05HvJ80/2.png">
  </p>
  but LT may be disable in some data (AKA `noise`), so after executing LT we had better apply a median filtering.
    <p align="center">
    <img width='300' height='300' src="https://i.postimg.cc/wTc49b24/2.png" hspace=10>
      <img width='300' height='300' src="https://i.postimg.cc/nzS2X6gp/2.png" hspace=10>
  </p>

  Congratulations! As now you have got the processed data after LT algorithm!🤞
  If you want to know more infomation about LandTrendr algorithm, we recommend you follow this [`link`](https://emapr.github.io/LT-GEE/index.html)🥳

  ## STEP 3: Time Series Clustering
  In this step we use KShape[^3] algorothm to achieve our ts data clustering. Before we begain, we'd better know what's `KShape`?
  
  The KShape clustering method is a clustering algorithm based on time series data. It groups time series into different clusters by calculating the similarity between them. The key to the KShape clustering method is to match the shape of the time series, not just the numerical value. This enables KShape to discover time series that have similar shapes but not necessarily similar values. The KShape clustering method has wide applications in data analysis in various fields, including finance, medical and weather prediction.The basic steps of the KShape clustering method include:
  - 1. Select the time series data set to cluster.
  - 2. Calculate the similarity between time series, usually using methods such as dynamic time warping (DTW).
  - 3. Clustering based on similarity, commonly used methods include k-means algorithm.
  - 4. Analyze the clustering results and perform further interpretation and application as needed.
   

[^1]:Kennedy, Robert E., Yang, Zhiqiang, & Cohen, Warren B. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr - Temporal segmentation algorithms. Remote Sensing of Environment, 114, 2897-2910
[^2]:Zhen Yang, Jing Li, Carl E. Zipper, Yingying Shen, Hui Miao, Patricia F. Donovan, Identification of the disturbance and trajectory types in mining areas using multitemporal remote sensing images,Science of The Total Environment,Volume 644,2018,Pages 916-927,ISSN 0048-9697,https://doi.org/10.1016/j.scitotenv.2018.06.341.
[^3]:John Paparrizos Columbia University jopa@cs.columbia.edu Luis Gravano Columbia University gravano@cs.columbia.edu k-Shape: Efficient and Accurate Clustering of Time Series. 
