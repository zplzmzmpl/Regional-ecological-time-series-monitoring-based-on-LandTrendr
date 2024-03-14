# Regional-ecological-time-series-monitoring-based-on-LandsatTrendr
*Use long-term series remote sensing image data obtained by LLR and LT for time series clustering and classification based on deep learning.*

## STEP 1: Collect Data

**Using [`ee-LandsatLinkr`](https://github.com/gee-community/ee-LandsatLinkr) tools in GEE platform or python scrips developed by *Justin Braaten* & *Annie Taylor*.**

- ### OPTION 1: Collect your data with `google colab`

  Click me to get it!⚓ [![Click here to use it](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gee-community/ee-LandsatLinkr/blob/main/colab_template.ipynb)

- ### OPTION 2: Collect your data in [`GEE`](https://code.earthengine.google.com/2ec3c28efc3ecf15504979a9698a8b0d?noload=true)
  *Follow these steps to complete your data collection*

  - View WRS-1 granules - figure out what WRS-1 granule to process
  - Make a processing [dir](https://gist.github.com/jdbcode/36f5a04329d5d85c43c0408176c51e6)
  - Create MSS WRS-1 reference image - for MSS WRS1 to MSS WRS2 harmonization
  - View WRS-1 collection - identify bad MSS images
  - Prepare MSS WRS-1 images
  - Get TM-to-MSS correction coefficients
  - Export MSS-to-TM corrected images
  - Inspect the full time series collection - explore time series via animation and inspector tool to check for noise
  - Run LandTrendr and display the fitted collection on the map
  - Display the year and magnitude of the greatest disturbance during the time series

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

  *now you need to storage data in `Google Drive` before you download it to local(becouse of gee doesn't support you dowanload to local directly)*
  *you can follow this code to check data and download it to Drive*
	
 
		 // Import the LandsatLinkr module
		 var llr = require('users/jstnbraaten/modules:landsatlinkr/landsatlinkr.js');
		 
		 var lt = ee.Image('projects/ee-opppmqqqo/assets/LandsatLinkr/041032/landtrendr')
		 var start_yr = 1972
		 var end_yr = 2022;
		 // This has to match the three bands listed in the LT_params ftvBands parameter
		 // when the landtrendr output was created (in the final step of the colab notebook)
		 var rgb_bands = {
		  r: 'ndvi',
		  g: 'tcg',
		  b: 'tcw'
		};
		var vis_params = {
		  min: 100,
		  max: 2000,
		  gamma: 1.2
		};
		
		var video_params = {
		  'dimensions': 512,
		  'crs': 'EPSG:3857',
		  'framesPerSecond': 8,
		};
		//print(lt.select('LandTrendr'));
		Map.addLayer(lt.select('LandTrendr'), {min:0,max:20000}, 'landtrendr', true);
		
		
		// Create the 50 year image collection from the LandTrendr output
		var fittedRGBCols = llr.getFittedRgbCol(lt, start_yr, end_yr, rgb_bands, vis_params);
		//print(fittedRGBCols);
		// Get the ImageCollection
		var collection = ee.ImageCollection(fittedRGBCols.rgb); // replace 'rgb' with the actual ID of your ImageCollection
		print(collection);
		
		var batch = require('users/fitoprincipe/geetools:batch')
		//COLLECTION
		batch.Download.ImageCollection.toDrive(collection,"041032", {
		scale: 30,
		crs:'EPSG:3857',
		region: geometry,
		type:"float" });`


<p align="center">
    <img width='600' height='300' src="https://i.postimg.cc/ZnRNN00G/04.png">
</p>

  *and then check data in ENVI*
<p align="center">
    <img width='600' height='300' src="https://i.postimg.cc/vZxGBnLJ/05.png">
</p>

Congratulations!㊗️ As now you have got the original data!🎆

---
> [!NOTE]  
> These steps will cost lots of time, keep patient🛏️.
---

## STEP 2: Fit Change Curve(e.g. TCG index)
- **use IDL to execute [`LandTrendr`](https://github.com/jdbcode/LLR-LandTrendr)**[^1]
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
    - **fitted curve**
    - **vertices and values, like [1972,1999,2002,2021]**
    - **final segments**
    - **[p value](https://www.investopedia.com/terms/p/p-value.asp)(lower is better)**
    - **[f statistic](https://www.statisticshowto.com/probability-and-statistics/f-statistic-value-test/)(bigger is better)**
  <p align="center">
    <img width='600' height='300' src="https://i.postimg.cc/28T5WL6H/14.png">
    <img src="https://i.postimg.cc/pT8fVJPc/17.png">
  </p>

  - *apply LT to all image, we get time series in 3 bands(ndvi/tcg/tcw), Salty Lake City as example*
  <p align="center">
    <img width='500' height='500' src="https://i.postimg.cc/J05HvJ80/2.png">
  </p>
	
  - *but LT may be disable in some data (AKA `noise`), so after executing LT we had better apply a median filtering.*
    <p align="center">
    <img width='300' height='300' src="https://i.postimg.cc/wTc49b24/2.png" hspace=10>
      <img width='300' height='300' src="https://i.postimg.cc/nzS2X6gp/2.png" hspace=10>
  </p>

  Congratulations! As now you have got the processed data after LT algorithm!🤞
  If you want to know more infomation about LandTrendr algorithm, we recommend you follow this [`link`](https://emapr.github.io/LT-GEE/index.html)🥳

  ## STEP 3: Time Series Clustering
  In this step we use **KShape**[^3] algorothm to achieve our ts data clustering. Before we begain, we'd better know what's **KShape**?
  
  The KShape clustering method is a clustering algorithm based on time series data. It groups time series into different clusters by calculating the similarity between them. The key to the KShape clustering method is to match the shape of the time series, not just the numerical value. This enables KShape to discover time series that have similar shapes but not necessarily similar values. The KShape clustering method has wide applications in data analysis in various fields, including finance, medical and weather prediction.The basic steps of the KShape clustering method include:
  - Select the time series data set to cluster.
  - Calculate the similarity between time series, usually using methods such as dynamic time warping (DTW).
  - Clustering based on similarity, commonly used methods include k-means algorithm.
  - Analyze the clustering results and perform further interpretation and application as needed.

  **There are two ways to use KShape by `python`**
- KShape integrated in sci-learn(only CPU engage in calulation)
  there is a simple example:
  
  	  from tslearn.clustering import KShape
	  def kshape_cpu(X, k):
	    print('begin kshape....\n')
	    seed = 0
	    ksc = KShape(n_clusters=k, n_init=5, verbose=True, random_state=seed)
	    ksc.fit(X)
	    print('\nend kshape\n')
	    return ksc


- Independent KShape lib(you can call GPU to accelerate calculation efficiency)
   there is a simple example:

	   from kshape import KShapeClusteringGPU
	   def kshape_gpu(X,k):
	     print('begin kshape gpu...\n')
	     ksg = KShapeClusteringGPU(n_clusters=k)
	     ksg.fit(np.expand_dims(X, axis=2))
	     print('\nend kshape gpu...')
	     return ksg
  
before we run KShape code, we need to define fixed number of clustering, for this we use **Elbow Law** to check probable number of clustering.

       distortions = []
       for i in range(4, 8):
           print('cluster num:', i, '\n-------------')
           ks = KShape(n_clusters=i, n_init=5, verbose=True, random_state=0)
           # Perform clustering calculation
           ks.fit(X)
           distortions.append(ks.inertia_)
        plt.plot(range(2, 7), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()
  
  code above will return a plot like this, you can see *slop* become more gentle when value of *x* equal 5, so we confirm number of clustering is 5
  <p align="center">
    <img width='500' height='400' src="https://i.postimg.cc/ydNRrcHn/2.png">
  </p>

  
[^1]:Kennedy, Robert E., Yang, Zhiqiang, & Cohen, Warren B. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr - Temporal segmentation algorithms. Remote Sensing of Environment, 114, 2897-2910
[^2]:Zhen Yang, Jing Li, Carl E. Zipper, Yingying Shen, Hui Miao, Patricia F. Donovan, Identification of the disturbance and trajectory types in mining areas using multitemporal remote sensing images,Science of The Total Environment,Volume 644,2018,Pages 916-927,ISSN 0048-9697,https://doi.org/10.1016/j.scitotenv.2018.06.341.
[^3]:John Paparrizos Columbia University jopa@cs.columbia.edu Luis Gravano Columbia University gravano@cs.columbia.edu k-Shape: Efficient and Accurate Clustering of Time Series. 
