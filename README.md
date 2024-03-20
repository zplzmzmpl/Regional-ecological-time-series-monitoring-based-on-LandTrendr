# Regional-ecological-time-series-monitoring-based-on-LandsatTrendr
*This repo use long-term series remote sensing image data obtained by LLR and LT for time series clustering and classification based on deep learning.*

## 🧭*Navigation*

- [1.Collect Data](#step-1-collect-data)
- [2.Fit Change Curve](#step-2-fit-change-curveeg-tcg-index)
- [3.Time Series Clustering](#step-3-time-series-clustering)
  - [3.1 Intro KShape Clustering](#intro-kshape)
  - [3.2 Usage in Python](#usage-in-python)
  - [3.3 Evaluation](#evaluation)
- [4.Time Series Classification](#step-4-time-series-classification)
  - [4.1 Prepare Data](#prepare-training-data)
  - [4.2 Training](#training-model)
  - [4.3 Result](#classification-result)
  - [4.5 TimeSeries2Image](#%EF%B8%8Ftime-series-to-image)
    - [4.5.1 Convert ts 2 Img](#convert-time-series-to-images-classification)
    - [4.5.2 Begin Training](#begin-training)
    - [4.5.3 Apply to RS Img](#apply-model-to-rs-image)
- [Output](#output-images)
- [To Do List](#to-do)
---

## 🤗STEP 1: Collect Data

**Using [`ee-LandsatLinkr`](https://github.com/gee-community/ee-LandsatLinkr) tools in GEE platform or python scrips developed by *Justin Braaten* & *Annie Taylor*.**

- ### OPTION 1: Collect data with `google colab`🍕

  Click me to get it!⚓ [![Click here to use it](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gee-community/ee-LandsatLinkr/blob/main/colab_template.ipynb)

- ### OPTION 2: Collect data in [`GEE`](https://code.earthengine.google.com/2ec3c28efc3ecf15504979a9698a8b0d?noload=true)🌏
  *Follow these steps to complete your data collection*

  - View ***WRS-1*** granules - figure out what WRS-1 granule to process
  - Make a processing [***dir***](https://gist.github.com/jdbcode/36f5a04329d5d85c43c0408176c51e6)
  - Create MSS WRS-1 ***reference*** image - for MSS WRS1 to MSS WRS2 harmonization
  - View ***WRS-1 collection*** - identify bad MSS images
  - Prepare MSS WRS-1 images
  - Get ***TM-to-MSS correction coefficients***.
  - Export MSS-to-TM ***corrected*** images
  - Inspect the full ***time series collection*** - explore time series via animation and inspector tool to check for noise
  - Run ***LandTrendr*** and display the fitted collection on the map
  - Display the year and ***magnitude of the greatest disturbance*** during the time series

*the tutorial is detailed enough, you will get study area data named `LandTrendr` you needed if you follow it step by step in `asset` filefolder within your GEE account. you need to storage data in `Google Drive` before you download it to local(becouse of gee doesn't support you dowanload to local directly). you can follow this code to check data and download it to Drive:*
		
  		var lt = ee.Image('projects/your_account_name/assets/LandsatLinkr/041032/landtrendr');
		// 50 Years! 
		var start_yr = 1972;
		var end_yr = 2022;
		var ltr =  lt.select('LandTrendr');
  
		// Create the 50 year image collection from the LandTrendr output
		var fittedRGBCols = llr.getFittedRgbCol(lt, start_yr, end_yr, rgb_bands, vis_params);
  
		// Get the ImageCollection
		var collection = ee.ImageCollection(fittedRGBCols.rgb); // replace 'rgb' with the actual ID of your ImageCollection
		
		var batch = require('users/fitoprincipe/geetools:batch')
		batch.Download.ImageCollection.toDrive(collection,"041032", {
			scale: 30,
		 	crs:'EPSG:3857',
		  	region: geometry,
		   	type:"float" });
      
*you also can use this [tool](https://emaprlab.users.earthengine.app/view/lt-gee-time-series-animator) developed by ***Justin Braaten*** to see the change map of your study area.*

<p align="center">
    <img src="https://github.com/zplzmzmpl/Regional-ecological-time-series-monitoring-based-on-LandsatTrendr/blob/main/asset/slc1984-2023_rgb.gif">
</p>

***Congratulations!㊗️ As now you have got the original data!🎆***

---
> [!NOTE]  
> These steps will cost lots of time, keep patient🛏️.
---

## 😇STEP 2: Fit Change Curve(e.g. TCG index)
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

  ## 🤖STEP 3: Time Series Clustering
  In this step we use **KShape**[^3] algorothm to achieve our ts data clustering. Before we begain, we'd better know what's **KShape**?

  ### *Intro KShape*
  ---
  The KShape clustering method is a clustering algorithm based on time series data. It groups time series into different clusters by calculating the similarity between them. The key to the KShape clustering method is to match the shape of the time series, not just the numerical value. This enables KShape to discover time series that have similar shapes but not necessarily similar values. The KShape clustering method has wide applications in data analysis in various fields, including finance, medical and weather prediction.The basic steps of the KShape clustering method include:
  - Select the time series data set to cluster.
  - Calculate the similarity between time series, usually using methods such as dynamic time warping (DTW).
  - Clustering based on similarity, commonly used methods include k-means algorithm.
  - Analyze the clustering results and perform further interpretation and application as needed.

  ### *Usage in python*
  ---
  **There are two ways to use KShape by `python`**
- KShape integrated in tslearn(only CPU engage in calulation)
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
  
   ### *Elbow Law*
   ---
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
  
  code above will return a plot like this, you can see *slop* become more gentle when value of *x* equal *5*, so we confirm number of clustering is *5*
  <p align="center">
    <img width='500' height='400' src="https://i.postimg.cc/ydNRrcHn/2.png">
  </p>

  but sometimes it is doesn't obvious just by this way, so we suggest use `yellowbrick` lib to visualize k-elow-law:
	
	 	from yellowbrick.cluster import KElbowVisualizer	
		#Instantiate the clustering model and visualizer
		model = KShape(n_init=1, verbose=True, random_state=0)
		
		#distortion: mean sum of squared distances to centers
		#silhouette: mean ratio of intra-cluster and nearest-cluster distance
		#calinski_harabasz: ratio of within to between cluster dispersion
		
		visualizer = KElbowVisualizer(model, k=(4,10),metric='distortion')
		
		visualizer.fit(img_data)        # Fit the data to the visualizer
		visualizer.show()        # Finalize and render the figure
  
  here are visualizations of `elbow law` using three evaluation indexs with `yellowbrick` lib.

<p align="center">
<img width='400' height='300' src="https://i.postimg.cc/fR31Cp9m/Calinski-Harabasz-Score-Elbow-for-KShape-Clustering.png">
<img width='400' height='300' src="https://i.postimg.cc/7ZvNvWRg/Distortion-Score-Elbow-for-KShape-Clustering.png">
<img width='400' height='300' src="https://i.postimg.cc/pV470xDC/Silhouette-Score-Elbow-for-KShape-Clustering.png">
</p>

  now we can apply KShape to image data we got above with 51 bands, you can just cluster ***Univariate*** data, also you can cluster ***Multivariate*** data.
  
  **First of all, read data and convert to fomat of time series.**
  - for univariate data clustering, like ndvi ts data [length(number of ts), time_span(51 for this), var(1 for this)]

	    import rasterio
	    #Load the .tif image
	    with rasterio.open("/content/NDVI51years_median.tif") as src:
	        image_data = src.read()
	        row = image_data.shape[1]
	        column = image_data.shape[2]
	        # Reshape to (rows, columns, bands)
	        transposed_image_data = np.transpose(image_data, (1, 2, 0))
	
	    print('transposed shape:', transposed_image_data.shape)
	
	    #Reshape the image data to (rows * columns, bands)
	    img_data = transposed_image_data.reshape(-1,transposed_image_data.shape[2])
	    print('reshaped shape:', img_data.shape)

  - for multivariate data clustering,like [ndvi, tcg, tcw] ts data [length(number of ts), time_span(51 for this), var(3 for this)]
    
		import rasterio
		import numpy as np
		from rasterio.transform import Affine
	
		#define image paths
		image_paths = [
			    '/*/*/*/ndvi51years_median.tif',
			     ·········
			    '/*/*/*/tcw51years_median.tif'
			]
		#read image data and reshape to 4D array
		combined_array = []
		for path in image_paths:
	    	with rasterio.open(path) as src:
	        	array = src.read()
	        	reshaped_array = array.reshape(array.shape[0], -1)
	        	combined_array.append(reshaped_array)
	        	if path == image_paths[0]:
	          		metadata = src.profile
	          		print(metadata)
	
		#convert list to array
		combined_array = np.array(combined_array)
	
		#transpose array to fomat as requires
		combined_array = combined_array.transpose(2, 1, 0)
	
		print(combined_array.shape)
		#save as npy file, you will don't need to read raster data again and again
		np.save("/*/tsdata_samples_51_3.npy",combined_array)
	
	 	#save metadata
		with rasterio.open('/*/metadata.tif', 'w', **metadata) as dst:
	    		dst.update_tags(**metadata)
**Second, draw centroid of clustering, take a look at how kshape breaks down the data into categories**
  <p align='center'>
    <img width='400' height='300' src="https://i.postimg.cc/mrTnGqpB/kshape-cluster.png" hspace='10'>
    <img width='400' height='300' src="https://i.postimg.cc/7Y388gP1/kshape-mul-5.png" hsapce='10'>
  </p>
  <p align='center'>univariate in the left and mulitvariate in the right</p>

**Finally, check the result after kshape clustering(univariate for left and multivariate for right)🥳**
  <p align='center'>
    <img width='300' height='300' src="https://i.postimg.cc/13X8tx8r/2.png" hspace='10'>
    <img width='300' height='300' src="https://i.postimg.cc/T1hp091g/2.png" hsapce='10'>
  </p>

---
> [!NOTE]  
> These steps will cost lots of time, keep patient🛏️.
---

### *Evaluation*
---
**but now😕, another question is how to evaluate accuracy of kshape clustering❓**

*alas, tslearn does not seem to provide an evaluation method for KShape clustering. so we recommend you **modify the source code** to achieve it, more information click [here](https://blog.csdn.net/qq_37960007/article/details/107937212). let's see how to realize it. Officially, three similarity measures, dtw-dba, softdtw, and Euclidean distance, are provided in tslearn, but nothing about KShape clustering. Notice that source code, there `metric="precomputed"`, indicates that a user-defined distance metric is provided, which is good, and then the key to evaluating KShape using **`silhouette_score`** is here:*

*here is needed information about `silhouette_score`:*
it is an indicator that measures the tightness and separation of clustering results. It is based on the distance of each sample to other samples within the cluster to which it belongs and the distance to samples in the nearest cluster. The value range of the silhouette coefficient is between [-1, 1]. The closer the value is to 1, the better the sample clustering is. The closer the value is to -1, the worse the sample clustering is.

- Positive value: indicates that the samples are clustered well, and the intra-cluster distance is closer than the inter-cluster distance.
- Negative value: indicates that the samples are poorly clustered, and the distance within clusters is farther than the distance between clusters.
- Close to 0: It means that the distance between samples within and between clusters is similar, and the clustering result is unclear.


*here is e.g. of how to use silhouette_score calculation function `silhouette_score(cdist_dtw(X),y_pred,metric="precomputed")`, so you need to figure out what is `cdist_dtw(X)`❔. Actually realname of it, is measuring distance of time series, different from `Euclidean`, `dtw` and `softdtw`. so here is definition of SBD which is foundation of kshape.*

$SBD(\vec{x},\vec{y})=1-\max_{w}\left(\frac{CC_w(\vec{x},\vec{y})}{\sqrt{R_0(\vec{x},\vec{x})\cdot R_0(\vec{y},\vec{y})}}\right)$

so we add the `_get_norms` function, which is responsible for calculating the modulus length, and the `_my_cross_dist` function, which returns the distance matrix of the temporal collection X, i.e., the matrix formed by the distance between each element ti in X.

	def _get_norms(self,X):
	    norms = []
	    for x in X
	        result = numpy.linalg.norm(x)
		norms.append(result)
	    norms = numpy.array(norms)
	    return norms
	
	def _my_cross_dist(self,X):
	    norms = self._get_norms(X)
	    return 1.-cdist_normalized_cc(X,X,norms1=norms,norms2=norms,self_similarity=False)

*after this we test how it work*🤯

	import numpy as np
	from tslearn.generators import random_walks
	import tslearn.metrics as metrics
	from tslearn.clustering import silhouette_score
	 
	def test_kshape_silhouette_score():
	    seed = 0
	    num_cluster = 2
	    ks = KShape(n_clusters= num_cluster ,n_init= 5 ,verbose= True ,random_state=seed)
	    X = random_walks(n_ts=20, sz=16, d=1) * 10
	    #generate label radomly
	    y_pred = np.random.randint(2, size=20)
	    dists = ks._my_cross_dist(X)
	    #set diagonal elments to zero
	    np.fill_diagonal(dists,0)
	    #calculate silhouette score
	    score = silhouette_score(dists,y_pred,metric="precomputed")
	    print(score)
       >>>output:0.026772646

---
> [!NOTE]  
> These steps will cost amounts of RAM resource, make sure enough ram space👽.
---

**now, part of time series clustering is done. if you have interest about it, you also can try kmean➕dtw➕dba or kmeans➕softdtw. it's similar workflow. good luck😆**

## 🙉STEP 4: Time Series Classification

- ### *Prepare Training Data*
  ---
  
  generate training data according to paper[^2]:
  
  <p align="center">
    <img width='700' height='900' src="https://i.postimg.cc/J4wFjj8v/2.png">
  </p>

  - (1) The sample point was disturbed during the study period but not restored and failed to exhibit vegetation recovery. It is classified as DN and its NDVI values oscillate below the vegetation threshold after a sharp decrease (a).
  - (2) The sample point was disturbed during the study period and was still recovering at the end of the study period. It is classified as DR1 and its NDVI values decline sharply from a vegetated to a non- 
  vegetated state, then increase gradually to a state of less than-full recovery with an upward trend in last several years (b).
  - (3) The sample point was disturbed during the study period and exhibited vegetative stability but did not exhibit full recovery. It is classified as DR2 and its NDVI values decline sharply from a vegetated to a non-vegetated state, then increase gradually to a vegetated status with NDVI values lower than pre-disturbance but with little change in last several years (c).
  - (4) The sample point was disturbed during the study period, was restored to be stable, and exhibited full vegetation recovery. It is classified as DR3 and its NDVI values decline sharply from a vegetated to a non-vegetated state, then increase gradually to a vegetated status that is approaching or exceeding pre-disturbance with little change in last several years (d).
  - (5) The sample point was disturbed prior to the study period and exhibited recovery but did not exhibit a stable state. It is classified as BD1 and its initial NDVI values of pixels are lower than the 
  vegetation threshold but then increase gradually to a level that is no less than the vegetation threshold with an upward trend in last several years (e).
  - (6) The sample point was disturbed prior to the study period was restored to be stable. It is classified as BD2 and its initial NDVI values are lower than the vegetation threshold but then increase gradually 
  to a level that is no less than the vegetation threshold with little change in last several years (f).
  - (7) The sample points exhibited disturbances two or more times (g and h). They are classified as CD, meaning that they had complex disturbance histories.

  use code to generate relative pure training data:
    <p align="center">
    <img src="https://i.postimg.cc/PxbQnvLV/3.png">
    </p>
    
- ### *Training Model*
  ---
  
  *here we list standard procedures to train a model:*
  
  - choose model(like `minirocket`,`transformer`,`timesnet` and so on)
  - normalize raw data(if needed) and visualize it
  - build learner
  - find best learning_rate
  - training
  - optimize hyperparameter
  - training
  - validation accuracy
  - visualize result
  - apply it to rs image
  - evaluate accuracy
    
- ### *Classification Result*
  ---
  ***here are the classification result after training based on dataset we generated above.***

  <p align="center"><img src="https://i.postimg.cc/9fJwQsvP/2.png">
  </p>

  <p align='center'>model info</p>

<div align=center>

| Model            | training accuracy | validation accuracy | training loss | validaion loss | actual accuracy |
|------------------|-------------------|---------------------|---------------|----------------|-----------------|
| InceptionTime    | 0.939286          | 0.9393              | 0.173723      | 0.157058       | i'm tring       |
| BOSSVS           | 0.792857          | N/A                 | N/A           | N/A            | i'm tring       |
| TimeSeriesForest | 0.939285          | N/A                 | N/A           | N/A            | i'm tring       |
| CNN Classifier   | 0.921428          | N/A                 | N/A           | N/A            | i'm tring       |
| MiniRocket       | 0.939286          | N/A                 | 0.176533      | 0.186470       | i'm tring       |
| Tansformer       | 0.941667          | 0.9542              | 0.571105      | 0.612427       | i'm tring       |

</div>

*you can also draw picture to check predicted probal per ture class and confusion matrix after training, like this:*
  <p align="center">
<img width='300' height='300' src="https://i.postimg.cc/JzxBrKdW/2.png">
<img width='300' height='300' src="https://i.postimg.cc/P5wF0p6Z/2.png">
</p>

- ### 🖼️*Time Series to Image*
  - #### *convert time series to images classification*
  
  we use `tsai` a friendly ts learning lib to achieve it. there are several methods to convert ts data to images:😮
  
  - **TSToPlot**: Create matplotlib line plots
  - **TSToMat**: Create a matplotlib imshow figure
  - **TSToGADF**: Create an image based on Gramian angle difference field transformation
  - **TSToGASF**: Creates an image based on Gramian angle summation field transformation
  - **TSToMTF**: Create image based on Markov transfer field transformation
  - **TSToRP**: Create images based on recursive graph transformations
  
  you can realize it by following code:

		import numpy as np
		import pandas as pd
		from sklearn.model_selection import train_test_split
		import tsai
		
		# Read the CSV file
		df = pd.read_csv('/content/datasets.csv')
		
		# Convert the 'dataset' column to numpy arrays
		df['dataset'] = df['dataset'].apply(lambda x: np.array(eval(x)))
		
		# Convert DataFrame to numpy arrays
		data = np.stack(df['dataset'].values)
		
		# Reshape the data to a three-dimensional array
		data = data.reshape(data.shape[0], 1, data.shape[1])
		print(data.shape)
		# Extract labels
		labels = df['label'].values
		X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
		X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
		tfms = [None, Categorize()]
		bts = [[TSNormalize(), TSToPlot()],
		       [TSNormalize(), TSToMat(cmap='viridis')],
		       [TSNormalize(), TSToGADF(cmap='spring')],
		       [TSNormalize(), TSToGASF(cmap='summer')],
		       [TSNormalize(), TSToMTF(cmap='autumn')],
		       [TSNormalize(), TSToRP(cmap='winter')]]
		btns = ['Plot', 'Mat', 'GADF', 'GASF', 'MTF', 'RP']
		for i, (bt, btn) in enumerate(zip(bts, btns)):
		    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
		    dls = TSDataLoaders.from_dsets(dsets.train,
		                                   dsets.valid,
		                                   bs=[64, 128],
		                                   batch_tfms=bt,
		                                   shuffle=False)
		    xb, yb = dls.train.one_batch()
		    print(f'\n\ntfm: TSTo{btn} - batch shape: {xb.shape}')
		    xb[0].show()
		    plt.show()
    ts data will be displayed as image like this:
    <p align="center"><img src="https://i.postimg.cc/qBW895jY/1.png"></p>

   we choose use TsToGADF to create image after comparison. vasualize result after it:
   <p align="center"><img width='800' height='500' src="https://i.postimg.cc/nLZ8T2y9/1.png"></p>
   
    - #### *begin training*

<div align=center>
	
|epoch|train_loss|valid_loss|accuracy|time|
|:----|:----|:----|:----|:----|
|0|1.825284|1.758930|0.225000|00:09|
|1|1.620805|1.517212|0.496429|00:08|
|2|1.432231|1.289679|0.607143|00:08|
····
|97|0.000065|0.233326|0.946429|00:08|
|98|0.000066|0.248137|0.946429|00:08|
|99|0.000064|0.244490|0.950000|00:08|
 
</div>

<p align="center"><img src="https://i.postimg.cc/sDWQFfLd/1.png"></p>

- #### *apply model to RS image*
  
		from sklearn.preprocessing import MinMaxScaler
		
		#Initialize the MinMaxScaler
		scaler = MinMaxScaler()
		
		#Fit and transform the data
		norm_data = scaler.fit_transform(img_data)
		
		#Reshape the data back to standard shape
		norm_data = to3d(norm_data)
		print(norm_data.shape)
		-->(389360, 1, 51)
		#Unlabelled data
		test_ds = dls.dataset.add_test(norm_data)
		test_dl = valid_dl.new(test_ds)
		next(iter(test_dl))
		-->(TSImage(shape:torch.Size([64, 1, 224, 224])),)
		
		test_probas, test_targets, test_decoded = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None)
		test_decoded]

❌**failed becourse of RAM run out**

## Output Images
*on the way*😥

## TO DO
- [ ] classification result accuracy evaluation
- [x] add stable class re classify dataset(got bad result)
- [ ] remote sensing mapping
- [ ] D-InSAR detect surface deformation rely on change map(year of disturbance)


[^1]:Kennedy, Robert E., Yang, Zhiqiang, & Cohen, Warren B. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr - Temporal segmentation algorithms. Remote Sensing of Environment, 114, 2897-2910
[^2]:Zhen Yang, Jing Li, Carl E. Zipper, Yingying Shen, Hui Miao, Patricia F. Donovan, Identification of the disturbance and trajectory types in mining areas using multitemporal remote sensing images,Science of The Total Environment,Volume 644,2018,Pages 916-927,ISSN 0048-9697,https://doi.org/10.1016/j.scitotenv.2018.06.341.
[^3]:John Paparrizos Columbia University jopa@cs.columbia.edu Luis Gravano Columbia University gravano@cs.columbia.edu k-Shape: Efficient and Accurate Clustering of Time Series. 
