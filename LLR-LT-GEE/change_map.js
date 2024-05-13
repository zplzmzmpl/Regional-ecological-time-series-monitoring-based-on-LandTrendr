//######################################################################################################## 
//#                                                                                                    #\\
//#                                LANDTRENDR GREATEST GROWTH MAPPING                                  #\\
//#                                                                                                    #\\
//########################################################################################################


// date: 2018-10-07
// author: Justin Braaten | jstnbraaten@gmail.com
//         Zhiqiang Yang  | zhiqiang.yang@oregonstate.edu
//         Robert Kennedy | rkennedy@coas.oregonstate.edu
// parameter definitions: https://emapr.github.io/LT-GEE/api.html#getchangemap
// website: https://github.com/eMapR/LT-GEE
// notes: 
//   - you must add the LT-GEE API to your GEE account to run this script. 
//     Visit this URL to add it:
//     https://code.earthengine.google.com/?accept_repo=users/emaprlab/public
//   - use this app to help parameterize: 
//     https://emaprlab.users.earthengine.app/view/lt-gee-change-mapper


//##########################################################################################
// START INPUTS
//##########################################################################################

// define collection parameters
var startYear = 1972;
var endYear = 2024;
var startDay = '06-20';
var endDay = '09-20';

//var aoi = ee.Geometry.Point(99.16355688947924, 38.12665635953645);
//var aoi = ee.Geometry.Point([-110.5596330453144, 44.23560408150123]);
//var aoi = ee.Geometry.Point(-112.14937319475919,40.532786446078745);

var aoi = ee.Geometry.Polygon([[
  [-112.22631597174929,40.482462430709305],
  [-112.05946111335085,40.482462430709305],
  [-112.05946111335085,40.61081684118072],
  [-112.22631597174929,40.61081684118072],
  [-112.22631597174929,40.482462430709305]]]);

// Map.centerObject(aoi);
// Map.addLayer(aoi)

var index = 'NDVI';
var maskThese = ['cloud', 'shadow', 'snow', 'water'];

// define landtrendr parameters
var runParams = { 
  maxSegments:            6,
  spikeThreshold:         0.9,
  vertexCountOvershoot:   3,
  preventOneYearRecovery: true,
  recoveryThreshold:      0.25,
  pvalThreshold:          0.05,
  bestModelProportion:    0.75,
  minObservationsNeeded:  6
};

// define change parameters
var changeParams = {
  delta:  'gain',
  sort:   'greatest',
  year:   {checked:true, start:startYear, end:endYear},
  mag:    {checked:true,  value:300,  operator:'>'},
  dur:    {checked:false,  value:4,    operator:'<'},
  preval: {checked:false,  value:300,  operator:'>'},
  mmu:    {checked:false,  value:11},
};

//##########################################################################################
// END INPUTS
//##########################################################################################

// load the LandTrendr.js module
var ltgee = require('users/emaprlab/public:Modules/LandTrendr.js'); 

// add index to changeParams object
changeParams.index = index;

// run landtrendr
var lt = ltgee.runLT(startYear, endYear, startDay, endDay, aoi, index, [], runParams, maskThese);

// get the change map layers
var changeImg = ltgee.getChangeMap(lt, changeParams);

// set visualization dictionaries
var palette = ['#9400D3', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000'];
var yodVizParms = {
  min: startYear,
  max: endYear,
  palette: palette
};

var magVizParms = {
  min: 200,
  max: 800,
  palette: palette
};

// display the change attribute map - note that there are other layers - print changeImg to console to see all
Map.centerObject(aoi, 11);
Map.addLayer(changeImg.select(['mag']), magVizParms, 'Magnitude of Change');
Map.addLayer(changeImg.select(['yod']), yodVizParms, 'Year of Detection');

/*
变化事件检测年份：（'yod'年份）
变化事件的幅度：（'mag'变化事件频谱增量的绝对值）
变化事件的持续时间：（'dur'年）
变化前事件频谱值：（'preval'频谱值）
事件的光谱变化率 'rate'( mag/dur)
DSNR 'dsnr' ( mag/fit rmse) 乘以 100 以保留 Int16 数据的两位小数精度。
*/

var castImage = changeImg.select(['yod']).toDouble();
var exportImage = changeImg.addBands(castImage, ['yod'], true);
print(exportImage);
Export.image.toDrive({
image:exportImage,
scale:30,
crs:'EPSG:3857',
region:aoi,
maxPixels:1e13});

var region = aoi.buffer(10000).bounds();
var exportImg = changeImg.clip(region).unmask(0).double();
var foldername = 'slc';
var crs = 'EPSG:4326';
Export.image.toDrive({
  image: exportImg.select('yod'), 
  description: foldername + '_yod_map', 
  folder:foldername,
  fileNamePrefix: foldername + '_yod_map', 
  region: region, 
  scale: 30, 
  crs: crs, 
  //maxPixels: 1e13
});
Export.image.toDrive({
  image: exportImg.select('mag'), 
  description: foldername + '_mag_map', 
  folder:foldername,
  fileNamePrefix: foldername + '_mag_map', 
  region: region, 
  scale: 30, 
  crs: crs, 
  //maxPixels: 1e13
});
Export.image.toDrive({
  image: exportImg.select('dur'), 
  description: foldername + '_dur_map', 
  folder:foldername,
  fileNamePrefix: foldername + '_dur_map', 
  region: region, 
  scale: 30, 
  crs: crs, 
  //maxPixels: 1e13
});
Export.image.toDrive({
  image: exportImg.select('preval'), 
  description: foldername + '_preval_map', 
  folder:foldername,
  fileNamePrefix: foldername + '_preval_map', 
  region: region, 
  scale: 30, 
  crs: crs, 
  //maxPixels: 1e13
});
Export.image.toDrive({
  image: exportImg.select('rate'), 
  description: foldername + '_rate_map', 
  folder:foldername,
  fileNamePrefix: foldername + '_rate_map', 
  region: region, 
  scale: 30, 
  crs: crs, 
  //maxPixels: 1e13
});
Export.image.toDrive({
  image: exportImg.select('dsnr'), 
  description: foldername + '_dsnr_map', 
  folder:foldername,
  fileNamePrefix: foldername + '_dsnr_map', 
  region: region, 
  scale: 30, 
  crs: crs, 
  //maxPixels: 1e13
});