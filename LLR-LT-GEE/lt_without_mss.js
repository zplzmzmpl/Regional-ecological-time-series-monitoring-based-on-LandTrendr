var msslib = require('users/jstnbraaten/modules:msslib/msslib.js');
var ltgee = require('users/emaprlab/public:Modules/LandTrendr.js'); 
var animation = require('users/gena/packages:animation');

function getCfmask(img){
  return img.select('QA_PIXEL').bitwiseAnd(parseInt('11111',2)).eq(0);
}

// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBand, null, true);
}

function scaleMask(img){
  function getFactorImg(factorNames){
    var factorList = img.toDictionary().select(factorNames).values();
    return ee.Image.constant(factorList);
  }
  var scaleImg=getFactorImg(['REFLECTANCE_MULT_BAND_.']);
  var offsetImg = getFactorImg(['REFLECTANCE_ADD_BAND_.']);
  var scaled = (img.select('SR_B.').multiply(scaleImg).add(offsetImg)
                                    .multiply(10000).round().int16());
  return (img.addBands(scaled,null,true)
              .select('SR_B.'))
              .updateMask(getCfmask(img));
}
  

// Define function to prepare OLI images.
function prepOli(img) {
  var orig = img;
  img = scaleMask(img);
  img = renameOli(img);
  img = tmAddIndices(img);
  //img = applyCfmask(img);
  return ee.Image(img.copyProperties(orig, orig.propertyNames()));
}

// Define function to prepare ETM+ images.
function prepTm(img) {
  var orig = img;
  img = scaleMask(img);
  img = renameTm(img);
  img = tmAddIndices(img);
  //img = applyCfmask(img);
  return ee.Image(img.copyProperties(orig, orig.propertyNames()));
}
//exports.prepTm = prepTm;

// Function to get and rename bands of interest from OLI.
function renameOli(img) {
  return img.select(
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
      ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
}

// Function to get and rename bands of interest from ETM+.
function renameTm(img) {
  return img.select(
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
      ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
}

function tmAddIndices(img) {
  var b = ee.Image(img).select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
  var brt_coeffs = ee.Image.constant([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303]);
  var grn_coeffs = ee.Image.constant([-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446]);
  var brightness = b.multiply(brt_coeffs).reduce(ee.Reducer.sum()).round().toShort();
  var greenness = b.multiply(grn_coeffs).reduce(ee.Reducer.sum()).round().toShort();
  var angle = (greenness.divide(brightness)).atan().multiply(180 / Math.PI).multiply(100).round().toShort();
  var ndvi = img.normalizedDifference(['nir', 'red']).rename('ndvi').multiply(1000).round().toShort();
  var tc = ee.Image.cat(ndvi, brightness, greenness, angle).rename(['ndvi', 'tcb', 'tcg', 'tca']);
  return img.addBands(tc);
}

function gatherTmCol(params) {
  var granuleGeom = msslib.getWrs1GranuleGeom(params.wrs1);
  var aoi = ee.Feature(granuleGeom.get('granule')).geometry();
  var dateFilter = ee.Filter.calendarRange(params.doyRange[0], params.doyRange[1], 'day_of_year');
  var startDate = ee.Date.fromYMD(params.yearRange[0], 1, 1);
  var endDate = startDate.advance(1, 'year');
  var oli2Col = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterBounds(aoi).filterDate(startDate, endDate).filter(dateFilter).map(prepOli);
  var oliCol = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterBounds(aoi).filterDate(startDate, endDate).filter(dateFilter).map(prepOli);
  var etmCol = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
        .filterBounds(aoi).filterDate(startDate, endDate).filter(dateFilter).map(prepTm);
  var tm5Col = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
        .filterBounds(aoi).filterDate(startDate, endDate).filter(dateFilter).map(prepTm);
  var tm4Col = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
        .filterBounds(aoi).filterDate(startDate, endDate).filter(dateFilter).map(prepTm);
  return ee.ImageCollection(ee.FeatureCollection([tm4Col, tm5Col, etmCol, oliCol, oli2Col]).flatten());
}
//exports.gatherTmCol = gatherTmCol;

function getMedoid(col, bands, parallelScale) {
  col = col.select(bands);
  var median = col.reduce(ee.Reducer.median(),parallelScale);
  
  var difFromMedian = col.map(function(img) {
    var dif = ee.Image(img).subtract(median).pow(ee.Image.constant(2));
    return dif.reduce(ee.Reducer.sum())
      .addBands(img);
  });
  
  var bandNames = difFromMedian.first().bandNames();
  var len = bandNames.length();
  var bandsPos = ee.List.sequence(1, len.subtract(1));
  var bandNamesSub = bandNames.slice(1);
  return difFromMedian.reduce(ee.Reducer.min(len),parallelScale).select(bandsPos, bandNamesSub);
}
//exports.getMedoid = getMedoid;

function getColForLandTrendrOnTheFly(params) { // Does not rely on WRS1_to_TM assets
  //var mssCol = getFinalCorrectedMssCol(params);

  var tmCol = ee.ImageCollection([]);
  for(var y=1996; y<=2023; y++) {
    params.yearRange = [y, y];
    var thisYearCol = getMedoid(gatherTmCol(params), ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'tcb', 'tcg', 'tca'],8)
      .set('system:time_start', ee.Date.fromYMD(y, 1 ,1).millis());
    tmCol = tmCol.merge(ee.ImageCollection(thisYearCol.toShort()));
  }
  
  var combinedCol = tmCol.map(function(img) {
    return img.select('ndvi').multiply(-1).toShort().rename('LTndvi')  // TODO: move this into the run landtrandr function - get the fitting index from params
      .addBands(img.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'tcb', 'tcg', 'tca']))  // TODO: move this into the run landtrandr function - what indices should be FTV
      .copyProperties(img,img.propertyNames());
      // .set('system:time_start', img.get('system:time_start'))
      // .set('system:index',img.get('system:index'))
      // .set('SPACECRAFT_ID',img.get('SPACECRAFT_ID'));
  }).sort('system:time_start');

  return combinedCol;
}
//exports.getColForLandTrendrOnTheFly = getColForLandTrendrOnTheFly;

var PROJ_PATH = 'users/ee-opppmqqqo/LandsatLinkr';   // Must be the same path used to create the asset folder - cannot contain / at end - check for this in the code.
var WRS_1_GRANULE = '144033';
var CRS = 'EPSG:4326';

var DOY_RANGE = [160, 254];
var MAX_CLOUD = 50;
var MAX_GEOM_RMSE = 0.5;

var params = {
  maxRmseVerify: MAX_GEOM_RMSE,
  maxCloudCover: MAX_CLOUD,
  doyRange: DOY_RANGE,
  wrs1: WRS_1_GRANULE,
  crs: CRS,
  baseDir: PROJ_PATH + '/' + WRS_1_GRANULE
};

var ltCol = getColForLandTrendrOnTheFly(params);
//Map.setCenter(99.15,38.12, 10);
//Map.addLayer(ltCol.first(), {bands: ['red', 'green', 'blue'], min:0.0, max:1}, 'image');
//print(ltCol.first());

// var img = ltCol.first();

// Export.image.toDrive({
//   image:img.select('ndvi'),
//   scale:30,
//   crs:'EPSG:4326',
//   region:geometry,
//   maxPixels:1e13
// });

// var batch = require('users/fitoprincipe/geetools:batch');
// //COLLECTION
// batch.Download.ImageCollection.toDrive(ltCol,"144033", {
// scale: 30,
// crs:'EPSG:4326',
// region: geometry,
// type:"float" });

function runLandTrendrMss2Tm(params) {
  // var ltCol = getColForLandTrendrOnTheFly(params); // alternative: getColForLandTrendrOnTheFly(params)
  var lt = ee.Algorithms.TemporalSegmentation.LandTrendr({
    timeSeries: ltCol,
    maxSegments: 10,
    spikeThreshold: 0.7,
    vertexCountOvershoot: 3,
    preventOneYearRecovery: true,
    recoveryThreshold: 0.5,
    pvalThreshold: 0.05,
    bestModelProportion: 0.75,
    minObservationsNeeded: 6
  });
  return lt;
}
//exports.runLandTrendrMss2Tm = runLandTrendrMss2Tm;

var lt = runLandTrendrMss2Tm(params);

function animateCollection(col) {
  var rgbviz = {
    bands: ['red','green','blue'],
    min: 100,
    max: 2000,
    gamma: [1.2]
  };
  // TODO: add year of image as label in animation
  // col = col.map(function(img) {
  //   img = img.set({label: ee.String(img.get('system:id'))})
  //   return img
  // })
  Map.centerObject(col.first(), 8);
  // run the animation
  animation.animate(col, {
    vis: rgbviz,
    timeStep: 1500,
    maxFrames: col.size()
  });
}

//animateCollection(ltCol);

function displayGreatestDisturbance(lt, params) {
  var granuleGeom = ee.Feature(msslib.getWrs1GranuleGeom(params.wrs1)
    .get('granule')).geometry();
  
  var currentYear = new Date().getFullYear();  // TODO: make sure there is not a better way to get year from image metadata eg
  var changeParams = { // TODO: allow a person to override these params
    delta:  'loss',
    sort:   'greatest',
    year:   {checked:true, start:1972, end:currentYear},  // TODO: make sure there is not a better way to get years from image metadata eg
    mag:    {checked:true, value:200,  operator:'>'},
    dur:    {checked:true, value:4,    operator:'<'},
    preval: {checked:true, value:300,  operator:'>'},
    mmu:    {checked:true, value:11},
  };
  // Note: add index to changeParams object this is hard coded to NDVI because currently that is the only option.
  changeParams.index = 'NDVI';
  var changeImg = ltgee.getChangeMap(lt, changeParams);
  var palette = ['#9400D3', '#4B0082', '#0000FF', '#00FF00',
                  '#FFFF00', '#FF7F00', '#FF0000'];
  var yodVizParms = {
    min: 1996, // TODO: make sure there is not a better way to get year from image metadata eg
    max: currentYear, // TODO: make sure there is not a better way to get year from image metadata eg
    palette: palette
  };
  var magVizParms = {
    min: 200,
    max: 800,
    palette: palette
  };
  // Assuming 'image' is your image
  var castImage = changeImg.select(['yod']).toDouble();
  var exportImage = changeImg.addBands(castImage, ['yod'], true);
  print(exportImage);
  Export.image.toDrive({
  image:exportImage,
  scale:30,
  crs:'EPSG:4326',
  region:geometry,
  maxPixels:1e13});
  Map.centerObject(geometry, 12);  // Zoom in pretty far otherwise the mmu filter is going to take forever (probably crash)
  // display two change attributes to map
  Map.addLayer(changeImg.select(['mag']), magVizParms, 'Magnitude of Change');
  Map.addLayer(changeImg.select(['yod']), yodVizParms, 'Year of Detection');
}

displayGreatestDisturbance(lt,params);


Export.image.toDrive({
  image:landTrendr,
  scale:30,
  crs:'EPSG:4326',
  region:geometry,
  maxPixels:1e13
});
