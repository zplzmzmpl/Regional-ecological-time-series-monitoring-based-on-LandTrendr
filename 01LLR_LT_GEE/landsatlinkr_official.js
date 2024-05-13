// #############################################################################
// #############################################################################
// #############################################################################

/**
 * 1. View WRS-1 granules - figure out what WRS-1 granule to process
 * -- Make a processing dir: https://gist.github.com/jdbcode/36f5a04329d5d85c43c0408176c51e6d
 * 2. Create MSS WRS-1 reference image - for MSS WRS1 to MSS WRS2 harmonization
 * 3. View WRS-1 collection - identify bad MSS images
 * 4. Prepare MSS WRS-1 images
 * 5. Get TM-to-MSS correction coefficients
 * 6. Export MSS-to-TM corrected images
 * 7. Inspect the full time series collection - explore time series via animation and inspector tool to check for noise
 * 8. Run LandTrendr and display the fitted collection on the map
 * 9. Display the year and magnitude of the greatest disturbance during the time series
 */

var LLR_STEP = 1;

// #############################################################################

var PROJ_PATH = 'projects/your_gee_account/assets/LandsatLinkr';   // Must be the same path used to create the asset folder - cannot contain / at end - check for this in the code.
var WRS_1_GRANULE = '041032';
var CRS = 'EPSG:3857';
var aoi = ee.Geometry.Rectangle({
  coords: [[-122.6,37.8],[-122.0,36.8]],
  geodesic:false
});
var DOY_RANGE = [160, 254];
var MAX_CLOUD = 50;
var MAX_GEOM_RMSE = 0.5;

var EXCLUDE_IDS = [
    'LM10470341972208GDS03',
    'LM10470341972244AAA02',
    'LM10470341974161AAA02',
    'LM10470341974197AAA04',
    'LM20470341975201GDS03',
    'LM10470341976205AAA05',
    'LM10470341976223GDS03',
    'LM20470341976250GDS03',
    'LM20470341977190GDS04',
    'LM30470341978194GDS03',
    'LM20470341977226AAA05',
    'LM20470341977244GDS03',
    'LM20470341978185AAA02',
    'LM20470341978203AAA02',
    'LM30470341978212GDS03',
    'LM20470341978221AAA02',
    'LM20470341978239AAA02',
    'LM30470341979207AAA02',
    'LM20470341979216XXX01',
    'LM20470341980211AAA10',
    'LM30470341980220AAA03',
    'LM20470341980229AAA06',
    'LM20470341981205AAA03',
    'LM20470341981223AAA03',
    'LM30470341982173AAA08',
    'LM30470341982227AAA03'
];

// #############################################################################
// #############################################################################
// #############################################################################

var params = {
  maxRmseVerify: MAX_GEOM_RMSE,
  maxCloudCover: MAX_CLOUD,
  doyRange: DOY_RANGE,
  wrs1: WRS_1_GRANULE,
  crs: CRS,
  excludeIds: EXCLUDE_IDS,
  baseDir: PROJ_PATH + '/' + WRS_1_GRANULE
};

var llr = require('users/jstnbraaten/modules:landsatlinkr/landsatlinkr.js');
switch (LLR_STEP) {
  case 1:
    llr.wrs1GranuleSelector();
    break;
  case 2:
    llr.exportMssRefImg(params);
    break;
  case 3:
    llr.viewWrs1Col(params);
    break;
  case 4:
    llr.processMssWrs1Imgs(params);
    break;
  case 5:
    llr.exportMss2TmCoefCol(params);
    break;
  case 6:
    llr.exportFinalCorrectedMssCol(params);
    break;
  case 7:
    var col = llr.getColForLandTrendrFromAsset(params);
    llr.displayCollection(col);
    //llr.animateCollection(col);
    var args = {
      crs: 'EPSG:3857',
      dimensions: '400',
      region: aoi,
      min: -500,
      max: 1000,
      palette: ['white', 'blanchedalmond', 'green', 'green'],
      framesPerSecond: 3,
    };
    var thumb = ui.Thumbnail({
      image: col.select('ndvi'),
      params: args,
      style: {
        position: 'bottom-right',
        width: '400px'
      }
    });
    Map.add(thumb);
    Map.centerObject(aoi,8);
    break;
  case 8:
    var lt = llr.runLandTrendrMss2Tm(params);
    //llr.displayFittedCollection() not built yet
    break;
  case 9:
    var lt = llr.runLandTrendrMss2Tm(params);
    llr.displayGreatestDisturbance(lt, params);
    break;
}