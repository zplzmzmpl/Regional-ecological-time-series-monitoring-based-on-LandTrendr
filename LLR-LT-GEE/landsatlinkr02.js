var featureValues = [5];
// define your study area
// var featureCol = 'users/emaprlab/SERVIR/ee_processing_tiles_2x15';
var featureKey = 'eetile2x15';
var startYear = 1972; // what year do you want to start the time series 
var endYear = 2023; // what year do you want to end the time series
var startDay = ['06-15']; // what is the beginning of date filter | month-day
var endDay =   ['09-15']; // what is the end of date filter | month-day
var indexList = [['NDVI', -1, true]]; // The indices to segment on and the invert coefficient
var ftvList = ['NDVI','TCG','TCW']; // List of images to export
var vertList = [];// 'YRS', 'SRC', 'FIT'
var mosaicType = "medoid"; // how to make annual mosaic - options: "medoid", "targetDay"
var targetDay = null ; // if running "targetDay" mosaic, what day of year should be the target
var outProj = 'EPSG:3857'; // what should the output projection be? 'EPSG:5070' is North American Albers
var gDriveFolder = featureKey + '_' + indexList[0][0]; // what is the name of the Google Drive folder that you want the outputs placed in
var affine = [30.0, 0, 15.0, 0, -30.0, 15.0];
var aoiBuffer = 1000;

// define the segmentation parameters - see paper (NEED CITATION)
var run_params = {
  maxSegments: 10,
  spikeThreshold: 0.9,
  vertexCountOvershoot: 3,
  preventOneYearRecovery: true,
  recoveryThreshold: 0.75,
  pvalThreshold: 0.05,
  bestModelProportion: 0.75,
  minObservationsNeeded: 6
};

//Map.addLayer(featureCol)

// ###############################################################################
// get geometry stuff


// #######################################################################################
// ###### ANNUAL SR TIME SERIES STACK BUILDING FUNCTIONS #################################
// #######################################################################################

// ------ DEFINE L8 to L7 ALIGN FUNCTION ---------

// slope and intercept citation: Roy, D.P., Kovalskyy, V., Zhang, H.K., Vermote, E.F., Yan, L., Kumar, S.S, Egorov, A., 2016, Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity, Remote Sensing of Environment, 185, 57-70.(http://dx.doi.org/10.1016/j.rse.2015.12.024); Table 2 - reduced major axis (RMA) regression coefficients
var harmonizationRoy = function(oli) {
  var slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]); // create an image of slopes per band for L8 TO L7 regression line - David Roy
  var itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]); // create an image of y-intercepts per band for L8 TO L7 regression line - David Roy
  var y = oli.select(['B2','B3','B4','B5','B6','B7'],['B1', 'B2', 'B3', 'B4', 'B5', 'B7']) // select OLI bands 2-7 and rename them to match L7 band names
             .resample('bicubic') // ...resample the L8 bands using bicubic
             .subtract(itcp.multiply(10000)).divide(slopes) // ...multiply the y-intercept bands by 10000 to match the scale of the L7 bands then apply the line equation - subtract the intercept and divide by the slope
             .set('system:time_start', oli.get('system:time_start')); // ...set the output system:time_start metadata to the input image time_start otherwise it is null
  return y.toShort(); // set image to signed 16-bit integer 
};


// harmonize tm and etm+ to oli
var harmonizationRoy2OLI = function(tm) {
  var slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]);        // RMA - create an image of slopes per band for L7 TO L8 regression line - David Roy
  var itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]);     // RMA - create an image of y-intercepts per band for L7 TO L8 regression line - David Roy
   var y = tm.select(['B1','B2','B3','B4','B5','B7'])                                  // select TM bands 1-5,7 
             .resample('bicubic')                                                          // ...resample the TM bands using bicubic
             .multiply(slopes).add(itcp.multiply(10000))                                  // ...multiply the y-intercept bands by 10000 to match the scale of the L8 bands then apply the line equation - scale by the slope and add the intercept 
             .set('system:time_start', tm.get('system:time_start'));                      // ...set the output system:time_start metadata to the input image time_start otherwise it is null
  return y.toShort();                                                                    // return the image as short to match the type of the other data
};


// ------ DEFINE FUNCTION TO RETRIEVE A SENSOR SR COLLECTION -----------------------------
 
var getSRcollection = function(year, startDay, endDay, sensor, box) {
  var srCollection = ee.ImageCollection('LANDSAT/'+ sensor + '/C01/T1_SR') // get surface reflectance images
                       .filterBounds(box) // filter them by a bounding box
                       .filterDate(year+'-'+startDay, year+'-'+endDay); // filter their dates from June 1st - Sept 30th
             
  srCollection = srCollection.map(function(img) {
    var dat = ee.Image(
      ee.Algorithms.If(
        sensor != 'LC08',
        harmonizationRoy2OLI(img.unmask()),
        img.select(['B2','B3','B4','B5','B6','B7'],['B1', 'B2', 'B3', 'B4', 'B5', 'B7'])
        //sensor == 'LC08', // condition - if image is OLI
        //harmonizationRoy(img.unmask()), // true - then apply the L8 TO L7 alignment function after unmasking pixels that were previosuly masked (why/when are pixels masked)
        //img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'])
           .unmask() // false - else select out the reflectance bands from the non-OLI image and unmask any previous pixels 
           .resample('bicubic') // ...resample by bicubic 
           .set('system:time_start', img.get('system:time_start')) // ...set the output system:time_start metadata to the input image time_start otherwise it is null
      )
    );

    //var cloudMask = img.select('pixel_qa').bitCount().lte(2); // select out the fmask layer and create a 0/1 mask - set 0,1 to 1, all else to 0; 0=clear; 1=water; 2=shadow; 3=snow; 4=cloud
    
    var qa = img.select('pixel_qa');
    var mask = qa.bitwiseAnd(8).eq(0).and( // Shadow
               qa.bitwiseAnd(16).eq(0)).and( // Snow
               qa.bitwiseAnd(32).eq(0)); // Clouds
    
    var datMasked = dat.mask(mask); //apply the mask - 0's in mask will be excluded from computation and set to opacity=0 in display
    return datMasked; // return
  });

  return srCollection; // return 
};



// ------ DEFINE FUNCTION TO COMBINE LT5, LE7, & LC8 COLLECTIONS -------------------------

var getCombinedSRcollection = function(year, startDay, endDay, box) {
    var lt5 = getSRcollection(year, startDay, endDay, 'LT05', box); // get TM collection for a given year and bounding area
    var le7 = getSRcollection(year, startDay, endDay, 'LE07', box); // get ETM+ collection for a given year and bounding area
    var lc8 = getSRcollection(year, startDay, endDay, 'LC08', box); // get OLI collection for a given year and bounding area
    return  ee.ImageCollection(lt5.merge(le7).merge(lc8)); // merge the individual sensor collections into one imageCollection object
};



// ------ DEFINE FUNCTION TO REDUCE COLLECTION TO SINGLE IMAGE BY MEDOID -----------------

// medoid composite with equal weight among indices
// Medoids are representative objects of a data set or a cluster with a data set whose average dissimilarity to all the objects in the cluster is minimal. Medoids are similar in concept to means or centroids, but medoids are always members of the data set.
var medoidMosaic = function(inCollection, dummyCollection) {
  
  var imageCount = inCollection.toList(1).length();
  var finalCollection = ee.ImageCollection(ee.Algorithms.If(imageCount.gt(0), inCollection, dummyCollection));
  
  var median = ee.ImageCollection(finalCollection).median(); // calculate the median of the annual image collection - returns a single 6 band image - the collection median per band
  
  var medoid = finalCollection.map(function(img) {
    var diff = ee.Image(img).subtract(median).pow(ee.Image.constant(2)); // get the difference between each image/band and the corresponding band median and take to power of 2 to make negatives positive and make greater differences weight more
    return diff.reduce('sum').addBands(img); //per image in collection, sum the powered difference across the bands - set this as the first band add the SR bands to it - now a 7 band image collection
  });
  
  return ee.ImageCollection(medoid).reduce(ee.Reducer.min(7)).select([1,2,3,4,5,6], ['B1','B2','B3','B4','B5','B7']); // find the powered difference that is the least - what image object is the closest to the median of teh collection - and then subset the SR bands and name them - leave behind the powered difference band
};



// ------ DEFINE FUNCTION TO REDUCE COLLECTION TO SINGLE IMAGE BY DISTANCE FROM MEDIAN DAY ---
var targetDayMoasic = function(inCollection, targetDay){
  var inCollectionDelta = inCollection.map(function(image) {
    var day = ee.Date(image.get('system:time_start')).getRelative('day', 'year');
    var delta = image.select(null).addBands(day.subtract(targetDay).abs().multiply(-1)).int16();
    return delta.select([0], ['delta']).addBands(image);
  });

  return inCollectionDelta.qualityMosaic('delta')
                          .select([1,2,3,4,5,6]);
};




// ------ DEFINE FUNCTION TO APPLY A MOSAIC FUNCTION TO A COLLECTION -------------------------------------------

var buildMosaic = function(year, startDay, endDay, box, mosaicType, targetDay, dummyCollection) {
  var tmp; // create a temp variable to hold the upcoming annual mosiac
  var collection = getCombinedSRcollection(year, startDay, endDay, box); // get the SR collection
  if(mosaicType == "medoid"){tmp = medoidMosaic(collection, dummyCollection);} // reduce the collection to single image per year by medoid 
  else if(mosaicType == "targetDay"){tmp = targetDayMoasic(collection, targetDay);} // reduce the collection to single image per year by medoid
  var img = tmp.set('system:time_start', (new Date(year,8,1)).valueOf()); // add the year to each medoid image
  return ee.Image(img); // return as image object
};



// ------ DEFINE FUNCTION TO BUILD ANNUAL MOSAIC COLLECTION ------------------------------
var buildMosaicCollection = function(startYear, endYear, startDay, endDay, box, mosaicType, targetDay, dummyCollection) {  //Null
  var imgs = []; //create empty array to fill
  for (var i = startYear; i <= endYear; i++) { // for each year from hard defined start to end build medoid composite and then add to empty img array
    var tmp = buildMosaic(i, startDay, endDay, box, mosaicType, targetDay, dummyCollection); // build the medoid mosaic for a given year
    imgs = imgs.concat(tmp.set('system:time_start', (new Date(i,8,1)).valueOf())); // concatenate the annual image medoid to the collection (img) and set the date of the image - hardwired to the year that is being worked on for Aug 1st
  }
  return ee.ImageCollection(imgs); //return the array img array as an image collection
};

// #######################################################################################
// #######################################################################################
// #######################################################################################





// #######################################################################################
// ###### INDEX CALCULATION FUNCTIONS ####################################################
// #######################################################################################

// TASSELLED CAP
var tcTransform = function(img){ 
  var b = ee.Image(img).select(["B1", "B2", "B3", "B4", "B5", "B7"]); // select the image bands
  var brt_coeffs = ee.Image.constant([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303]); // set brt coeffs - make an image object from a list of values - each of list element represents a band
  var grn_coeffs = ee.Image.constant([-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446]); // set grn coeffs - make an image object from a list of values - each of list element represents a band
  var wet_coeffs = ee.Image.constant([0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]); // set wet coeffs - make an image object from a list of values - each of list element represents a band
  
  var sum = ee.Reducer.sum(); // create a sum reducer to be applyed in the next steps of summing the TC-coef-weighted bands
  var brightness = b.multiply(brt_coeffs).reduce(sum); // multiply the image bands by the brt coef and then sum the bands
  var greenness = b.multiply(grn_coeffs).reduce(sum); // multiply the image bands by the grn coef and then sum the bands
  var wetness = b.multiply(wet_coeffs).reduce(sum); // multiply the image bands by the wet coef and then sum the bands
  var angle = (greenness.divide(brightness)).atan().multiply(180/Math.PI).multiply(100);
  var tc = brightness.addBands(greenness)
                     .addBands(wetness)
                     .addBands(angle)
                     .select([0,1,2,3], ['TCB','TCG','TCW','TCA']) //stack TCG and TCW behind TCB with .addBands, use select() to name the bands
                     .set('system:time_start', img.get('system:time_start'));
  return tc;
};

// NBR
var nbrTransform = function(img) {
    var nbr = img.normalizedDifference(['B4', 'B7']) // calculate normalized difference of B4 and B7. orig was flipped: ['B7', 'B4']
                 .multiply(1000) // scale results by 1000
                 .select([0], ['NBR']) // name the band
                 .set('system:time_start', img.get('system:time_start'));
    return nbr;
};

// NDVI
var ndviTransform = function(img){ 
  var ndvi = img.normalizedDifference(['B4', 'B3']) // calculate normalized dif between band 4 and band 3 (B4-B3/B4_B3)
                .multiply(1000) // scale results by 1000
                .select([0], ['NDVI']) // name the band
                .set('system:time_start', img.get('system:time_start'));
  return ndvi;
};
                
// NDSI
var ndsiTransform = function(img){ 
  var ndsi = img.normalizedDifference(['B2', 'B5']) // calculate normalized dif between band 4 and band 3 (B4-B3/B4_B3)
                .multiply(1000) // scale results by 1000
                .select([0], ['NDSI']) // name the band
                .set('system:time_start', img.get('system:time_start'));
  return ndsi;
};

// NDMI
var ndmiTransform = function(img) {
    var ndmi = img.normalizedDifference(['B4', 'B5']) // calculate normalized difference of B4 and B7. orig was flipped: ['B7', 'B4']
                 .multiply(1000) // scale results by 1000
                 .select([0], ['NDMI']) // name the band
                 .set('system:time_start', img.get('system:time_start'));
    return ndmi;
};




// CALCULATE A GIVEN INDEX
var calcIndex = function(img, index, flip){
  // make sure index string in upper case
  index = index.toUpperCase();
  
  // figure out if we need to calc tc
  var tcList = ['TCB', 'TCG', 'TCW', 'TCA'];
  var doTC = tcList.indexOf(index);
  if(doTC >= 0){
    var tc = tcTransform(img);
  }
  
  // need to flip some indices if this is intended for segmentation
  var indexFlip = 1;
  if(flip == 1){
    indexFlip = -1;
  }
  
  // need to cast raw bands to float to make sure that we don't get errors regarding incompatible bands
  // ...derivations are already float because of division or multiplying by decimal
  var indexImg;
  switch (index){
    case 'B1':
      indexImg = img.select(['B1']).float();//.multiply(indexFlip);
      break;
    case 'B2':
      indexImg = img.select(['B2']).float();//.multiply(indexFlip);
      break;
    case 'B3':
      indexImg = img.select(['B3']).float();//.multiply(indexFlip);
      break;
    case 'B4':
      indexImg = img.select(['B4']).multiply(indexFlip).float();
      break;
    case 'B5':
      indexImg = img.select(['B5']).float();//.multiply(indexFlip);
      break;
    case 'B7':
      indexImg = img.select(['B7']).float();//.multiply(indexFlip);
      break;
    case 'NBR':
      indexImg = nbrTransform(img).multiply(indexFlip);
      break;
    case 'NDMI':
      indexImg = ndmiTransform(img).multiply(indexFlip);
      break;
    case 'NDVI':
      indexImg = ndviTransform(img).multiply(indexFlip);
      break;
    case 'NDSI':
      indexImg = ndsiTransform(img).multiply(indexFlip);
      break;
    case 'TCB':
      indexImg = tc.select(['TCB'])//.multiply(indexFlip);
      break;
    case 'TCG':
      indexImg = tc.select(['TCG']).multiply(indexFlip);
      break;
    case 'TCW':
      indexImg = tc.select(['TCW']).multiply(indexFlip);
      break;
    case 'TCA':
      indexImg = tc.select(['TCA']).multiply(indexFlip);
      break;
    default:
      print('The index you provided is not supported');
  }

  return indexImg.set('system:time_start', img.get('system:time_start'));
};


// MAKE AN LT STACK
var makeLtStack = function(img){
  var allStack = calcIndex(img, index, 1);
  var ftvimg;
  for(var ftv in ftvList){
    ftvimg = calcIndex(img, ftvList[ftv], 0)
                      .select([ftvList[ftv]],['ftv_'+ftvList[ftv].toLowerCase()]);
    
    allStack = allStack.addBands(ftvimg)
                       .set('system:time_start', img.get('system:time_start'));
  }
  
  return allStack;
};




// #######################################################################################
// #######################################################################################
// #######################################################################################





// #######################################################################################
// ###### LANDTRENDR #####################################################################
// #######################################################################################

// ------ DEFINE FUNCTION TO EXTRACT VERTICES FROM LT RESULTS AND STACK BANDS ------------

var getLTvertStack = function(lt, runParams) {
  lt = lt.select('LandTrendr');
  var emptyArray = [];                              // make empty array to hold another array whose length will vary depending on maxSegments parameter    
  var vertLabels = [];                              // make empty array to hold band names whose length will vary depending on maxSegments parameter 
  for(var i=1;i<=runParams.maxSegments+1;i++){     // loop through the maximum number of vertices in segmentation and fill empty arrays                        // define vertex number as string 
    vertLabels.push("vert_"+i.toString());               // make a band name for given vertex
    emptyArray.push(0);                             // fill in emptyArray
  }
  
  var zeros = ee.Image(ee.Array([emptyArray,        // make an image to fill holes in result 'LandTrendr' array where vertices found is not equal to maxSegments parameter plus 1
                                 emptyArray,
                                 emptyArray]));
  
  var lbls = [['yrs_','src_','fit_'], vertLabels,]; // labels for 2 dimensions of the array that will be cast to each other in the final step of creating the vertice output 

  var vmask = lt.arraySlice(0,3,4);           // slices out the 4th row of a 4 row x N col (N = number of years in annual stack) matrix, which identifies vertices - contains only 0s and 1s, where 1 is a vertex (referring to spectral-temporal segmentation) year and 0 is not
  
  var ltVertStack = lt.arrayMask(vmask)       // uses the sliced out isVert row as a mask to only include vertice in this data - after this a pixel will only contain as many "bands" are there are vertices for that pixel - min of 2 to max of 7. 
                      .arraySlice(0, 0, 3)          // ...from the vertOnly data subset slice out the vert year row, raw spectral row, and fitted spectral row
                      .addBands(zeros)              // ...adds the 3 row x 7 col 'zeros' matrix as a band to the vertOnly array - this is an intermediate step to the goal of filling in the vertOnly data so that there are 7 vertice slots represented in the data - right now there is a mix of lengths from 2 to 7
                      .toArray(1)                   // ...concatenates the 3 row x 7 col 'zeros' matrix band to the vertOnly data so that there are at least 7 vertice slots represented - in most cases there are now > 7 slots filled but those will be truncated in the next step
                      .arraySlice(1, 0, runParams.maxSegments+1) // ...before this line runs the array has 3 rows and between 9 and 14 cols depending on how many vertices were found during segmentation for a given pixel. this step truncates the cols at 7 (the max verts allowed) so we are left with a 3 row X 7 col array
                      .arrayFlatten(lbls, '');      // ...this takes the 2-d array and makes it 1-d by stacking the unique sets of rows and cols into bands. there will be 7 bands (vertices) for vertYear, followed by 7 bands (vertices) for rawVert, followed by 7 bands (vertices) for fittedVert, according to the 'lbls' list

  return ltVertStack;                               // return the stack
};


var split_bname = function(bname) {
  var dict = {'type':'','name':'','series':''};
  var pieces = bname.split('_');
  if(pieces[0] == 'rmse'){
    dict.type = 'rmse';
    dict.name = 'rmse';
    dict.series = 'na';
  } else {
    dict.type = pieces[1],
    dict.name = pieces[1]+'_'+pieces[0],
    dict.series = pieces[2]; 
  }
  return dict;
};


// ------ RUN LANDTRENDR -----------------------------------------------------------------

for (var f=0; f < featureValues.length; f++){
  
  // Get bounding geometry and coordinates
  var featureValue = featureValues[f];
  //print(featureValue);
  var aoi = ee.FeatureCollection(featureCol)//  ee.FeatureCollection('ft:'+featureCol) // load the aoi
            //.filter(ee.Filter.equals(featureKey, featureValue)) //filter by park - left arg is the key name, right arg is the value to match
            .geometry()
            .buffer(aoiBuffer);
  var box = aoi.bounds(); // get the bounds from the drawn polygon


  // make a dummy collection for filling missing years (if there are any)
  var dummyCollection = ee.ImageCollection([ee.Image([0,0,0,0,0,0]).mask(ee.Image(0))]); // make a dummy collection to fill in potentially missing years
  var indexFlip;
  var index;
  var years = []; // make an empty array to hold years
  for (var i = startYear; i <= endYear; ++i) years.push('yr'+i.toString()); // fill the array with years from the startYear to the endYear and convert them to string
  
  var vertYearLabels = [];
  var rawVertLabels = [];
  var ftvVertLabels = [];
  var iString;
  var includeRMSE;
  for(var i=1;i<=run_params.maxSegments+1;i++){
    iString = i.toString();
    vertYearLabels.push("yrs_vert_"+iString);
    rawVertLabels.push("src_vert_"+iString);
    ftvVertLabels.push("fit_vert_"+iString);
  }
  
  
  for (var i = 0; i < startDay.length; ++i) {
    // build the annual SR collection
    var annualSRcollection = buildMosaicCollection(startYear, endYear, startDay[i], endDay[i], box, mosaicType, targetDay, dummyCollection); // put together the cloud-free medoid surface reflectance annual time series collection
    
    for (var j = 0; j < indexList.length; j++){
      // get the index and the index flipper
      index = indexList[j][0]; // pull out the index to segment on
      indexFlip = indexList[j][1];
      includeRMSE = indexList[j][2]
      // make the collection for this index and add the collection to the run parameters
      var ltCollection = annualSRcollection.map(makeLtStack); // make the LT collection for this run
      run_params.timeSeries = ltCollection; // add the single spectral index annual time series collection to the segmentation run parameter object
      
      // run LT
      var lt = ee.Algorithms.TemporalSegmentation.LandTrendr(run_params); // run LandTrendr spectral temporal segmentation algorithm
      
      print(lt);
      
      
      
      
      
      // make an empty image to append other images to
      var exportStack = ee.Image();
      
      // go through the vertList
      var nVertLayers = vertList.length;
      if(nVertLayers !== 0){
        var ltVertStack = getLTvertStack(lt.select("LandTrendr")); // extract the year, raw spectral value, and fitted values for vertices as a stacked bands
        
        for(var vert=0; vert<nVertLayers ; vert++){
          switch(vertList[vert]){
            case 'YRS':
              exportStack = exportStack.addBands(ltVertStack.select(vertYearLabels));
              break;
            case 'SRC':
              exportStack = exportStack.addBands(ltVertStack.select(rawVertLabels).multiply(indexFlip));
              break;
            case 'FIT':
              exportStack = exportStack.addBands(ltVertStack.select(ftvVertLabels).multiply(indexFlip));
              break;
            default:
              print('The index you provided is not supported');
          }
        }
      }
  
      // go through the vertList
      var nFtvLayers = ftvList.length;
      if(nFtvLayers !== 0){
        // pull out the FTV layers
        var ltInfo = lt.getInfo();
        var theseBands = [];
        var nBands = [];
        for(var b=0; b<ltInfo.bands.length; b++){
          var thisBand = ltInfo.bands[b].id;
          if(thisBand.indexOf("ftv_") >= 0){
            theseBands.push(thisBand);
          }
        }
        
        // for each ftv index - make a band name array and add the bands to the export stack
        for(var b=0; b<theseBands.length; b++){
          var pieces = theseBands[b].split("_");
          var dataSetName = pieces[1]+'_'+pieces[0];
          var bnames = [];
          for (var y = startYear; y <= endYear; ++y) bnames.push(dataSetName+'_'+y.toString()); 
          exportStack = exportStack.addBands(lt.select([theseBands[b]]).arrayFlatten([bnames]));
        }
      }
      
      
      // add the RMSE band if requested
      if(includeRMSE === true){
        exportStack = exportStack.addBands(lt.select(['rmse'],['rmse__']));
      }
      
      // exclude the first band which is just a starter
      var nBandsExportStack = exportStack.getInfo().bands.length;
      var theseBands = [];
      for(var b=1;b<nBandsExportStack;b++){
        theseBands.push(b);
      }
      exportStack = exportStack.select(theseBands).round().toShort();
  
      
      
      // make a dictionary of band names 
      var nBandsExportStack = exportStack.bandNames().getInfo();
      for(var b=0;b<nBandsExportStack.length;b++){
        var stackBand = b+1;
        var attr = split_bname(nBandsExportStack[b]);
        var bandInfo = ee.Dictionary({'type':attr.type, 'name':attr.name, 'series':attr.series, 'band':stackBand});
        if(b === 0){
          var bandDict = ee.FeatureCollection(ee.Feature(null, bandInfo));
        } else{
          bandDict = bandDict.merge(ee.FeatureCollection(ee.Feature(null, bandInfo)));
        }
      }
      
      
      
      
      // make a file name
      var nVert = parseInt(run_params.maxSegments)+1;
      var fileNamePrefix = featureKey+'-'+featureValue+'-'+index+'-'+nVert.toString()+'-'+startYear.toString()+endYear.toString() + '-' + startDay[i].replace('-', '') + endDay[i].replace('-', '');                   
  
  
      // make a dictionary of the run info
      var runInfo = ee.Dictionary({
        'featureKey': featureKey, 
        'featureValue': featureValue, 
        'segIndex': index, 
        'nVerts': nVert,
        'startYear': startYear,
        'endYear': endYear,
        'startDay': startDay[i],
        'endDay': endDay[i],
        'run_name': fileNamePrefix,
        'aoiDef': featureCol,
        'affine': affine,
        'aoiBuffer': aoiBuffer,
        'gDriveFolder': gDriveFolder,
        'mosaicType': mosaicType,
        'maxSegments': run_params.maxSegments,
        'spikeThreshold': run_params.spikeThreshold,
        'vertexCountOvershoot': run_params.vertexCountOvershoot,
        'preventOneYearRecovery': run_params.preventOneYearRecovery,
        'recoveryThreshold': run_params.recoveryThreshold,
        'pvalThreshold': run_params.pvalThreshold,
        'bestModelProportion': run_params.bestModelProportion,
        'minObservationsNeeded': run_params.minObservationsNeeded,
        'ftvList': ftvList,
        'vertList': vertList,
        'rmse': includeRMSE
      });
      
      // print(runInfo);
      // print(exportStack);
      // print(bandDict);
      
      var runInfo = ee.FeatureCollection(ee.Feature(null, runInfo));
  
      // EXPORT STUFF
      // export the run info.
      Export.table.toDrive({
        collection: runInfo,
        description: fileNamePrefix+'-run_info',
        folder: gDriveFolder,
        fileNamePrefix: fileNamePrefix+'-run_info',
        fileFormat: 'CSV'
      });
  
  
      // export the band info.
      Export.table.toDrive({
        collection: bandDict,
        description: fileNamePrefix+'-band_info',
        folder: gDriveFolder,
        fileNamePrefix: fileNamePrefix+'-band_info',
        fileFormat: 'CSV'
      });
      
     
      // export the data stack
      Export.image.toDrive({'image': exportStack.clip(aoi), 'region': aoi, 'description': fileNamePrefix, 'folder': gDriveFolder, 'fileNamePrefix': fileNamePrefix, 'crs': outProj, 'crsTransform': affine, 'maxPixels': 1e13});
  
    }
  }
}

Map.centerObject(featureCol, 12)
Map.addLayer( exportStack.clip(aoi))