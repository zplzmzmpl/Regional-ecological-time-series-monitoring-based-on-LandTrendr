pro get_ndvi_values,directory,init_year,duration,$
  max_segments,  vertexcountovershoot, recovery_threshold,$
  desawtooth_val, pval, bestmodelproportion
  
  compile_opt idl2

  ; Define the directory, filename format, and years of interest
  ;directory = 'F:\esriContest\041029\'
  fileformat = '.tif'
  years = indgen(duration) + init_year

  ; Initialize an array to hold the NDVI values
  values = fltarr(n_elements(years))
  ; Loop over each year/file
  for i=0, n_elements(years)-1 do begin
    ; Construct the filename
    filename = string(directory, string(strtrim(years[i],2)), fileformat)
    print, 'reading file: ',filename
    ; Open the file
    openr, unit, filename, /get_lun
    ; Read the data
    raster = read_tiff(filename)
    ;help, raster
    values[i] = raster[0, 4787.9009,6203.8521]
    ; Close the file
    free_lun, unit
  endfor

  ; Print the results
  for i=0, n_elements(years)-1 do print, years[i], values[i]
  ;run_seg_fit,years,values
  ;years = [2000, 2001, 2002, 2003, 2004, 2005, 2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
  ;values = [645, 672, 701, 362, 648, 624, 535, 601, 527, 665, 553,  81,  85,  72,  53,  44, 227, 318, 292, 207, 223, 300]
  ;allyears = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]
  ;  goods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,$
  ;    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,$
  ;    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,$
  ;    48, 49, 50] ; Assuming all years are valid
  goods = indgen(n_elements(years))

  background_val       = 0
  divisor              = 1
  minneeded            = 6
  kernelsize           = 1
  ;pval                 = 0.05
  fix_doy_effect       = 1
  ;max_segments         = 6
  ;recovery_threshold   = 0.25
  skipfactor           = 1
  ;desawtooth_val       = 0.9
  distweightfactor     = 2
  ;vertexcountovershoot = 3
  ;bestmodelproportion  = 0.75
  modifier             = 1

  result = fit_trajectory_v2(years, goods, values, $
    minneeded, background_val, modifier, seed, $
    desawtooth_val, pval, $
    max_segments, recovery_threshold, $
    distweightfactor, vertexcountovershoot, $
    bestmodelproportion)

  ;  best_model = {   vertices:intarr(max_count), $
  ;    vertvals:intarr(max_count), $
  ;    yfit:fltarr(n_all_yrs)+mean(vvals), $
  ;    n_segments:0, $
  ;    p_of_f:1.0, $
  ;    f_stat:0., $
  ;    ms_regr:0., $
  ;    ms_resid:0., $   ;1 = autocorr, 2, = find_segments6, 3=findsegments 7
  ;    segment_mse:intarr(max_count-1)-1} ;set to negative one as flag
  print,result.best_model

  p1=plot(years,values,color='red',name='Origin NDVI',linestyle = 1,THICK=3)
  p2=plot(years,result.best_model.yfit,color='blue',/overplot,name='Fitted NDVI',linestyle = 0,THICK=2)
  leg = legend(target = [p1,p2],position = [1,0.95])
  p1.FONT_SIZE = 15
  p1.title = 'Landtrendr'
  p1.title.FONT_SIZE=30

  ax = p2.axes
  ax[0].TITLE = 'years'
  ax[1].TITLE = 'NDVI'
  ax[2].hide = 1
  ax[3].hide = 1
  ax[0].ticklen=0.01
  ax[1].ticklen=0.01
end
