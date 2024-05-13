pro save_tif

  template_image = READ_TIFF('F:\esriContest\composited\041032\0.tif',GEOTIFF = GeoKeys)
  data = size(template_image,/dimensions)
  print,data
  
  fileformat = '.tif'
  dem = 51
  years = indgen(dem)

  ; Initialize an array to hold the NDVI/TCG/TCW values
  values = fltarr(dem, data[1], data[2]);demensions,columns,rows
  
  ; Loop over each year/file
  for i=0, n_elements(years)-1 do begin
    ; Construct the filename
    filename = string('F:\esriContest\composited\041032\', string(strtrim(years[i],2)), fileformat)
    ;print, 'reading file: ',filename
    print,'reading...',filename
    ; Open the file
    openr, unit, filename, /get_lun
    ; Read the data
    raster = read_tiff(filename)
    ;help, raster
    values[i,*,*] = raster[0,*,*];0->ndvi,1->tcg,2->tcw
    ; Close the file
    free_lun, unit
  endfor
  
  goods = indgen(dem)

  background_val       = 0
  divisor              = 1
  minneeded            = 6
  kernelsize           = 1
  pval                 = 0.05
  fix_doy_effect       = 1
  max_segments         = 6
  recovery_threshold   = 0.25
  skipfactor           = 1
  desawtooth_val       = 0.9
  distweightfactor     = 2
  vertexcountovershoot = 3
  bestmodelproportion  = 0.75
  modifier             = 1
  
  pixel_data=fltarr(dem, data[1], data[2])
  
  for cols=0, data[1]-1 do begin
    print,'loop(total 620):',cols
    for rows=0, data[2]-1 do begin
      result = fit_trajectory_v2(years, goods, values[*,cols,rows], $
        minneeded, background_val, modifier, seed, $
        desawtooth_val, pval, $
        max_segments, recovery_threshold, $
        distweightfactor, vertexcountovershoot, $
        bestmodelproportion)
      if result.ok eq 1 then begin
        pixel_data[*,cols,rows] = result.best_model.yfit
      endif
    endfor
  endfor
  
;  tlb=WIDGET_BASE(xsize=400,ysize=400)
;  WIDGET_CONTROL,tlb,map=0,/real
;  prsbar=IDLITWDPROGRESSBAR(Group_LEADER=tlb,title='Progress',Cancel=cancelIn)
;  
;  for cols=0, data[1]-1 do begin
;    for rows=0, data[2]-1 do begin
;      if widget_info(prsbar,/valid) then begin
;        IDLITWDPROGRESSBAR_SETVALUE,prsbar,(100/(data[1]*data[2])*(cols+1)*(rows+1))
;        result = fit_trajectory_v2(years, goods, values[*,cols,rows], $
;          minneeded, background_val, modifier, seed, $
;          desawtooth_val, pval, $
;          max_segments, recovery_threshold, $
;          distweightfactor, vertexcountovershoot, $
;          bestmodelproportion)
;          if result.ok eq 1 then begin
;            pixel_data[*,cols,rows] = result.best_model.yfit
;          endif
;      endif else begin
;        tmp=DIALOG_MESSAGE('Cancel Percent'+STRING(i)+'%',/info)
;        Break
;      endelse
;    endfor
;  endfor
;  Widget_Control,tlb,/destroy

  print,'Begin writing...'
  output_filename = 'F:\esriContest\composited\041032\tcw51years.tif'
  WRITE_TIFF, output_filename, pixel_data, geotiff = GeoKeys,/APPEND,/FLOAT
  print,'Done...'
  
;  image_filenames = ['F:\esriContest\composited\041032\1.tif',$
;    'F:\esriContest\composited\041032\2.tif','F:\esriContest\composited\041032\3.tif','F:\esriContest\composited\041032\4.tif',$
;    'F:\esriContest\composited\041032\5.tif','F:\esriContest\composited\041032\6.tif','F:\esriContest\composited\041032\7.tif',$
;    'F:\esriContest\composited\041032\8.tif','F:\esriContest\composited\041032\9.tif','F:\esriContest\composited\041032\10.tif']
;  num_bands = 10
;  pixel_data = FLTARR(num_bands, data[1], data[2])
;  output_filename = 'F:\esriContest\composited\tcg\output_image2.tif'
;  FOR i = 0, num_bands - 1 DO BEGIN
;    image_data = READ_TIFF(image_filenames[i])
;    ;print,image_filenames[i]
;    pixel_data[i,*,*] = image_data[0,*,*]
;    ;WRITE_TIFF, output_filename, pixel_data,geotiff = GeoKeys,/APPEND
;  ENDFOR
;  print,pixel_data[*,0,0]

end