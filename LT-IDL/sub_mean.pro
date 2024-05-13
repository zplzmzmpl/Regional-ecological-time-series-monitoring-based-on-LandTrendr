pro sub_mean
  template_image = READ_TIFF('F:\esriContest\composited\041032\NDVI51years.tif',GEOTIFF = GeoKeys)
  data = size(template_image,/dimensions)
  temp = FLOAT(template_image)
  print,temp(*,0,0)
;  mean_val = fltarr(data[1],data[2])
;  for cols = 0,data[1]-1 do begin
;    print,'loop(total 619):',cols
;    for rows = 0,data[2]-1 do begin
;      mean_val(cols,rows) = mean(temp(*,cols,rows))
;    endfor
;  endfor
;  print,mean_val(0,0)
;  for bands = 0,data[0]-1 do begin
;    print,'step(total 19):',bands
;    temp(bands,*,*) -= mean_val
;  endfor
;  print,temp(*,0,0)
;  filtering = MEAN_FILTER(temp,3,/GEOMETRIC)
  for bands = 0, data[0]-1 do begin
    single_img = temp(bands,*,*)
    print,size(single_img,/dimensions)
    temp(bands,*,*) = ESTIMATOR_FILTER(single_img,3,/median)
  endfor
  ;filtering = ESTIMATOR_FILTER(temp,3,3,/median)
  print,'Begin writing...'
  output_filename = 'F:\esriContest\composited\041032\NDVI51years_median.tif'
  WRITE_TIFF, output_filename, temp, geotiff = GeoKeys,/APPEND,/FLOAT
  print,'Done...'
end