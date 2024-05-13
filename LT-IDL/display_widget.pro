PRO display_widget_event,ev
  widget_control,ev.top,get_uvalue=pstate
  uname=widget_info(ev.id,/uname)
  ;print,'you click:',uname
  case uname of
    'open':begin
      file=dialog_pickfile(title='pick folder',$
        ;filter=[],$
        path=(*pstate).curpath,$
        /DIRECTORY,$
        get_path=curpath)
      if curpath ne '' then (*pstate).curpath=curpath
      ;if ~file_test(file) then return
      ;print,file
      widget_control,(*pstate).ttxt,set_value=file
    end
    'ok':begin
      tlb=WIDGET_BASE(xsize=400,ysize=400)
      WIDGET_CONTROL,tlb,map=0,/real
      prsbar=IDLITWDPROGRESSBAR(Group_LEADER=tlb,title='Progress',Cancel=cancelIn)
      fileformat = '.tif'
      years = indgen((*pstate).duration) + (*pstate).init_year
      
      ; Initialize an array to hold the NDVI values
      values = fltarr(n_elements(years), 620, 628)
      ; Loop over each year/file
      for i=0, n_elements(years)-1 do begin
        IF WIDGET_INFO(prsbar,/valid)THEN BEGIN
          IDLITWDPROGRESSBAR_SETVALUE,prsbar,(100/n_elements(years)*(i+1))
          ; Construct the filename
          filename = string((*pstate).curpath, string(strtrim(years[i],2)), fileformat)
          ;print, 'reading file: ',filename
          widget_control,(*pstate).ttxt,set_value='reading file: ' + filename
          ; Open the file
          openr, unit, filename, /get_lun
          ; Read the data
          raster = read_tiff(filename)
          ;help, raster
          values[i,*,*] = raster[0,*,*]
          ; Close the file
          free_lun, unit
        ENDIF ELSE BEGIN
          tmp=DIALOG_MESSAGE('Cancel Percent'+STRING(i)+'%',/info)
          Break
        ENDELSE
      endfor
      Widget_Control,tlb,/destroy
      ; Print the results
      ;for i=0, n_elements(years)-1 do print, years[i], values[i]
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
      ;row=4563.67
      ;column=2353,78
      ;4716.2979,6214.4004
      ;print,values[*,234,468]
      
      result = fit_trajectory_v2(years, goods, values[*,234,468], $
        minneeded, background_val, modifier, seed, $
        (*pstate).desawtooth_val, (*pstate).pval, $
        (*pstate).max_segments, (*pstate).recovery_threshold, $
        distweightfactor, (*pstate).vertexovershoot, $
        (*pstate).bestmodelproportion)
      
      ;  best_model = {   vertices:intarr(max_count), $
      ;    vertvals:intarr(max_count), $
      ;    yfit:fltarr(n_all_yrs)+mean(vvals), $
      ;    n_segments:0, $
      ;    p_of_f:1.0, $
      ;    f_stat:0., $
      ;    ms_regr:0., $
      ;    ms_resid:0., $   ;1 = autocorr, 2, = find_segments6, 3=findsegments 7
      ;    segment_mse:intarr(max_count-1)-1} ;set to negative one as flag
      if result.ok eq 1 then begin
        print,result.best_model
        v = result.best_model.vertices[0:result.best_model.n_segments]
        widget_control,(*pstate).vertex,set_value=v
        vval = result.best_model.vertvals[0:result.best_model.n_segments]
        widget_control,(*pstate).vertex_val,set_value=vval
        s = result.best_model.n_segments
        widget_control,(*pstate).final_segs,set_value=s
        pp = result.best_model.p_of_f
        widget_control,(*pstate).p,set_value= pp
        fs = result.best_model.f_stat
        widget_control,(*pstate).f_stat,set_value= fs
        p1=plot(years,values,color='red',name='ORIGIN VALUE',linestyle = 1,THICK=3)
        p2=plot(years,result.best_model.yfit,color='blue',/overplot,name='FITTED VALUE',linestyle = 0,THICK=2)
        leg = legend(target = [p1,p2],position = [1,0.95])
        p1.FONT_SIZE = 15
        p1.title = 'Landtrendr'
        p1.title.FONT_SIZE=30

        ax = p2.axes
        ax[0].TITLE = 'YEARS'
        ax[1].TITLE = 'VALUE'
        ax[2].hide = 1
        ax[3].hide = 1
        ax[0].ticklen=0.01
        ax[1].ticklen=0.01
        ;      print,(*pstate).curpath,(*pstate).init_year,(*pstate).duration,$
        ;        (*pstate).max_segments,(*pstate).vertexovershoot,(*pstate).recovery_threshold,$
        ;        (*pstate).desawtooth_val,(*pstate).pval,(*pstate).bestmodelproportion
        ;
        ;      get_ndvi_values,(*pstate).curpath,(*pstate).init_year,(*pstate).duration,$
        ;        (*pstate).max_segments,(*pstate).vertexovershoot,(*pstate).recovery_threshold,$
        ;        (*pstate).desawtooth_val,(*pstate).pval,(*pstate).bestmodelproportion,$
        ;        (*pstate).row,(*pstate).column,(*pstate).index
      endif
      
    end
    'year':begin
      ;widget_control,(*pstate).init_year,set_value=ev.value
      (*pstate).init_year=ev.value
      ;print,'init year:',(*pstate).init_year
    end
    'duration':begin
      (*pstate).duration=ev.value
      ;print,'duration:',(*pstate).duration
    end
    'maxseg':begin
      (*pstate).max_segments=ev.value
      ;print,'max_segments:',(*pstate).max_segments
    end
    'vos':begin
      (*pstate).vertexovershoot=ev.value
      ;print,'vertexovershoot:',(*pstate).vertexovershoot
    end
    'rt':begin
      (*pstate).recovery_threshold=ev.value
      ;print,'recovery_threshold:',(*pstate).recovery_threshold
    end
    'desaw':begin
      (*pstate).desawtooth_val=ev.value
      ;print,'desawtooth_val:',(*pstate).desawtooth_val
    end
    'pval':begin
      (*pstate).pval=ev.value
      ;print,'pval:',(*pstate).pval
    end
    'bestmodelp':begin
      (*pstate).bestmodelproportion=ev.value
      ;print,'bestmodelproportion:',(*pstate).bestmodelproportion
    end
;    'row':begin
;      (*pstate).row=ev.value
;      ;print,'row:',(*pstate).row
;    end
;    'col':begin
;      (*pstate).column=ev.value
;      print,'row:',(*pstate).column
;    end
    'index':begin
      id = widget_info(ev.id,/droplist_select)
      (*pstate).index = id
      ;print,(*pstate).index
    end
  endcase
end
PRO display_widget

  tlb=widget_base(title='landtrendr',/column)
  tbbase=widget_base(tlb,/frame,/row)
  tcbase=widget_base(tlb,/frame,/column)
  ;tcbase2=widget_base(tlb,/frame,/row)
  ;tcbase3=widget_base(tlb,/frame,/row)
  
  topen=widget_button(tbbase,value='open.bmp',$
    /bitmap,uname='open')
  ttxt=widget_text(tbbase,value='',xsize=50)
  f1 = cw_field(tcbase,value=1972,title = 'initail year',uname='year',/integer,/RETURN_EVENTS,/FOCUS_EVENTS)
  f2 = cw_field(tcbase,value=50,title='duration',uname='duration',/integer,/RETURN_EVENTS,/FOCUS_EVENTS)
  ;print,duration
  f3 = cw_field(tcbase,value=6,title = 'max_segments',uname='maxseg',/integer,/RETURN_EVENTS,/FOCUS_EVENTS)
  f4 = cw_field(tcbase,value=3,title = 'vertexcountovershoot',uname='vos',/integer,/RETURN_EVENTS,/FOCUS_EVENTS)
  f5 = cw_field(tcbase,value=0.25,title = 'recovery_threshold',uname='rt',/floating,/RETURN_EVENTS,/FOCUS_EVENTS)
  f6 = cw_field(tcbase,value=0.9,title = 'desawtooth_val',uname='desaw',/floating,/RETURN_EVENTS,/FOCUS_EVENTS)
  f7 = cw_field(tcbase,value=0.05,title = 'pval',uname='pval',/floating,/RETURN_EVENTS,/FOCUS_EVENTS)
  f8 = cw_field(tcbase,value=0.75,title = 'bestmodelproportion',uname='bestmodelp',/floating,/RETURN_EVENTS,/FOCUS_EVENTS)
  ;f9 = cw_field(tcbase,value=0,title = 'row',uname='row',/floating,/RETURN_EVENTS,/FOCUS_EVENTS)
  ;f10 = cw_field(tcbase,value=0,title = 'column',uname='col',/floating,/RETURN_EVENTS,/FOCUS_EVENTS)
  f11 = widget_droplist(tcbase,title='index',uname='index',value = ['BAND0', 'BAND1', 'BAND2'])
  f12 = cw_field(tcbase,value=0,title = 'vertices',uname='vertices',/noedit)
  f13 = cw_field(tcbase,value=0,title = 'vertices_vals',uname='vertices_vals',/noedit)
  f14 = cw_field(tcbase,value=0,title = 'final segments',uname='segments',/noedit)
  f15 = cw_field(tcbase,value=0,title = 'p_val',uname='ppp',/noedit)
  f16 = cw_field(tcbase,value=0,title = 'f_statistic',uname='fs',/noedit)
  ok = WIDGET_BUTTON(tcbase, VALUE='OK', uname='ok')

  widget_control,tlb,/realize
  ;set window center in src
  device,get_screen_size=ss
  info=widget_info(tlb,/geometry)
  tlb_xy=[info.scr_xsize,info.scr_ysize]
  offset=[ss-tlb_xy]/2
  widget_control,tlb,xoffset=offset[0],$
    yoffset=offset[1]

;  widget_control,wdraw,get_value=winid
;  wset,winid
;  device,decomposed=0
;  loadct,0
;  erase,255

  pstate={ttxt:ttxt,  curpath:'',$
    init_year:f1, duration:f2,  max_segments:f3,$
    vertexovershoot:f4, recovery_threshold:float(f5),$
    desawtooth_val:float(f5), pval:float(f7), bestmodelproportion:float(f8),$
    index:f11,vertex:f12, vertex_val:f13, final_segs:f14, $
    p:f15,  f_stat:f16}
  ; innitial params
  pstate.init_year = 1972
  pstate.duration = 50
  pstate.max_segments = 6
  pstate.vertexovershoot = 3
  pstate.recovery_threshold = 0.25
  pstate.desawtooth_val = 0.9
  pstate.pval = 0.05
  pstate.bestmodelproportion = 0.75
  ;print,pstate
  ;row:4255.78
  ;column6453.45
  widget_control,tlb,set_uvalue=ptr_new(pstate)


  XMANAGER,'display_widget',tlb,/no_block
end