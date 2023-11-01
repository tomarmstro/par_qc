pro pcorrB
  ;this program takes  par data from a station with zenith angle, model PAR,and raw PAR
  ;estimates cloudless days  by looking at  stdev as model values are 
  ;lowered by constant amount over the day. 
  close,/all
  ; input data: data=float array with columns: day sequence(dn1), julian day(dn), day of month (day),
  ; month(mo), year(yr),hour(hr), minute (minute), model PAR (modpar),raw PAR (rawpar). 
  ;daviesB_15062021_29062023.txt'= input data from 15/06/2021 to 29/06/2023.
  ;clear_stats_15062021_29052023.txt': data containing goodness of fit information for every day in series.
  ; cloudlessin_15062021_29052023.txt': file containing all cloudless days that have passed the test.
  n=51733
  data=fltarr(10,n)
  dn1=fltarr(1,n)
  dn=fltarr(1,n)
  day=fltarr(1,n)
  mo=fltarr(1,n)
  yr=fltarr(1,n)
  hr=fltarr(1,n)
  z=fltarr(1,n)
  minute=fltarr(1,n)
  rawpar=fltarr(1,n)
  modpar=fltarr(1,n)
  const=fltarr(19)
  const1=fltarr(19)
  x=fltarr(1,n)
  y=fltarr(1,n)
  openr,10,'c:\data\AIMS\rv_correction\Davies Reef\daviesB_15062021_29062023.txt',data
  openw,11,'c:\data\AIMS\rv_correction\Davies Reef\clear_stats_15062021_29062023.txt'
  openw,12,'c:\data\AIMS\rv_correction\Davies Reef\cloudlessin_15062021_29062023.txt'
  readf,10,data
  dn1(*)=data(0,*)
  dn(*)=data(1,*)
  day(*)=data(2,*)
  mo(*)=data(3,*)
  yr(*)=data(4,*)
  hr(*)=data(5,*)
  minute(*)=data(6,*)
  z(*)=data(7,*)
  modpar(*)=data(8,*)
  rawpar(*)=data(9,*)
  dnstart=dn(0)
  yrstart=yr(0) 
  dayold=day(0)
  mold=mo(0)
  yrold=yr(0)
  ii=0L
  iold=0L
  ; Two series are created for each day: x containing raw PAR and y containing model PAR. 
  ; Next a procedure is called named "statsxy,x,y,const,const1,sumx,sumy"  Each element in y is multiplied 
  ; by a ratio which can take values 2.0, 1.9, 1.8,...,0.2. Each value of the constant creates
  ; a new series y1. std dev (y1:x) and stddev(y1:x)/max(y) are estimated for each of the 19 series 
  ; that y1 can take. A series with a threshold stdev(y1:x)/max(y) less than or equal to "dx" is labelled cloudless. 
  ; value for "dx" can vary. In this program dx=0.1 
  ; Once a cloudless day is selected, a "ratiop" is created and is equal to  sum(y)/sum(x). This ratiop is 
  ; a correction factor which is applied to all the 10-minute raw data. 
  for i=0,n-1 do begin
    dn10=dn1(i)
    dn0=dn(i)
    mo0=mo(i)
    yr0=yr(i)
    day0=day(i)
    if day0 ne dayold then begin
;      print,dayold,day0
    x=rawpar(iold:ii-1)
    y=modpar(iold:ii-1)
    hr1=hr(iold:ii-1)
    minute1=minute(iold:ii-1)
    statsxy,x,y,const,const1,sumx,sumy
    
    printf,11,dn1(i-1),dn(i-1),dayold,mo(i-1),yr(i-1),const1(0:18) $
    ,sumx,sumy,format='(1x,5(2x,f7.0),2x,19(1x,f8.3),2(1x,f8.2))' 
    dx=0.10
    ; Only factors from 1.5 to 0.5 are considered to get cloudless days. Good enough!!!
   print,iold,ii-1,dn0,dayold,mold,yrold,rawpar(iold),rawpar(ii-1)
    if const1(5) le dx or const1(5) le dx or const1(6) le dx or const1(7) le dx $
      or const1(8) le dx or const1(9) le dx or const1(10) le dx or const1(11) le dx $
      or const1(12) le dx or const1(13) le dx or const1(14) le dx or const1(15) le dx $
      then begin
      ratiop=sumy/sumx
      printf,12,dn1(i-1),dn(i-1),dayold,mo(i-1),yr(i-1),ratiop
       endif
    dayold=day0
    mold=mo0
    yrold=yr0
    iold=ii
      endif
       ii=ii+1
  endfor
     close,/all
  end
  pro statsxy,x,y,const,const1,sumx,sumy
  const=fltarr(19)
  const1=fltarr(19)
  for j=0,18 do begin
  dely=2.0-j*0.1
  y1=y*dely
  const(j)=stddev(x-y1)
  const1(j)=const(j)/max(y)
  sumx=(600.0/10^6)*total(x)
  sumy=(600.0/10^6)*total(y)
  endfor
  end
  
  
   