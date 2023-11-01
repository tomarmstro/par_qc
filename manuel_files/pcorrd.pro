pro pcorrd
  ;this program takes  par data from daviesB_15062021_29062023.txt
  ;and applies a linear correction ratio to 10 minte PAR data. Slope and intercept
  ;are obtained from pcorrc.pro which uses a robust fit to ratiop vs day sequence
  ; for all cloudless days. The method assumes that a corrected PAR for cloud 
  ; conditions (PARcorr) equals PARraw*ratiop(day). The technique has been validated
  ; in Nunez et al., 2022. 
   close,/all
  ; input data: data=float array with columns: day sequence(dn1), julian day(dn), day of month (day),
  ; month(mo), year(yr),hour(hr), minute (minute), model PAR (modpar),raw PAR (rawpar).
  ;daviesB_15062021_29062023.txt'= input data from 15/06/2021 to 29/06/2023.
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
  ;  x=fltarr(1,n)
  ;  y=fltarr(1,n)
  openr,10,'c:\data\AIMS\rv_correction\Davies Reef\daviesB_15062021_29062023.txt',data
  openw,11,'c:\data\AIMS\rv_correction\Davies Reef\PARcorr_15062021_29062023.txt'
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
  for i=0,n-1 do begin
  pratio=0.0001221*dn1(i)+0.95767
  corrpar=rawpar(i)*pratio
  printf,11,dn1(i),dn(i),day(i),mo(i),yr(i),minute(i),rawpar(i),corrpar,$
    format='(1x,6(2x,f5.0),2(2x,f8.3))'
  endfor
  close,/all
  end