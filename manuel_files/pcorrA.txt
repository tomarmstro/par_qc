pro pcorrA
; this program takes raw par data from a station,
; estimates solar zenith angle and calculates clear sky PAR every 10 minutes. 
; input data=data=float array with columns: day(day), month (mo), year(yr), hr (hour), minute (minute), raw par(rawpar)
; data extends from 15/06/2021 to 29/06/2023
; n=number of lines. Each line is a 10 minute record 
; column vectors are established for each variable in the record
; rv*ptheor = final value of Model PAR based on ptheor but corrected for sun-earth distance (rv) for that particular day
; formulas for declination (dec), equation of time (et),radius vector (rv), zenith angle (z) are taken
;  from Iqbal, M. 1983: Introduction to Solar Radiation, Academic Press.
;         PROCEDURES (subroutines)
; daynumer,day0,mo0,yr0,dn                 estimates Julian date (dn) based on day,month, year
; dayseq,dnstart,yrstart,dn,yr0,dn1        estimates day number (dn1) since start of record. Might extend over several years
; zen,lat0,long0,dn,hr0,min0,rv,et,dec,z   estimates solar zenith angle z (90deg minus elevation)based on latitude(lat0 in radians), longitude (long0 in degrees)
;                                          ,jul day (dn), hour (hr0), minute (min0,) et (equation of time), declination (dec)
; theor,z, ptheor                          estimates model PAR (ptheor) based on cosine of solarzenith angle cz. 

close,/all
n=60529
pi=3.1415926
data=fltarr(6,n)
day=fltarr(1,n)
mo=fltarr(1,n)
yr=fltarr(1,n)
hr=fltarr(1,n)
minute=fltarr(1,n)
rawpar=fltarr(1,n)
lat0=-18.8316*3.1415926/180.0
long0=147.6345
;openr,10,'c:\data\AIMS\Neal_Nov_2021\liz_par.txt',data
;openw,11,'c:\data\AIMS\Neal_Nov_2021\liz_par_out.txt'
openr,10,'c:\data\AIMS\rv_correction\Davies Reef\daviesA_15062021_29062023.txt',data
openw,11,'c:\data\AIMS\rv_correction\Davies Reef\daviesB_15062021_29062023.txt'
readf,10,data
day(*)=data(0,*)
mo(*)=data(1,*)
yr(*)=data(2,*)
hr=data(3,*)
minute(*)=data(4,*)
rawpar(*)=data(5,*)
day0=day(0)
mo0=mo(0)
yr0=yr(0)
; day number procedure is called at start to establish start day.
daynumber,day0,mo0,yr0,dn
dnstart=dn
yrstart=yr(0)
for i=0,n-1 do begin
;  for i=0,100 do begin
  day0=day(i)
  mo0=mo(i)
  hr0=hr(i)
  min0=minute(i)
  yr0=yr(i)
; time scale moved back 10 minutes to coincide with start of integration period.
; later in the program, when calculating True Solar Time (TST), 5 minutes are added
; to make calculation coincide with mid-point of integration period
  min0=min0-10.0
  if min0 lt 0.0 then begin
    min0=50.0
    hr0=hr0-1.0
  endif
  rawpar0=rawpar(i)
;  printf,11,day0,mo0,hr0,min0,yr0,rawpar0
  ; day number estimated
  daynumber,day0,mo0,yr0,dn
  dayseq,dnstart,yrstart,dn,yr0,dn1
  ; zenith angle z estimated.if z lt 0 skip
  if hr0 ge 5.0 and hr0 le 19.00 then begin
  zen,lat0,long0,dn,hr0,min0,rv,et,dec,z
 if z gt 0.0 and z lt 90.0 then begin
 theor,z,ptheor
 ptheor=ptheor*rv
 printf,11,dn1,dn,day0,mo0,yr0,hr0,min0,z,ptheor,rawpar0 $
  ,format='(1x,7(2x,f5.0),3(2x,f8.3))'
;printf,11,hr0,min0,rv,dec*180/pi,et*60.0,z,ptheor,rawpar0,format='(8(1x,f10.5))
 endif 
 endif
endfor
close,/all
end
pro daynumber,day0,mo0,yr0,dn
  if mo0 eq 1 then dold=0.0
  if mo0 eq 2 then dold=31.0
  if mo0 eq 3 then dold=59.0
  if mo0 eq 4 then dold=90.0
  if mo0 eq 5 then dold=120.0
  if mo0 eq 6 then dold=151.0
  if mo0 eq 7 then dold=181.0
  if mo0 eq 8 then dold=212.0
  if mo0 eq 9 then dold=243.0
  if mo0 eq 10 then dold=273.0
  if mo0 eq 11 then dold=304.0
  if mo0 eq 12 then dold=334.0
  dn=dold+day0
; adjustment for leap years 
  if yr0 eq 2012 or yr0 eq 2016 or yr0 eq 2020 and day0 gt 59 then dn=dn+1
  return
end
pro zen,lat0,long0,dn,hr0,min0,rv,et,dec,z
;xl is a yearly time scale extending from 0 to 2 pi.
; declination angle estimated
pi=3.1415926
xl=2.0*pi*(dn-1)/365.0
dec=(0.006918-0.399912*cos(xl)+0.07257*sin(xl)$
-0.006758*cos(2.0*xl)+0.000907*sin(2.0*xl) $
-0.002697*cos(3.0*xl)+0.00148*sin(3.0*xl))
;
; radius vector estimated
rv=1.000110+0.034221*cos(xl)+0.001280*sin(xl)$
+0.000719*cos(2.0*xl)+0.000077*sin(2.0*xl)
;
; equation of time estimated
et=(0.000075+0.001868*cos(xl)-0.032077*sin(xl) $
-0.014615*cos(2.0*xl)-0.04089*sin(2.0*xl))*229.18/60
;
;zenith angle estimated (z)
tst=hr0+((min0+5.0)/60.0)-(4.0/60.0)*abs(150.0-long0)+et
hrang=(12.0-tst)*15.0*pi/180.0
cz=sin(lat0)*sin(dec)+cos(lat0)*cos(dec)*cos(hrang)
z=(180.0/pi)*acos(cz)
cc1=sin(lat0)
cc2=sin(dec)
cc3=cos(lat0)
cc4=cos(dec)
cc5=hrang
cc6=hrang*180.0/3.14159
;printf,11,hr0,min0,tst,cc1,cc2,cc3,cc4,cc5,cc6,cz,et*180/3.1415926,dn,format='(12(1x,f10.5))'
end
pro theor,z, ptheor
pi=3.1415926
z0=z*pi/180.0
cz=cos(z0)
ptheor=-7.1165+768.894*cz+4023.167*cz^2-4180.1969*cz^3 +1575.0067*cz^4
end
pro dayseq,dnstart,yrstart,dn,yr0,dn1
  del1=yr0-yrstart
  if yr0 eq yrstart then dn1=dn-dnstart
  if del1 eq 1 then dn1=365-dnstart+dn
  if del1 eq 2 then dn1=365-dnstart+365+dn
  if del1 eq 3 then dn1=365 -dnstart +730 +dn
  if del1 eq 4 then dn1=365 -dnstart +1095 +dn
  if del1 eq 5 then dn1=365 -dnstart +1460 +dn
  if del1 eq 6 then dn1=365 -dnstart +1825 +dn
end
