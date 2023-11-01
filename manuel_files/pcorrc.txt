pro pcorrC
  ;this program takes  daily cloudless data with ratios and 
  ;fits a least square regression: ratios vs day number. 
  ;All data is accepted in the first pass. Then data greater than 2 stdev are removed.
  ; This procedure is followed until there are no more data greater than 2 stdev.
    close,/all
  ; input data: data=float array with columns: day sequence(dn1), julian day(dn), day of month (day),
  ; month(mo), year(yr),ratiop
  n=142
  x=fltarr(1,n)
  y=fltarr(1,n)
  data=fltarr(6,n)
  dn1=fltarr(1,n)
  dn=fltarr(1,n)
  day=fltarr(1,n)
  mo=fltarr(1,n)
  yr=fltarr(1,n)
  rp=fltarr(1,n)
  openr,10,'c:\data\AIMS\rv_correction\Davies Reef\cloudlessin_21_23.txt',data
  readf,10,data
  x(*)=data(0,*)
  dn(*)=data(1,*)
  day(*)=data(2,*)
  mo(*)=data(3,*)
  yr(*)=data(4,*)
  y(*)=data(5,*)
  xx=x[sort(x)]
  yy=y[sort(x)]
 v=ladfit(xx,yy)
  print,v(0)
  print,v(1)
  end