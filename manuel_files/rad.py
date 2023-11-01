# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:24:14 2023

@author: nunez
"""
# script estimates solar radiation given date, lat, long coordinates\
# and EST time
# rv= mean sun-earth distance normalised to "1".
# et=equation of time given in minutes. A small correction that only 
# depends on day of year
# dec is declimation of eath orbit around the sun. It is in degrees.
# TST is true solar time. needed to get solar zenith angle
# hang: hour angle is the angular distance between local time and 
# solar noon. Measured in degrees
# solar zenith angle = 90 degrees minus solar elevation
#cz=cosine of zenith angle.
import math
import astro_cnst
day=float(input("please input day "))
mo=float(input("please input month "))
yr=float(input("please input year "))
lat=float(input("please input lat in deg:-ve in S. hemisphere "))
long=float(input("please input long in deg "))
time=float(input("please give decimal time in EST "))
pi=3.1415926
lat=lat*pi/180
# the following commands change day of month (mo) to Julian day (dn0)
if mo == 1:
    dn0=day
if mo == 2: 
    dn0=day+31.0
if mo == 3:
   dn0=day+59.0
if mo == 4:
   dn0=day+90.0
if mo == 5:
   dn0=day+120.0
if mo == 6:
   dn0=day+151.0
if mo == 7:
   dn0=day+181.0
if mo ==8:
   dn0=day+212.0
if mo == 9:
   dn0=day+243.0
if mo == 10:
   dn0==dn0+273.0
if mo == 11:
   dn0=day+304.0
if mo == 12:
   dn0=day+334.0
dn=dn0
rv=astro_cnst.rv(dn) 
et=astro_cnst.et(dn)
decl=astro_cnst.decl(dn)
tst=time-(4.0/60.0)*abs(150.0-147.326)+et
hrang=abs(12.0-tst)*15.0*pi/180.0
cz=math.sin(lat)*math.sin(decl)+math.cos(lat)*math.cos(decl)*math.cos(hrang)
par=(7.153+800.792*cz+3980.229*(cz)**2-4160.261*(cz)**3 +1574.795*(cz)**4)*rv
print("par = ",par,"micromol m-2 s-1")
