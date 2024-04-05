#
#  Code to convolve Vesta data
#
#
#  ARC - 15nov2018
#
#  30mar2024:  adapting to Io/Live/SV instead of Vesta/GMT/Gem
#
import numpy as np
import matplotlib
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import imageio as iio
import scipy.ndimage as ndi
# import scipy.ndimage.interpolation as interp
from astropy.convolution import AiryDisk2DKernel, convolve_fft
import copy

import os

os.system( "rm -f ./foo.fits" )

#
#  Constants
#
width    = 1024

nr=2
nc=2
fig, ax = plt.subplots(nrows=nr,ncols=nc)

#imJpg = iio.imread("PIA00740.jpg")
imPng = iio.imread("SHARK_Nov_2023_264W_3.3N_0rot.png")
print( "png shape", imPng.shape )
#imoffx = 240
#imoffy = 300
midp = width/2
imoffx = int(midp-(imPng.shape[0]/2))
imoffy = int(midp-(imPng.shape[1]/2))

im = np.zeros([width,width])
im[imoffx:imoffx+imPng.shape[0],
   imoffy:imoffy+imPng.shape[1]] = \
     imPng[0:imPng.shape[0],0:imPng.shape[1],0]

for i in range(0,nr):
  for j in range(0,nc):
    ax[i,j].set_xticks([])
    ax[i,j].set_yticks([])

#
# Devise dlim of psf (in pixels):
#
#   0. Estimate subtended arcseconds of sci FoV
#   1. plateScale = imageWidth /FoV  (make image square first if nccssry)
#   2. Define the 3 dlims
#   3. psfPixWid = plateScale * dlim (3 times)
#   4. Gen the psf using airyDisk2Dkernel
#   5. Convolve using convolve_fft
#
FoV           = 1000   # milliarsec
livDlim       = 5      # scaling from 4 for GMT GMagAO-X
                       # at 0.5 microns from j males talk
svDlim        = 15
fizDlim       = livDlim * 2.2 / 0.5

plateScale = im.shape[0] / FoV   #  chk if square

livPsfPixWid  = plateScale * livDlim
radius = livPsfPixWid       # Size in pixels of Airy pattern (to first null)
psf = AiryDisk2DKernel(radius)
livRim = convolve_fft(im, psf )

svPsfPixWid  = plateScale * svDlim
radius = svPsfPixWid       # Size in pixels of Airy pattern (to first null)
psf = AiryDisk2DKernel(radius)
svRim = convolve_fft(im, psf )

fizPsfPixWid  = plateScale * fizDlim
radius = fizPsfPixWid       # Size in pixels of Airy pattern (to first null)
psf = AiryDisk2DKernel(radius)
fizRim = convolve_fft(im, psf )

print( "plateScale = " + "%lf" % plateScale + " pixel/mas" )
print( "livDlim = " + "%lf" % livDlim )
print( "livPsfPixWid = " + "%lf" % livPsfPixWid + " = plateScale * livDlim" )

#
#  Plots start here
#
#yf=-40    #  yfudge
#xf=+10
yf=0    #  yfudge
xf=0
zwid = 400

zim1 = np.zeros([zwid*2,zwid*2])
zim1 = livRim[int(midp-zwid+yf):int(midp+zwid-1+yf), \
              int(midp-zwid+xf):int(midp+zwid-1+xf)]
ax[0,0].imshow( zim1, cmap='gray' )
ax[0,0].text(0.5,-0.08, "LIVE at 0.5 microns", size=12, ha="center",
        transform=ax[0,0].transAxes)

zim2 = np.zeros([zwid*2,zwid*2])
zim2 = fizRim[int(midp-zwid+yf):int(midp+zwid-1+yf),
              int(midp-zwid+xf):int(midp+zwid-1+xf)]
ax[1,1].imshow( zim2, cmap='gray' )
ax[1,1].text(0.5,-0.08, "LBTI Fizeau at 2.2 microns", size=12, ha="center",
        transform=ax[1,1].transAxes)

zim3 = np.zeros([zwid*2,zwid*2])
# zim3 = im[midp-zwid+yf:midp+zwid-1+yf,midp-zwid+xf:midp+zwid-1+xf]
zim3 = im[int(midp-zwid+yf):int(midp+zwid-1+yf),
          int(midp-zwid+xf):int(midp+zwid-1+xf)]
ax[0,1].imshow( zim3, cmap='gray' )
ax[0,1].text(0.5,-0.08, "Spacecraft image", size=12, ha="center",
        transform=ax[0,1].transAxes)

zim4 = np.zeros([zwid*2,zwid*2])
zim4 = svRim[int(midp-zwid+yf):int(midp+zwid-1+yf),
              int(midp-zwid+xf):int(midp+zwid-1+xf)]
ax[1,0].imshow( zim4, cmap='gray' )
ax[1,0].text(0.5,-0.08, "SHARK-VIS at 0.5 microns", size=12, ha="center",
        transform=ax[1,0].transAxes)

#

plt.show()
