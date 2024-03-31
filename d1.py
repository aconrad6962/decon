#
#  Code to convolve Vesta data
#
#
#  ARC - 15nov2018
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
eeWid = 15     # edge effect width
yPadTop  = 5000000  # to center plots
yPadBot  = 10000
lbtiErr  = 2000000
jwstErr  = 500000
fname = './FC21A0002835_11191050436F4N.FIT'
width    = 1024


nr=2
nc=2
fig, ax = plt.subplots(nrows=nr,ncols=nc)

# loh - use:  img = iio.imread("g4g.png")
im = fits.getdata( fname )

for i in range(0,nr):
  for j in range(0,nc):
    ax[i,j].set_xticks([])
    ax[i,j].set_yticks([])

#
# Devise JWST or LBTI psf from 1st pncpls: 
#
#   0. (Guess at subtended arcseconds of sci foV = foVst)
#   1. plateScale = imageWidth /foVst  (make image square first if nccssry)
#   2. Define dlim = 4 mas for GMT GMagAO-X (at 0.5 microns from j males talk)
#   3. Gen the BMT psf (??)
#       a. Find any airy patern = apat
#       b. psfPixWid = plateScale * dlim
#
#   Vesta is about 0.7 arcsec and 160 out of 1024 pix
#   so FoV is 1024/160*.7 = 4.5 arcsec
#
#
foVst         = 4500   # milliarsec
gmtDlim       = 4    # for GMT GMagAO-X (at 0.5 microns from j males talk)
gemDlim       = gmtDlim * 2.2/0.5 * 26 / 8  # Gemini at K
gemDlim_R     = gmtDlim           * 26 / 8  # Gemini at K

plateScale = im.shape[0] / foVst   #  chk if square

gmtPsfPixWid  = plateScale * gmtDlim
radius = gmtPsfPixWid       # Size in pixels of Airy pattern (to first null)
psf = AiryDisk2DKernel(radius)
gmtRim = convolve_fft(im, psf )

gemPsfPixWid  = plateScale * gemDlim
radius = gemPsfPixWid       # Size in pixels of Airy pattern (to first null)
psf = AiryDisk2DKernel(radius)
gemKim = convolve_fft(im, psf )

gemPsfPixWid_R  = plateScale * gemDlim_R
radius = gemPsfPixWid_R       # Size in pixels of Airy pattern (to first null)
psf = AiryDisk2DKernel(radius)
gemRim = convolve_fft(im, psf )

print( "plateScale = " + "%lf" % plateScale + " pixel/mas" )
print( "gmtDlim = " + "%lf" % gmtDlim )
print( "gmtPsfPixWid = " + "%lf" % gmtPsfPixWid + " = plateScale * gmtDlim" )

#
#  Plots start here
#
midp = width/2
yf=-40    #  yfudge
xf=+10
zwid = 100

zim1 = np.zeros([zwid*2,zwid*2])
# print( "err val = ", midp-zwid+yf )
zim1 = gmtRim[int(midp-zwid+yf):int(midp+zwid-1+yf), \
              int(midp-zwid+xf):int(midp+zwid-1+xf)]
ax[0,0].imshow( zim1, cmap='gray' )
ax[0,0].text(0.5,-0.08, "(GMT at 0.5 microns)", size=12, ha="center",
        transform=ax[0,0].transAxes)

zim2 = np.zeros([zwid*2,zwid*2])
zim2 = gemKim[int(midp-zwid+yf):int(midp+zwid-1+yf),
              int(midp-zwid+xf):int(midp+zwid-1+xf)]
ax[1,1].imshow( zim2, cmap='gray' )
ax[1,1].text(0.5,-0.08, "(Gem at 2.2 microns)", size=12, ha="center",
        transform=ax[1,1].transAxes)

zim3 = np.zeros([zwid*2,zwid*2])
# zim3 = im[midp-zwid+yf:midp+zwid-1+yf,midp-zwid+xf:midp+zwid-1+xf]
zim3 = im[int(midp-zwid+yf):int(midp+zwid-1+yf),
          int(midp-zwid+xf):int(midp+zwid-1+xf)]
ax[0,1].imshow( zim3, cmap='gray' )
ax[0,1].text(0.5,-0.08, "Spacecraft image)", size=12, ha="center",
        transform=ax[0,1].transAxes)

zim4 = np.zeros([zwid*2,zwid*2])
# zim4 = gemRim[midp-zwid+yf:midp+zwid-1+yf,midp-zwid+xf:midp+zwid-1+xf]
zim4 = gemRim[int(midp-zwid+yf):int(midp+zwid-1+yf),
              int(midp-zwid+xf):int(midp+zwid-1+xf)]
ax[1,0].imshow( zim4, cmap='gray' )
ax[1,0].text(0.5,-0.08, "Gem at 0.5 microns)", size=12, ha="center",
        transform=ax[1,0].transAxes)

#

plt.show()
