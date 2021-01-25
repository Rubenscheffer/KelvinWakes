from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon, rescale
from skimage.data import shepp_logan_phantom

pathfile = r'C://Users/Ruben/Documents/Thesis/Data/Imagetest/'

fp = pathfile + '20201228_033335_48_2424_3B_AnalyticMS_clip.tiff'

image = gdal.Open(fp, gdal.GA_ReadOnly)
band = image.GetRasterBand(4)
arr = band.ReadAsArray()

image = rescale(arr, scale=0.4, mode='reflect', multichannel=False)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)

sinogram = radon(image, theta=theta, circle=True)
# Load specific band (1-4)

plt.style.use('fast')

dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]

plt.figure()
plt.imshow(image)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(sinogram, cmap=plt.cm.Greys_r,
            extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
            aspect='auto')
plt.xlabel('Angle (degrees)')
plt.show()

#%% Test Image


test_image = shepp_logan_phantom()
test_image = rescale(test_image, scale=0.4, mode='reflect', multichannel=False)
theta = np.linspace(0., 180., max(test_image.shape), endpoint=False)
test_sinogram = radon(test_image, theta=theta, circle=True)
dx, dy = 0.5 * 180.0 / max(test_image.shape), 0.5 / test_sinogram.shape[0]

plt.figure()
plt.imshow(test_sinogram, cmap=plt.cm.Greys_r,
            extent=(-dx, 180.0 + dx, -dy, test_sinogram.shape[0] + dy),
            aspect='auto')
plt.show()
