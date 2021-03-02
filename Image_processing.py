from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon, rescale
from skimage.filters import threshold_otsu
from scipy import ndimage as nd


def sobel_angle(img, theta):
    dx = nd.sobel(img, 0)  #horizontal Sobel filter
    dy = nd.sobel(img, 1)  #vertical Sobel
    theta = theta * np.pi / 180
    image = np.cos(theta) * dx + np.sin(theta) * dy
    return image


pathfile = r'C://Users/Ruben/Documents/Thesis/Data/Angles1/RawImages/'

fp = pathfile + 'Schip8.tif'

image = gdal.Open(fp, gdal.GA_ReadOnly)

#Average of RGB bands

r_band = image.GetRasterBand(1).ReadAsArray()/2**14
g_band = image.GetRasterBand(2).ReadAsArray()/2**14
b_band = image.GetRasterBand(3).ReadAsArray()/2**14

arr = (r_band + g_band + b_band) / 3


plt.style.use('fast')
plt.figure()
plt.imshow(arr)

plt.show()

image = rescale(arr, scale=1, anti_aliasing=True,
                mode='reflect', multichannel=False)

median = np.median(image)


#%% Filter using otsu

otsu_threshold = threshold_otsu(image, 2)
# image[image > (otsu_threshold - 500)] = 0

im_max = np.max(arr)
im_min = np.min(arr)
print(im_max, im_min)


#Otsu threshold picture
plt.style.use('fast')
plt.figure()
plt.imshow(image)
plt.colorbar()
# plt.savefig(r'C://Users/Ruben/Documents/Thesis/Data/Angles1/Images/Schip8filtered.png')
plt.show()

#Show RGB picture in stead of average over bands

rgb = np.dstack((r_band, g_band, b_band))

#RGB picture
plt.figure()
plt.imshow(rgb)
plt.colorbar()
plt.show()


# #%%Preprocessing

# # Filters

# dx = nd.sobel(image, 0)  #horizontal Sobel filter
# dy = nd.sobel(image, 1)  #vertical Sobel filter
# theta_sobel = 0
# dxdy = sobel_angle(image, theta_sobel)


# #Overview picture
# # fig = plt.figure()


# # ax1 = fig.add_subplot(221)
# # col1 = ax1.imshow(image)
# # # plt.colorbar(col1)

# # ax2 = fig.add_subplot(222)
# # col2 = ax2.imshow(dy)
# # # plt.colorbar(col2)

# # ax3 = fig.add_subplot(223)
# # col3 = ax3.imshow(dx)
# # # plt.colorbar(col3)

# # ax4 = fig.add_subplot(224)
# # col4 = ax4.imshow(dxdy)
# # # plt.colorbar(col4)


# # ax1.set_title("Unfiltered Image")
# # ax2.set_title("Horizontal Sobel Filter")
# # ax3.set_title("Vertical Sobel Filter")
# # ax4.set_title(f"Sobel Filter under angle of {theta_sobel} degrees")

# # plt.show()

# # Radon transform

# theta = np.linspace(0., 180., max(image.shape), endpoint=False)

# sinogram = radon(dxdy, theta=theta, circle=True)
# # Load specific band (1-4)

# plt.style.use('fast')

# dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]

# #Base picture
# plt.figure()
# plt.title(f"Sobel filter under an angle of {theta_sobel} degrees")
# plt.imshow(dxdy)
# plt.colorbar()
# plt.show()


# #Sinogram
# # plt.figure()
# # plt.imshow(sinogram, cmap=plt.cm.Greys_r,
# #             extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
# #             aspect='auto')
# # plt.xlabel('Angle (degrees)')
# # plt.ylabel('y')
# # plt.colorbar()
# # plt.show()
