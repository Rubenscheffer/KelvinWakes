# generic tools
import glob
import os
from xml.etree import ElementTree

# geospatial libraries
from osgeo import gdal, osr

# raster/image libraries
import numpy as np
from scipy import ndimage 
from scipy.interpolate import griddata
from PIL import Image, ImageDraw

# plotting libraries
import matplotlib.pyplot as plt 

# pre-processing functions
def get_array_from_xml(treeStruc):
    """
    Arrays within a xml structure are given line per line
    Output is an array
    """
    for i in range(0, len(treeStruc)):
        Trow = [float(s) for s in treeStruc[i].text.split(' ')]
        if i == 0:
            Tn = Trow
        elif i == 1:
            Trow = np.stack((Trow, Trow), 0)
            Tn = np.stack((Tn, Trow[1, :]), 0)
        else:
            # Trow = np.stack((Trow, Trow),0)
            # Tn = np.concatenate((Tn, [Trow[1,:]]),0)
            Tn = np.concatenate((Tn, [Trow]), 0)
    return Tn

def get_S2_image_locations(fname):
    """
    The Sentinel-2 imagery are placed within a folder structure, where one 
    folder has an ever changing name, when this function finds the path from
    the meta data
    """
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()
    granule_list = root[0][0][11][0][0]
    
    im_paths = []
    for im_loc in root.iter('IMAGE_FILE'):
        im_paths.append(im_loc.text)
    #    print(im_loc.text)
    return im_paths

def meta_S2string(S2str):  # generic
    """
    get meta data of the Sentinel-2 file name
    input:   S2str          string            filename of the L1C data
    output:  S2time         string            date "YYYYMMDD"
             S2orbit        string            relative orbit "RXXX"
             S2tile         string            tile code "TXXXXX"
    """
    S2split = S2str.split('_')
    S2time = S2split[2][0:8]
    S2orbit = S2split[4]
    S2tile = S2split[5]
    return S2time, S2orbit, S2tile

def read_band_s2(fname):
    """
    This function takes as input the Sentinel-2 band name and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array
    input:   fname          string            path of the folder and its filename
    output:  data           array (n x m)     array of the band image
             spatialRef     string            projection
             geoTransform   tuple             affine transformation
                                              coefficients
             targetprj                        spatial reference
    """
    img = gdal.Open(glob.glob(fname)[0])
    data = np.array(img.GetRasterBand(1).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj

def read_sun_angles_s2(path):
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts an
    array with sun angles, as these vary along the scene.
    input:   path           string            path where xml-file of
                                              Sentinel-2 is situated
    output:  Zn             array (n x m)     array of the Zenith angles
             Az             array (n x m)     array of the Azimtuh angles

    """
    fname = os.path.join(path, 'MTD_TL.xml')
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res == 10:  # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)

    # get Zenith array
    Zenith = root[1][1][0][0][2]
    Zn = get_array_from_xml(Zenith)
    znSpac = float(root[1][1][0][0][0].text)
    znSpac = np.stack((znSpac, float(root[1][1][0][0][1].text)), 0)
    znSpac = np.divide(znSpac, 10)  # transform from meters to pixels

    zi = np.linspace(0 - 10, mI + 10, np.size(Zn, axis=0))
    zj = np.linspace(0 - 10, nI + 10, np.size(Zn, axis=1))
    Zi, Zj = np.meshgrid(zi, zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)
    del zi, zj, Zi, Zj, znSpac

    iGrd = np.arange(0, mI)
    jGrd = np.arange(0, nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)

    Zn = griddata(Zij, Zn.reshape(-1), (Igrd, Jgrd), method="linear")

    # get Azimuth array
    Azimuth = root[1][1][0][1][2]
    Az = get_array_from_xml(Azimuth)
    azSpac = float(root[1][1][0][1][0].text)
    azSpac = np.stack((azSpac, float(root[1][1][0][1][1].text)), 0)

    ai = np.linspace(0 - 10, mI + 10, np.size(Az, axis=0))
    aj = np.linspace(0 - 10, nI + 10, np.size(Az, axis=1))
    Ai, Aj = np.meshgrid(ai, aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)
    del ai, aj, Ai, Aj, azSpac

    Az = griddata(Aij, Az.reshape(-1), (Igrd, Jgrd), method="linear")
    del Igrd, Jgrd, Zij, Aij
    return Zn, Az

def read_detector_mask(path_meta, msk_dim, boi, geoTransform):   
    det_stack = np.zeros(msk_dim, dtype='int8')    
    for i in range(len(boi)):
        im_id = boi[i]
        f_meta = os.path.join(path_meta, 'MSK_DETFOO_B'+ f'{im_id:02.0f}' + '.gml')
        dom = ElementTree.parse(glob.glob(f_meta)[0])
        root = dom.getroot()  
        
        mask_members = root[2]
        for k in range(len(mask_members)):
            # get detector number from meta-data
            det_id = mask_members[k].attrib
            det_id = list(det_id.items())[0][1].split('-')[2]
            det_num = int(det_id)
        
            # get footprint
            pos_dim = mask_members[k][1][0][0][0][0].attrib
            pos_dim = int(list(pos_dim.items())[0][1])
            pos_list = mask_members[k][1][0][0][0][0].text
            pos_row = [float(s) for s in pos_list.split(' ')]
            pos_arr = np.array(pos_row).reshape((int(len(pos_row)/pos_dim), pos_dim))
            
            # transform to image coordinates
            i_arr, j_arr = map2pix(geoTransform, pos_arr[:,0], pos_arr[:,1])
            ij_arr = np.hstack((j_arr[:,np.newaxis], i_arr[:,np.newaxis]))
            # make mask
            msk = Image.new("L", [np.size(det_stack,0), np.size(det_stack,1)], 0)
            ImageDraw.Draw(msk).polygon(tuple(map(tuple, ij_arr[:,0:2])), \
                                        outline=det_num, fill=det_num)
            msk = np.array(msk)    
            det_stack[:,:,i] = np.maximum(det_stack[:,:,i], msk)
    return det_stack

def read_cloud_mask(path_meta, msk_dim, geoTransform):   
    msk_clouds = np.zeros(msk_dim, dtype='int8')    

    f_meta = os.path.join(path_meta, 'MSK_CLOUDS_B00.gml')
    dom = ElementTree.parse(glob.glob(f_meta)[0])
    root = dom.getroot()  
    
    mask_members = root[2]
    for k in range(len(mask_members)):    
        # get footprint
        pos_dim = mask_members[k][1][0][0][0][0].attrib
        pos_dim = int(list(pos_dim.items())[0][1])
        pos_list = mask_members[k][1][0][0][0][0].text
        pos_row = [float(s) for s in pos_list.split(' ')]
        pos_arr = np.array(pos_row).reshape((int(len(pos_row)/pos_dim), pos_dim))
        
        # transform to image coordinates
        i_arr, j_arr = map2pix(geoTransform, pos_arr[:,0], pos_arr[:,1])
        ij_arr = np.hstack((j_arr[:,np.newaxis], i_arr[:,np.newaxis]))
        # make mask
        msk = Image.new("L", [msk_dim[0], msk_dim[1]], 0)
        ImageDraw.Draw(msk).polygon(tuple(map(tuple, ij_arr[:,0:2])), \
                                    outline=1, fill=1)
        msk = np.array(msk)    
        msk_clouds = np.maximum(msk_clouds, msk)
    return msk_clouds

def read_sensor_angles_s2(path):
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts an
    array with sensor angles, as these vary along the scene and create ramps.
    input:   path           string            path where xml-file of
                                              Sentinel-2 is situated
    output:  Zn             array (n x m)     array of the Zenith angles
             Az             array (n x m)     array of the Azimtuh angles

    """
    fname = os.path.join(path, 'MTD_TL.xml')
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res == 10:  # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)

    # get Zenith array
    Zenith = root[1][1][0][0][2]
    Zn = get_array_from_xml(Zenith)
    znSpac = float(root[1][1][0][0][0].text)
    znSpac = np.stack((znSpac, float(root[1][1][0][0][1].text)), 0)
    znSpac = np.divide(znSpac, 10)  # transform from meters to pixels

    zi = np.linspace(0 - 10, mI + 10, np.size(Zn, axis=0))
    zj = np.linspace(0 - 10, nI + 10, np.size(Zn, axis=1))
    Zi, Zj = np.meshgrid(zi, zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)
    del zi, zj, Zi, Zj, znSpac

    iGrd = np.arange(0, mI)
    jGrd = np.arange(0, nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)

    Zn = griddata(Zij, Zn.reshape(-1), (Igrd, Jgrd), method="linear")

    # get Azimuth array
    Azimuth = root[1][1][0][1][2]
    Az = get_array_from_xml(Azimuth)
    azSpac = float(root[1][1][0][1][0].text)
    azSpac = np.stack((azSpac, float(root[1][1][0][1][1].text)), 0)

    ai = np.linspace(0 - 10, mI + 10, np.size(Az, axis=0))
    aj = np.linspace(0 - 10, nI + 10, np.size(Az, axis=1))
    Ai, Aj = np.meshgrid(ai, aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)
    del ai, aj, Ai, Aj, azSpac

    Az = griddata(Aij, Az.reshape(-1), (Igrd, Jgrd), method="linear")
    del Igrd, Jgrd, Zij, Aij
    return Zn, Az

def mat_to_gray(I, notI):  # pre-processing or generic
    """
    Transform matix to float, omitting nodata values
    input:   I              array (n x m)     matrix of integers with data
             notI           array (n x m)     matrix of boolean with nodata
    output:  Inew           array (m x m)     linear transformed floating point
                                              [0...1]
    """
    yesI = ~notI
    Inew = np.float64(I)  # /2**16
    Inew[yesI] = np.interp(Inew[yesI],
                           (Inew[yesI].min(),
                            Inew[yesI].max()), (0, +1))
    Inew[notI] = 0
    return Inew

# mapping tools
def map2pix(geoTransform, x, y):  # generic
    """
    Transform map coordinates to image coordinates
    input:   geoTransform   array (1 x 6)     georeference transform of
                                              an image
             x              array (n x 1)     map coordinates
             y              array (n x 1)     map coordinates
    output:  i              array (n x 1)     row coordinates in image space
             j              array (n x 1)     column coordinates in image space
    """
    j = x - geoTransform[0]
    i = y - geoTransform[3]

    if geoTransform[2] == 0:
        j = j / geoTransform[1]
    else:
        j = (j / geoTransform[1]
             + i / geoTransform[2])

    if geoTransform[4] == 0:
        i = i / geoTransform[5]
    else:
        i = (j / geoTransform[4]
             + i / geoTransform[5])

    return i, j

def pix2map(geoTransform, i, j):  # generic
    """
    Transform image coordinates to map coordinates
    input:   geoTransform   array (1 x 6)     georeference transform of
                                              an image
             i              array (n x 1)     row coordinates in image space
             j              array (n x 1)     column coordinates in image space
    output:  x              array (n x 1)     map coordinates
             y              array (n x 1)     map coordinates
    """
    x = (geoTransform[0]
         + geoTransform[1] * j
         + geoTransform[2] * i
         )
    y = (geoTransform[3]
         + geoTransform[4] * j
         + geoTransform[5] * i
         )
    return x, y

# image matching functions
def prepare_grids(im_stack, ds):
    '''
    prepare_grids
    the image stack is sampled by a template, that samples without overlap. all
    templates need to be of the same size, thus the image stack needs to be 
    enlarged if the stack is not a multitude of the template size
    input:   im_stack array (n x m x k) imagery array
             ds       integer           size of the template
    output:  im_stack array (n x m x k) imagery array
             I_ul     array (p x q)     row-coordinate of the upper-left pixel
                                        of the template
             J_ul     array (p x q)     collumn-coordinate of the upper-left 
                                        pixel of the template
    '''
    # padding is needed to let all imagery be of the correct template size
    i_pad = np.int(np.ceil(im_stack.shape[0]/ds)*ds - im_stack.shape[0])
    j_pad = np.int(np.ceil(im_stack.shape[1]/ds)*ds - im_stack.shape[1])
    im_stack = np.pad(im_stack, \
                      ((0, i_pad), (0, j_pad), (0, 0)), \
                      'constant', constant_values=(0, 0))
    
    # ul
    i_samp = np.arange(0,im_stack.shape[0]-ds,ds)
    j_samp = np.arange(0,im_stack.shape[1]-ds,ds)
    J_ul, I_ul = np.meshgrid(j_samp,i_samp)
    return (im_stack, I_ul, J_ul)

def raised_cosine(I, beta=0.35):
    '''
    input:   I       array (n x m)     image template
             beta    float             roll-off factor
    output:  W       array (n x m)     weighting mask
    '''
    
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    # filter formulation 
    Hamm = np.cos( (np.pi/(2*beta)) * (R - (.5-beta)))**2
    selec = np.logical_and((.5 - beta) <= R , R<=.5)
    
    # compose filter
    W = np.zeros((m,n))
    W[(.5 - beta) > R] = 1
    W[selec] = Hamm[selec]
    return W 

def thresh_masking(S, m=1/1e3, s=10):
    '''
    input:   S       array (n x m)     spectrum
                                       i.e.: S = np.fft.fft2(I)
             m       float             cut-off
             s       integer           kernel size of the median filter
    output:  M       array (n x m)     mask
    '''
    
    Sbar = np.abs(S)
    th = np.max(Sbar)*m
    
    # compose filter
    M = Sbar>th
    M = ndimage.median_filter(M, size=(s,s))
    return M

def get_top(Q, ds=1):

    (subJ,subI) = np.meshgrid(np.linspace(-ds,+ds, 2*ds+1), np.linspace(-ds,+ds, 2*ds+1))    
    #subI = np.flipud(subI)
    #subJ = np.fliplr(subJ)
    subI = subI.ravel()
    subJ = subJ.ravel()
    # transform back to spatial domain
    C = np.real(np.fft.fftshift(np.fft.ifft2(Q)))
    
    # find highest score
    max_corr = np.amax(C)
    snr = max_corr/np.mean(C)
    ij = np.unravel_index(np.argmax(C), C.shape)
    x, y = ij[::-1]
    
    i_range = np.arange(np.ceil(C.shape[0]/2)-C.shape[0],np.ceil(C.shape[0]/2))
    j_range = np.arange(np.ceil(C.shape[1]/2)-C.shape[1],np.ceil(C.shape[1]/2))
    m_int = np.array([ i_range[y], j_range[x]])
    
    # estimate sub-pixel top
    if (np.min((x-ds,y-ds))>=0) & (np.max((x+ds,y+ds))<(C.shape[0]-1)):
        # a neighborhood is needed
        
        idx_mid = np.int(np.floor((2.*ds+1)**2/2))
        
        Csub = C[y-ds:y+ds+1,x-ds:x+ds+1].ravel()
        Csub = Csub - np.mean(np.hstack((Csub[0:idx_mid],Csub[idx_mid+1:])))
        #Csub = Csub - np.min(Csub)
        
        IN = Csub>0

        m0 = np.array([ np.divide(np.sum(subI[IN]*Csub[IN]), np.sum(Csub[IN])) , 
                       np.divide(np.sum(subJ[IN]*Csub[IN]), np.sum(Csub[IN]))])
    else:
        m0 = np.array([ 0, 0], dtype='float64')
    
    m0 += m_int
    return (m0, snr)

def tpss(Q, W, m0, p=1e-4, l=4, j=5, n=3):
    '''
    TPSS two point step size for phase correlation minimization
    input:   Q        array (n x m)     cross spectrum
             m0       array (1 x 2)     initial displacement  
             p        float             closing error threshold
             l        integer           number of refinements in iteration
             j        integer           number of sub routines during an estimation
             n        integer           mask convergence factor
    output:  m        array (1 x 2)     sub-pixel displacement
             snr      float             signal-to-noise
    '''
    
    (m, n) = Q.shape
    fy = 2*np.pi*(np.arange(0,m)-(m/2)) /m
    fx = 2*np.pi*(np.arange(0,n)-(n/2)) /n
        
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    Fx = np.fft.fftshift(Fx)
    Fy = np.fft.fftshift(Fy)

    # initialize
    m = m0;
    W_0 = W;
    
    m_min = m0 + np.array([+.1, +.1])
    C_min = np.random.random(10) + np.random.random(10) * 1j
    
    C_min = 1j*np.sin(Fx*m_min[0] + Fy*m_min[1])
    C_min += np.cos(Fx*m_min[0] + Fy*m_min[1])
    
    QC_min = Q-C_min # np.abs(Q-C_min)
    dXY_min = 2*W_0*(QC_min*np.conjugate(QC_min))
    
    g_min = np.real(np.array([np.nansum(Fx*dXY_min), np.nansum(Fy*dXY_min)]))
    
    for i in range(l):
        k = 1
        while True:
            # main body of iteration
            C = 1j*np.sin(Fx*m[0] + Fy*m[1])
            C += np.cos(Fx*m[0] + Fy*m[1])
            QC = Q-C # np.abs(Q-C)
            dXY = 2*W*(QC*np.conjugate(QC))
            g = np.real(np.array([np.nansum(Fx*dXY), np.nansum(Fy*dXY)]))
            
            # difference
            dm = m - m_min
            # if dm.any()<np.finfo(float).eps:
            #     break
            dg = g - g_min
            alpha = (np.transpose(dm)*dg)/np.sum(dg**2)
            
            if np.all(np.abs(m - m_min)<=p):
                break
            if k>=j:
                break
            
            # update
            m_min = np.copy(m)
            g_min = np.copy(g)
            dXY_min = np.copy(dXY)
            m += alpha*dg
            k += 1
            
        # optimize weighting matrix
        phi = np.abs(QC*np.conjugate(QC))/2
        W = W*(1-(dXY/8))**n
    snr = 1 - (np.sum(phi)/(4*np.sum(W)))
    return (m, snr)

def cosicorr(I1, I2, beta1=0.35, beta2=0.5):
    S1 = np.fft.fft2(I1)
    S2 = np.fft.fft2(I2)
    
    W1 = raised_cosine(I1, beta1)
    W2 = raised_cosine(I2, beta2)
    
    Q = (W1*S1)*np.conj((W2*S2))
    (m0, SNR) = get_top(Q)

    WS = thresh_masking(S1, m=1/1e4)

    Qn = 1j*np.sin(np.angle(Q))
    Qn += np.cos(np.angle(Q))
    Qn[Q==0] = 0
    
    (m, snr) = tpss(Qn, WS, m0)
    return (m, snr, m0, SNR)

# the main processing chain

inter_band = np.array([[7.0, 0.3], 
                       [4.7, 2.6],  
                       [5.2, 2.1],    
                       [5.7, 1.6],  
                       [6.0, 1.3],  
                       [6.2, 1.1],  
                       [6.5, 0.8],  
                       [4.9, 2.4],  
                       [6.8, 0.5],  
                       [7.3, 0.0],  
                       [5.6, 1.8],    
                       [6.2, 1.1],                             
                       [6.8, 0.5]]) # from Yurovskaya, 10.1016/j.rse.2019.111468
inter_band = np.tile(inter_band, 6)

def main():
    # inital administration
    S2name = 'S2B_MSIL1C_20200420T031539_N0209_R118_T48NUG_20200420T065223.SAFE'
    boi = (2,3,4,8) # band of interest
    ds = 2**6
    
    dat_path = os.getcwd()
    
    # get data structure
    fname = os.path.join(dat_path, S2name, 'MTD_MSIL1C.xml')
    im_paths = get_S2_image_locations(fname)
    
    # read imagery data
    for i in range(len(boi)):
        im_id = boi[i]
        f_full = os.path.join(dat_path, S2name, im_paths[im_id-1]+'.jp2')
        im, spatialRef, geoTransform, targetprj = read_band_s2(f_full)
        if i == 0:
            im_stack = im
        else:
            im_stack = np.dstack((im_stack, im))
        del im
    
    # read meta data
    split_path = im_paths[0].split('/') # get location of imagery meta data
    path_meta = os.path.join(dat_path, S2name, split_path[0], split_path[1])
    (sun_zn, sub_az) = read_sun_angles_s2(path_meta)
    
    
    # get sensor configuration
    path_meta = os.path.join(dat_path, S2name, split_path[0], split_path[1], \
                             'QI_DATA')
    msk_dim = np.shape(im_stack)
    det_stack = read_detector_mask(path_meta, msk_dim, boi, geoTransform)    
        
    # get sensor geometry

    # get cloud information
    msk_cloud = read_cloud_mask(path_meta, msk_dim[0:2], geoTransform)
    
    # create grid and estimate velocities
    im_stack, I_ul, J_ul = prepare_grids(im_stack, ds)
    
    UV = np.zeros(I_ul.shape, dtype=complex)
    UV_0 = np.zeros(I_ul.shape, dtype=complex)
    P,C = np.zeros(I_ul.shape), np.zeros(I_ul.shape)
    
    I_ul, J_ul = I_ul.flatten(), J_ul.flatten()
    for i in range(UV.size):
        i_b, i_e = I_ul[i], I_ul[i]+ds # row begining and ending
        j_b, j_e = J_ul[i], J_ul[i]+ds # collumn begining and ending
        
        sub_b1 = im_stack[i_b:i_e,j_b:j_e,0] # image templates
        sub_b2 = im_stack[i_b:i_e,j_b:j_e,2] 
    
        # simple normalization
        sub_b1 = mat_to_gray(sub_b1, sub_b1==0)
        sub_b2 = mat_to_gray(sub_b2, sub_b2==0)
        
        m, snr, m0, SNR = cosicorr(sub_b1, sub_b2)
        
        ij = np.unravel_index(i, UV.shape)
        
        UV_0[ij] = np.complex(m0[0], m0[1]) # sub-pioxel localization
        UV[ij] = np.complex(m[0], m[1])     # sub-pixel refinement
        P[ij], C[ij] = np.real(snr), SNR #  
    
    # some plotting
    fig, ax = plt.subplots()
    im = plt.imshow(np.angle(UV_0))
    #fig.colorbar(im, ax=ax)
    im.set_clim(-np.pi, np.pi)
    plt.show()    
    
    fig, ax = plt.subplots()
    im = plt.imshow(np.real(UV_0))
    #fig.colorbar(im, ax=ax)
    im.set_clim(-1, 1)
    plt.show() 
    
if __name__ == "__main__":
    main()    