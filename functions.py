import time
import skimage
import math

import cupy as cp
import numpy as np
import imageio.v3 as iio

from tqdm import tqdm
from pathlib import Path
from scipy.special import j1
from cupyx.scipy.signal import fftconvolve
from skimage.filters import window as d2_win





def load_video_to_gpu(video_path):

    data = list(iter_video(video_path))
    padded_video = cp.array(data)
    return padded_video

def load_video_to_cpu(video_path):

    data = list(iter_video(video_path))
    padded_video = np.array(data)
    return padded_video

def iter_video(video_path):    
    for x in iio.imiter(video_path):
        yield (skimage.color.rgb2gray(x)*255) 

#400 iterations might not be enough for very small lambdas
def Chambole_pock_TV(u_0,lambd,num_iter=400,use_tqdm =False):
    u_0 = u_0.astype('float64')
    [nt,ny,nx] = u_0.shape
    L=np.sqrt(8)  #operatornorm of nabla
    tau=1/L
    sigma=1/L
    gamma=0.7*lambd 
    
    #notation as in Chambole Pock 2010 paper: y_new:=y^n+1   u_new:=u^n+1   u=u^n y=y^n  u-bar=x^bar^n+1 
    #     
    u=u_0.copy()
    u_bar=cp.zeros((nt+1,ny,nx))
    u_bar[1:,:,:]=u_0.copy()
    u_bar[0,:,:]=u_0[0,:,:]
    u_new=cp.zeros(u.shape)
    y = cp.zeros((nt+1,ny,nx))        
    
        
    for step in tqdm(range(num_iter),disable =not(use_tqdm)):             
            
        #calculate nable u_bar and store in u_bar   
        u_bar[1:,:,:] -= u_bar[:-1,:,:] 
    
        #scale for dual gradient step
        u_bar *= sigma
    
        #make dual gradient step
        y[:-1,:,:] +=(u_bar[1:,:,:])
    
        #project onto unit square
        y[:-1,:,:]= cp.minimum(cp.maximum(y[:-1,:,:],-1),1) 
    
        #primal gradient step
    
        u_new[1:-1,:,:]=((u[1:-1,:,:]+tau*(y[2:-1,:,:]- y[1:-2,:,:]) ) + tau*lambd*u_0[1:-1,:,:])/(1+tau*lambd)   
        u_new[0,:,:]=((u[0,:,:]+tau*(y[1,:,:]- y[0,:,:]) ) + tau*lambd*u_0[0,:,:])/(1+tau*lambd)   
        u_new[-1,:,:]=((u[-1,:,:]+tau*(y[-1,:,:]- y[-2,:,:]) ) + tau*lambd*u_0[-1,:,:])/(1+tau*lambd)  
        
        theta=1/math.sqrt(1+2*gamma*tau)
        tau=theta*tau    
        sigma=sigma/theta
    
        #u_bar update, calculated in u since we will assign u later on anyway
        u -= u_new
        u *= theta   
        u_bar[1:,:,:]=u_new -u 
        u_bar[0,:,:]=u_bar[1,:,:]
    
        #copy u
        u=u_new.copy()

        
    return u

def remove_honey_comb(video,kernel_size = 17,num_samples_for_peak_estimation = 200):
    nt,_,_ = video.shape    
    sample_frames_for_peak_estimation = [video[frame_id,:,:].get() for frame_id in np.linspace(0,nt-1,num_samples_for_peak_estimation,dtype = int)]
    new_cut_off = find_new_frequency(sample_frames_for_peak_estimation)
    win=d2_win(window_type='hamming',shape=[kernel_size,kernel_size])
    circular_filter=cp.array(circular_low_pass(new_cut_off, kernel_size ) * win)[None, :, :]
    filtered = fftconvolve(
        video,
        circular_filter,
        mode="same",
        axes=(1, 2)   
    )
    
    return frame_wise_hist_eq(filtered)

def find_new_frequency(frames):   
    #crop image to square so the math behind the peak estimation becomes simpler
    n = min(frames[0].shape)
    n = n -1 + (n % 2)   
    pds=[]
    for im in frames:
        pds.append(np.abs(np.fft.fftshift(np.fft.fft2(im[:n,:n]))))
    pds=np.array(pds)
    spectral_mean = pds.mean(axis=0)

    X=np.array(list(range(-n//2,n//2))   )
    YY, XX = np.meshgrid(X, X)
    Z=np.sqrt( XX**2 + (YY)**2)
    li=[]
    for r in range(n//2 +1 ):
        ring=np.abs((Z-r))<1   
        if len(spectral_mean[ring])==0 :
            m=1
        else:
            m=spectral_mean[ring].mean()
        li.append(m)

    #adda s afety margin, as we do not want to accidentally pick the center of the hexagon as a peak
    safety_margin = int(0.15 * n) 
    radius=np.argmax(li[safety_margin:])+safety_margin
    relative_radius = 2* radius /n 
    return relative_radius*np.pi /(2)

#implementation of the filter taken from Fat32's response at:  https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
def circular_low_pass(omega_c, kernel_size ):  # omega = cutoff frequency in radians (pi is max), kernel_size  = horizontal size of the kernel, also its vertical size.
  with np.errstate(divide='ignore',invalid='ignore'):
    kernel = np.fromfunction(lambda x, y: omega_c*j1(omega_c*np.sqrt((x - (kernel_size  - 1)/2)**2 + (y - (kernel_size  - 1)/2)**2))/(2*np.pi*np.sqrt((x - (kernel_size  - 1)/2)**2 + (y - (kernel_size  - 1)/2)**2)), [kernel_size , kernel_size ])
  if kernel_size  % 2:
    kernel[(kernel_size  - 1)//2, (kernel_size  - 1)//2] = omega_c**2/(4*np.pi)
  return kernel


def frame_wise_hist_eq(video, nbins=256):
    T, _, _ = video.shape
    out = cp.empty_like(video)

    for t in range(T):
        frame = video[t]

        hist, _ = cp.histogram(frame, bins=nbins, range=(0, 255))
        cdf = cp.cumsum(hist, dtype=cp.float32)
        cdf /= cdf[-1]

        idx = frame.astype(cp.int32)
        idx = cp.clip(idx, 0, 255)

        out[t] = cdf[idx] * 255.0

    return out
  

def write_to_file(video,filename,fps=30,codec="rawvideo"):

    if video.dtype != cp.uint8:
        video = video.astype(cp.uint8, copy=False)

    video = cp.asnumpy(video) if isinstance(video, cp.ndarray) else np.asarray(video)
    video = np.ascontiguousarray(video)


    with iio.imopen(filename, "w", plugin="pyav") as out:
        out.init_video_stream(codec=codec, fps=fps)       
        for frame in video:
            out.write_frame(frame,pixel_format = 'gray')

#a usefull helper function for partitioning a video, in case the GPU is to small
def get_particion_idxs(arr_length,num_slices):
    length,remainder= divmod(arr_length,num_slices)
    start_idx =  [j *length + min(j,remainder) for j in range(num_slices +1)]
    return (list(zip(start_idx,start_idx[1:])))