"""
Script for analyzing numpy arrays and extracting dF/F from them.
This script takes in a numpy array which is a video file extracted from CaImAn,
and another array which is a mask for each of the neurons.
"""

import os
import numpy as np
import scipy
from scipy import signal
import pyqtgraph as pg
from pyqtgraph import FileDialog
import matplotlib.pyplot as plt
import caiman as cm
import math
import matplotlib
import imageio
from skimage.morphology import dilation, disk, rectangle

def generate_baseline_image(video):
    # This function generates a single baseline average image, by average a
    # video into a single frame
    # Args:
    #   video   : 3D image of dimensions TxYxX to flatten into a YxX image.
    return np.sum(video, 0)//video.shape[0]

def get_mask_centroid(mask):
    # This function gets the coordinates of the centroid (center of mass) of a
    # mask
    # Args:
    #   mask    : 2D array as the mask for which to find the center of mass.
    # Return should be integers as they are pixel coordinates
    maskon  = np.argwhere(mask)
    center  = np.rint(maskon.sum(0)/maskon.shape[0])
    return center.astype(int)

def get_mask_shape(mask):
    # This function gets the shape of the active area of a mask layer by finding
    # the smallest and largest coordinates in each dimension that are True. So
    # it returns a rectangular shape
    # Args:
    #   mask    : 2D array as the mask for which to find the shape
    coords  = np.argwhere(mask)
    width   = np.amax(coords[:,0]) - np.amin(coords[:,0]) + 1
    height  = np.amax(coords[:,1]) - np.amin(coords[:,1]) + 1
    return [width, height]

def extract_neuron_trace_none(video, mask):
    # This function extracts an intensity trace for a neuron, without
    # subtracting any background (i.e. this is just the spatial average of the
    # masked pixels)
    #   video   : 3D video containing neuron
    #   mask    : 2D array masking the neuron we want to extract
    neurons     = apply_mask(video, mask)
    Npix        = np.sum(mask)
    Nvals       = np.zeros(neurons.shape[0])
    for t, frame in enumerate(neurons):
        Nvals[t]= np.sum(frame)//Npix
    return Nvals

def get_mask_rectangle(mask): 
    # This function gets a rectangle that exactly covers the X and Y dims of 
    # the mask in question 
    # Args: 
    #   mask    : 2D array as the mask for which to find the shape
    # Return is a 2D array as the rectangular outline of the mask. 
    maskout     = np.array(mask, copy=True) 
    coords      = np.argwhere(maskout) 
    topleft     = [np.amin(coords[:,0]), np.amin(coords[:,1])] 
    botright    = [np.amax(coords[:,0]), np.amax(coords[:,1])] 
    maskout[topleft[0]:botright[0]+1, topleft[1]:botright[1]+1] = True
    return maskout

def get_background_mask(mask, method = "rectangle", r=10):
    # This function gets the background mask shape of the neuron.
    # Args:
    #   mask    : 2D array as the mask for which to find the local background
    #   method  : "rectangle" gets a rectangle surrounding the input mask, with
    #               dimensions twice the dimensions of the mask on either side.
    #             "disk" uses a disk shaped kernel to calculate dilation
    #               function on the input mask.
    #             "dilation" uses a square shaped kernel to calculate the
    #               dilation function on the input mask.
    #             TODO: add alternative versions that scale by a factor instead
    #               of by a fixed radius.
    #   r       : radius for disk or edge size for rectangle to use for the kernel.
    # Return is a 2D array with the same shape as the mask array, in which the
    # local background is 1 but the input mask and external background is 0.
    if method=="rectangle":
        dims    = get_mask_shape(mask)
        bgmask  = get_mask_rectangle(mask)
        bgmask  = dilation(bgmask, rectangle(dims[0]+1, dims[1]+1))
        return np.logical_xor(bgmask, mask)
    elif method=="disk":
        dkern   = disk(r)
        bgmask  = dilation(mask, dkern)
        return np.logical_xor(bgmask, mask)
    elif method=="dilation":
        dkern   = rectangle(r,r)
        bgmask  = dilation(mask, dkern)
        return np.logical_xor(bgmask, mask)
    else:
        # default to rectangle
        print('Incorrect method specified, defaulting to rectangle')
        dims    = get_mask_shape(mask)
        bgmask  = get_mask_rectangle(mask)
        bgmask  = dilation(bgmask, rectangle(dims[0], dims[1]))
        return np.logical_xor(bgmask, mask)

def apply_mask(video, mask):
    # This function applies a mask to a video file, returning only the pixels
    # of the video which are exposed by the mask. Automatically transposes the
    # mask if the video and mask are different dimensions, and if that also
    # fails it throws an error
    # Args:
    #   video   : 3D array containing video data
    #   mask    : 2D array of bools
    try:
        return np.multiply(video, mask)
    except:
        return np.multiply(video, mask.T)

def extract_neuron_trace_uniform(video, mask, flatmask, method = "rectangle", r=10):
    # This function extracts an intensity trace for a neuron, subtracting the
    # baseline background intensity first with uniform weighting
    #   video   : 3D video containing neuron
    #   mask    : 2D array masking the neuron we want to extract
    #   flatmask: 2D array of masks of all of the detected neurons
    #   method  : Method for the background calculation
    #   r       : radius/edge length for certain values of method.
    # Neuron trace values: apply mask, count number of active pixels (or sum
    # values if it is weighted), then sum each frame and divide to average.
    Nvid    = apply_mask(video, mask)
    Npixs   = np.sum(mask)
    Nvals   = np.sum(Nvid, (1,2))//Npixs
    # Background trce values: As above but using the calculated background
    # masks. Get the background, remove the flatmasks (logical AND of the
    # enabled background with the inverse of the enabled flatmask should give
    # only the spaces which are unique)
    bmask   = get_background_mask(mask, method, r)
    bmask   = np.logical_and(bmask, np.logical_not(flatmask))
    Bvid    = apply_mask(video, bmask)
    Bpixs   = np.sum(bmask)
    Bvals   = np.sum(Bvid, (1,2))//Bpixs
    return Nvals, Bavg

def trace_df(tracein, Nwdw, Nbase):
    # Extracts the baseline and dF of an input trace using a moving window
    # quantile. A window of width Nwdw is swept across trace, and the bottom
    # q percent of data points are used as the baseline. Then, dF is the delta
    # between the input trace and this calculated baseline. The data is
    # extended to the edges by copying such that the baseline has the same
    # length as the input trace.
    # Args:
    #   tracein : Input trace for which to calculate this.
    #   Nwdw    : Window size for baseline calculation
    #   Nbase   : Number of points to use for baseline
    l       = tracein.shape[0]
    base    = np.zeros(l)
    for i in range(l-Nwdw):
        wdw     = tracein[i:i+Nwdw]
        base[i+Nwdw//2] = np.mean(wdw[np.argpartition(wdw, Nbase)[:Nbase]])
    base[0:Nwdw//2] = base[Nwdw//2]
    base[l-Nwdw//2:] = base[l-Nwdw//2-1]
    return np.subtract(tracein, base), base

def trace_df_f(tracein, Bavg):
    # Computes delta-F over F for the input trace and background region. The
    # input trace is asssumed to have already had the background average
    # subtracted out. This algorithm finds delta-F of the input trace using a
    # windowing method (trace_df(tracein, Nwdw, Nbase)), then re-adds the
    # background intensity to the calcualted baseline to get F.
    # Args:
    #   tracein : Input trace for which to calculate this
    #   Bavg    : Background intensity trace
    df, base    = trace_df(tracein, 70, .1)
    f           = np.add(base, Bavg)
    return np.divide(df, f)

def calculate_dff_set(vid, masks):
    # Generates the dF/F curves for a full set of masks for a given video
    # Args:
    #   vid
    #   masks
    flatmask    = flatten_masks(masks)
    traces      = np.empty((masks.shape[0], vid.shape[0]))
    bval        = np.empty((masks.shape[0], vid.shape[0]))
    dff         = np.empty((masks.shape[0], vid.shape[0]))
    for i, mask in enumerate(masks):
        traces[i], bval[i]  = extract_neuron_trace_uniform(vid, mask, flatmask, method = "rectangle", r = 10)
        dff[i]              = trace_df_f(traces[i], bval[i])
    return dff

def argsort_traces(tracearray):
    # Sorts the array of traces by some maximum dF/F. Returns the indices that
    # sorts the array
    # Args:
    #   tracearray  : NxT array, where N is the number of traces and T is time
    maxdff      = np.amax(tracearray, axis = -1)
    indices     = np.argsort(maxdff)
    return indices

def flatten_masks(masks):
    # This function flattens a set of individual neuron maks into a single mask
    # of all of the neurons. Better than a summation method as there may be
    # overlap between neurons.
    # Args:
    #   mask    : 3D array of bools of dimension AxXxY to be flattened into a
    #             2D array of bools of dimension XxY
    maskout = np.copy(masks[0])
    for mask in masks:
        maskout = np.logical_or(maskout, mask)
    return maskout

def mask_joint(maska, maskb, thresh):
    # Generates a set of masks that is the overlap between maska and maskb.
    # The overlap must contain more pixels than thresh to be kept.
    # Args:
    #   maska   : NxXxY array, where N is the number of masks, XxY is the size
    #   maskb   : MxXxY array, where M is the number of masks, XxY is the size
    #   thresh  : Overlap regions with fewer pixels than this are discarded
    flata   = flatten_masks(maska)
    flatb   = flatten_masks(maskb)
    flatov  = np.logical_and(flata, flatb)

    masko   = []
    for mask in maska:
        if np.sum(np.logical_and(mask, flatov)) >= thresh:
            masko.append(mask)
    return np.asarray(masko)

def mask_disjoint(maska, maskb, thresh):
    # Generates a set of masks that is the disjoint set of the two masks. Check
    # for overlap between any mask area in the sets of maska and maskb, and
    # discard when the overlap is greater than thresh pixels.
    # Args:
    #   maska   : NxXxY array, where N is the number of masks, XxY is the size
    #   maskb   : MxXxY array, where M is the number of masks, XxY is the size
    #   thresh  : Overlap regions with more pixels than this are discarded
    flata   = flatten_masks(maska)
    flatb   = flatten_masks(maskb)
    flatov  = np.logical_and(flata, flatb)

    masko   = []
    for mask in np.concatenate([maska, maskb], axis=0):
        if np.sum(np.logical_and(mask, flatov)) < thresh:
            masko.append(mask)
    return np.asarray(masko)

def mask_union(maska, maskb, thresh):
    # Generates a set of masks that is the set union of the two sets of input
    # masks. This is the combination set of the joint and disjoint sets.
    # Args:
    #   maska   : NxXxY array, where N is the number of masks, XxY is the size
    #   maskb   : MxXxY array, where M is the number of masks, XxY is the size
    #   thresh  : Neurons with this much overlap are considered to be the same
    joint   = mask_joint(maska, maskb, thresh)
    djoint  = mask_disjoint(maska, maskb, thresh)
    if not (joint.size>0):
        return djoint
    if not (djoint.size>0):
        return joint
    
    return np.concatenate([joint, djoint], axis=0)

def mask_outline(mask):
    # Generates a new mask that is the outline shape of a given mask.
    # Args:
    #   mask    : Input mask for which to generate the outline
    mask_dx     = np.diff(mask, n=1, axis=0, prepend=0)
    mask_dy     = np.diff(mask, n=1, axis=1, prepend=0)
    return np.logical_or(mask_dx, mask_dy)

def filter_movingaverage(tracein, n):
    # Performs moving average filtering on an input trace, with filter kernel
    # of width N.
    # Args:
    #   tracein : Input trace (1D array)
    #   n       : Filter width
    ret     = np.cumsum(tracein, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def load_masks(path):
    # Wraps the loading function for masks
    return np.transpose(np.load(path+'.npy'), axes=(2,1,0))

def main():

    # Prompt user for directory containing files to be analyzed
    F           = FileDialog()  # Calls Qt backend script to create a file dialog object
    mcdir       = F.getExistingDirectory(caption='Select Motion Corrected Video Directory')
    fvids=[]
    for file in os.listdir(mcdir):
        if file.endswith("_mc.tif"):
            fvids.append(os.path.join(mcdir, file))

    # Set up a variable to determine which sections are lightsheet and which
    # are epi. This is a horrible way to handle it - need to write new code to
    # either automatically determine or prompt user for input.
    # Use 1 for lightsheet and 0 for epi.
    lsepi       = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    # Threshold for non-binary masks to convert to binary
    th = 0.05

    # Load masks, videos, and dF/F curves
    lsmasks = None
    epimasks = None
    lsunion = None
    epiunion = None
    lsvidconcat = None
    epividconcat = None
    lsvidconcattimes = [0]
    epividconcattimes = [0]
    for i, file in enumerate(fvids):
        vid  = cm.load(file)
        mask = load_masks(file)
        if lsepi[i]:
            if lsmasks is None:
                lsmasks = np.empty((0, mask.shape[1], mask.shape[2]))
                lsunion = mask>th
                lsvidconcat = np.empty((0, vid.shape[1], vid.shape[2]))

            lsvidconcat = cm.concatenate([lsvidconcat, vid], axis=0)
            lsmasks = np.concatenate((lsmasks, mask))
            lsunion = mask_union(lsunion, mask>th, 10)
            lsvidconcattimes.append(lsvidconcat.shape[0])
        else:
            if epimasks is None:
                epimasks = np.empty((0, mask.shape[1], mask.shape[2]))
                epiunion = mask>th
                epividconcat = np.empty((0, vid.shape[1], vid.shape[2]))

            epividconcat = cm.concatenate([epividconcat, vid], axis=0)
            epimasks = np.concatenate((epimasks, mask))
            epiunion = mask_union(epiunion, mask>th, 10)
            epividconcattimes.append(epividconcat.shape[0])

    print(epividconcattimes)
    print(epividconcat.shape)

    # Plot out the flattened masks for light-sheet and epi, and count the
    # number of unique detected neurons
    flFig, flAx = plt.subplots(1,2)
    unFig, unAx = plt.subplots(1,2)
    flsunion = np.zeros((lsmasks.shape[1], lsmasks.shape[2]))
    ff = np.zeros((lsmasks.shape[1], lsmasks.shape[2]))
    for lsmask in lsmasks:
        ff = np.add(ff, lsmask>th)
    for unionmask in lsunion:
        flsunion = np.add(flsunion, unionmask)
    flAx[0].imshow(ff)
    unAx[0].imshow(flsunion)
    print('Number of ls neurons: '+ str(lsunion.shape[0]))

    ff = np.zeros((lsmasks.shape[1], lsmasks.shape[2]))
    fepiunion = np.zeros((lsmasks.shape[1], lsmasks.shape[2]))
    for epimask in epimasks:
        ff = np.add(ff, epimask>th)
    for unionmask in epiunion:
        fepiunion = np.add(fepiunion, unionmask)
    flAx[1].imshow(ff)
    unAx[1].imshow(fepiunion)
    print('Number of epi neurons: '+ str(epiunion.shape[0]))

    # Mask operations to create the various sets and then plot them all out
    sharedneurons = mask_joint(lsunion, epiunion, 10)
    lsunique = mask_disjoint(sharedneurons, lsunion, 10)
    epunique = mask_disjoint(sharedneurons, epiunion, 10)
    allFig, allAx =plt.subplots(1,3)
    allAx[0].imshow(np.sum(lsunique, axis=0))
    allAx[1].imshow(np.sum(sharedneurons, axis=0))
    allAx[2].imshow(np.sum(epunique, axis=0))
    print('Number of unique-to-ls neurons: ' + str(lsunique.shape[0]))
    print('Number of unique-to-epi neurons: ' + str(epunique.shape[0]))
    print('Number of shared neurons: ' + str(sharedneurons.shape[0]))
    
    lsallmasks= mask_union(sharedneurons, lsunique, 10)
    epallmasks= mask_union(sharedneurons, epunique, 10)
    # Plot out df/F traces, custom calculated, for 'zz' number of elements
    #zz=-1
    #lslsdff = calculate_dff_set(lsvidconcat, lsallmasks)
    #lsepdff = calculate_dff_set(lsvidconcat, epunique[0:zz])
    #epepdff = calculate_dff_set(epvidconcat, epallmasks)
    #eplsdff = calculate_dff_set(epvidconcat, lsunique[0:zz])
    
    lslsdff = np.empty((lsallmasks.shape[0], 0))
    for i, el in enumerate(lsvidconcattimes):
        if not i==0:
            start = lsvidconcattimes[i-1]+1
            end = lsvidconcattimes[i]
            lslsdff = np.concatenate((lslsdff, calculate_dff_set(lsvidconcat[start:end], lsallmasks)), axis=1)
            print(lslsdff.shape)
    lslsdff = np.clip(lslsdff, 0, None)
    
    epepdff = np.empty((epallmasks.shape[0], 0))
    for i, el in enumerate(epividconcattimes):
        if not i==0:
            start = epividconcattimes[i-1]+1
            end = epividconcattimes[i]
            epepdff = np.concatenate((epepdff, calculate_dff_set(epividconcat[start:end], epallmasks)), axis=1)
            print(epepdff.shape)
    epepdff = np.clip(epepdff, 0, None)
    
    lspeakmax = np.zeros(lslsdff.shape[0])
    lspeakcount = np.zeros(lslsdff.shape[0])
    for i, lsdff in enumerate(lslsdff):
        peaks,props = scipy.signal.find_peaks(lsdff, distance=10, prominence=(0.05, None), width=(3,None), height=(0.1, None))
        lspeakmax[i] = max(lsdff)
        if peaks.size>0:
            lspeakcount[i] = peaks.size
    
    eppeakmax = np.zeros(epepdff.shape[0])
    eppeakcount = np.zeros(epepdff.shape[0])
    for i, epdff in enumerate(epepdff):
        peaks,props = scipy.signal.find_peaks(epdff, distance=10, prominence=(0.05, None), width=(3,None), height=(0.1, None))
        eppeakmax[i] = max(epdff)
        if peaks.size>0:
            eppeakcount[i] = peaks.size
    
    toplscount_idxs = np.argsort(lspeakcount)
    toplspeak_idxs = np.argsort(lspeakmax)
    topls_idxs = np.concatenate((toplscount_idxs[-4:], toplspeak_idxs[-2:]))
    topsixlscount = lslsdff[topls_idxs]

    topepcount_idxs = np.argsort(eppeakcount)
    topeppeak_idxs = np.argsort(eppeakmax)
    topep_idxs = np.concatenate((topepcount_idxs[-4:], topeppeak_idxs[-2:]))
    topsixepcount = epepdff[topep_idxs]
    
    lsmaxheatmap = np.zeros(lsallmasks[0].shape)
    lscountheatmap = np.zeros(lsallmasks[0].shape)
    for i, mask in enumerate(lsallmasks):
        lsmaxheatmap = np.add(lsmaxheatmap, mask*lspeakmax[i])
        lscountheatmap = np.add(lscountheatmap, mask*lspeakcount[i])
    
    epmaxheatmap = np.zeros(epallmasks[0].shape)
    epcountheatmap = np.zeros(epallmasks[0].shape)
    for i, mask in enumerate(epallmasks):
        epmaxheatmap = np.add(epmaxheatmap, mask*eppeakmax[i])
        epcountheatmap = np.add(epcountheatmap, mask*eppeakcount[i])
    
    lsrankedheatmap = np.zeros(lsallmasks[0].shape)
    for i, idx in enumerate(topls_idxs):
        lsrankedheatmap = np.add(lsrankedheatmap, lsallmasks[idx]*2)
    eprankedheatmap = np.zeros(epallmasks[0].shape)
    for i, idx in enumerate(topep_idxs):
        eprankedheatmap = np.add(eprankedheatmap, epallmasks[idx]*2)
    imageio.imwrite('ls_topmasks.png',lsrankedheatmap)
    imageio.imwrite('epi_topmasks.png',eprankedheatmap)
    ff,ax = plt.subplots()
    ax.imshow(lsrankedheatmap)
    ff, ax = plt.subplots()
    ax.imshow(eprankedheatmap)
    
    # Setting up plot information
    cmap = plt.get_cmap('jet')
    # Light-sheet, maximum dF/F figure
    ff,ax = plt.subplots()
    ax.imshow(lsmaxheatmap, cmap='jet')
    vmin = math.floor(np.min(lsmaxheatmap[np.nonzero(lsmaxheatmap)])*100)/100
    vmax = math.ceil(np.max(lsmaxheatmap[np.nonzero(lsmaxheatmap)])*100)/100
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(np.linspace(1.-(vmax-vmin)/float(vmax), 1, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_jet', colors)
    cax, _  = matplotlib.colorbar.make_axes(plt.gca())
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=color_map, norm=norm,)
    cbar.set_ticks([vmin,(vmax+vmin)/2,vmax])
    cbar.set_ticklabels([vmin,(vmax+vmin)/2,vmax])
    #cax.setlabel('Max. dF/F')
    ax.axis('off')
    ff.suptitle('Heatmap of light-sheet neurons by maximum dF/F')
    plt.show()
    # Light-sheet, spike count figure
    ff,ax = plt.subplots()
    ax.imshow(lscountheatmap, cmap='jet')
    vmin = np.min(lscountheatmap[np.nonzero(lscountheatmap)])
    vmax = np.max(lscountheatmap[np.nonzero(lscountheatmap)])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(np.linspace(1.-(vmax-vmin)/float(vmax), 1, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_jet', colors)
    cax, _  = matplotlib.colorbar.make_axes(plt.gca())
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=color_map, norm=norm,)
    cbar.set_ticks([vmin,math.floor((vmax+vmin)/2),vmax])
    cbar.set_ticklabels([vmin,math.floor((vmax+vmin)/2),vmax])
    #cax.setlabel('Event Count')
    ax.axis('off')
    ff.suptitle('Heatmap of light-sheet neurons by spike count')
    ff.show()
    # Epi-illumination, maximum dF/F figure
    ff,ax = plt.subplots()
    ax.imshow(epmaxheatmap, cmap='jet')
    vmin = math.floor(np.min(epmaxheatmap[np.nonzero(epmaxheatmap)])*100)/100
    vmax = math.ceil(np.max(epmaxheatmap[np.nonzero(epmaxheatmap)])*100)/100
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(np.linspace(1.-(vmax-vmin)/float(vmax), 1, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_jet', colors)
    cax, _  = matplotlib.colorbar.make_axes(plt.gca())
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=color_map, norm=norm,)
    cbar.set_ticks([vmin,(vmax+vmin)/2,vmax])
    cbar.set_ticklabels([vmin,(vmax+vmin)/2,vmax])
    #cbar.ax.setlabel('Max. dF/F')
    ax.axis('off')
    ff.suptitle('Heatmap of epi-illuminated neurons by maximum dF/F')
    ff.show()
    # Epi-illumination, spike count figure
    ff,ax = plt.subplots()
    ax.imshow(epcountheatmap, cmap='jet')
    vmin = np.min(epcountheatmap[np.nonzero(epcountheatmap)])
    vmax = np.max(epcountheatmap[np.nonzero(epcountheatmap)])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(np.linspace(1.-(vmax-vmin)/float(vmax), 1, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_jet', colors)
    cax, _  = matplotlib.colorbar.make_axes(plt.gca())
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=color_map, norm=norm,)
    cbar.set_ticks([vmin,math.floor((vmax+  vmin)/2),vmax])
    cbar.set_ticklabels([vmin,math.floor((vmax+vmin)/2),vmax])
    #cax.setlabel('Event Count')
    ax.axis('off')
    ff.suptitle('Heatmap of epi-illuminated neurons by spike count')
    ff.show()

    ffls, axls = plt.subplots(6,1)
    ffep, axep = plt.subplots(6,1)
    for i in range(6):
        axls[i].plot(topsixlscount[i])
        axep[i].plot(topsixepcount[i])
    ffls.suptitle('Top 6 Lightsheet neurons')
    ffep.suptitle('Top 6 Epi neurons')
    
    # export data (df/f traces and concat times) to be plotted in matlab
    scipy.io.savemat(os.path.join(mcdir, 'lightsheet_dff_data.mat'), {'trace':lslsdff, 'timebreaks':lsvidconcattimes})
    scipy.io.savemat(os.path.join(mcdir, 'epi_dff_data.mat'), {'trace':epepdff, 'timebreaks':epividconcattimes})
    scipy.io.savemat(os.path.join(mcdir, 'lightsheet_dff_datatops.mat'), {'trace':topsixlscount, 'timebreaks':lsvidconcattimes, 'idxs':topls_idxs})
    scipy.io.savemat(os.path.join(mcdir, 'epi_dff_datatops.mat'), {'trace':topsixepcount, 'timebreaks':epividconcattimes, 'idxs':topep_idxs})
    scipy.io.savemat(os.path.join(mcdir, 'epi_masks.mat'), {'mask':epallmasks})
    scipy.io.savemat(os.path.join(mcdir, 'ls_masks.mat'), {'mask':lsallmasks})

if __name__ == "__main__":
    main()