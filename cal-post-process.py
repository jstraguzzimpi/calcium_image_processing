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

def filter_movingaverage(tracein, n):
    # Performs moving average filtering on an input trace, with filter kernel
    # of width N.
    # Args:
    #   tracein : Input trace (1D array)
    #   n       : Filter width
    ret     = np.cumsum(tracein, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def trace_df(tracein, Nwdw, Nbase):
    # Attempts to extract the dF of a trace. dF should be the trace value minus
    # the "baseline" trace value, where baseline is a kind of trendline
    # corresponding to the neuron's intensity when it is not active.
    # This is done by moving a window of width Nwdw across the image, and
    # taking the bottom Nbase points within that window to be the "baseline"
    # Args:
    #   tracein : Input trace for which to calculate this.
    #   Nwdw    : Window size for baseline calculation
    #   Nbase   : Number of points to consider to be baseline
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
    df, base    = trace_df(tracein, 70, 10)
    f           = np.add(base, Bavg)
    return np.divide(df, f)

def argsort_traces(tracearray):
    # Sorts the array of traces by some maximum dF/F. Returns the indices that
    # sorts the array
    # Args:
    #   tracearray  : NxT array, where N is the number of traces and T is time
    maxdff      = np.amax(tracearray, axis = -1)
    indices     = np.argsort(maxdff)
    return indices

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
        traces[i], bval[i]  = extract_neuron_trace_uniform(vid, mask, flatmask, 2)
        dff[i]              = trace_df_f(traces[i], bval[i])
    return dff

def mask_outline(mask):
    # Generates a new mask that is the outline shape of a given mask.
    # Args:
    #   mask    : Input mask for which to generate the outline
    mask_dx     = np.diff(mask, n=1, axis=0, prepend=0)
    mask_dy     = np.diff(mask, n=1, axis=1, prepend=0)
    return np.logical_or(mask_dx, mask_dy)

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
    lsepi       = [1, 0, 1, 0, 1, 0, 0, 1, 0]

	# Grab the first two masks and assume that they are representative of every
	# lightsheet and epi neuron. Also generate an overlap mask by logical ANDing
    # the two masks together.
    maskLSin    = np.transpose(np.load(fvids[0]+'.npy'), axes=(2,1,0))
    maskEpiin   = np.transpose(np.load(fvids[1]+'.npy'), axes=(2,1,0))
    maskLSt     = []
    maskEpit    = []
    for mask in maskLSin:
        if (np.argwhere(mask).size > 0):
            maskLSt.append(mask)
    for mask in maskEpiin:
        if (np.argwhere(mask).size > 0):
            maskEpit.append(mask)
    maskLS      = np.asarray(maskLSt)
    maskEpi     = np.asarray(maskEpit)

    # Take the first lightsheet vid, find the top N neurons, do the same for
    # the first epi vid. Take the 2N masks corresponding to this, find the set
    # of unique ones, and then use the remaining masks to go through all of the
    # other vids.
    N       = 14
    # Lightsheet:
    vid     = cm.load(fvids[0])
    LSdff   = calculate_dff_set(vid, maskLS)
    # Epi:
    vid     = cm.load(fvids[1])
    EPdff   = calculate_dff_set(vid, maskEpi)

    # Sort by top, get the overlap:
    threshold   = 10
    topsLS      = argsort_traces(LSdff)
    topsEpi     = argsort_traces(EPdff)
    masks       = mask_union(maskLS[topsLS[-N:]], maskEpi[topsEpi[-N:]], threshold)
    masksTopLS  = mask_disjoint(maskLS[topsLS[-N:]], masks, threshold)
    masksTopEpi = mask_disjoint(maskEpi[topsEpi[-N:]], masks, threshold)
    maskov      = mask_joint(maskLS[topsLS[-N:]], maskEpi[topsEpi[-N:]], threshold)
    print(masksTopLS.shape)
    print(masksTopEpi.shape)
    print(maskov.shape)

    # The variable tops now contains the non-overlapping union of the top N
    # neurons from epi and from light sheet. Now run through the rest of the
    # analyses using only these masks.
    # Can grab the top epi and top lightsheet values for each neuron. Can also
    # grab on a neuron-by-neuron basis whether the peak dF/F was lightsheet or
    # epi.
    dff         = np.empty((masks.shape[0], 0))
    divs        = np.zeros(len(fvids))
    max_idx     = np.zeros((masks.shape[0], 1))
    max_val     = np.zeros((masks.shape[0], 1))
    maxepi_idx  = np.zeros((masks.shape[0], 1))
    maxepi_val  = np.zeros((masks.shape[0], 1))
    maxls_idx   = np.zeros((masks.shape[0], 1))
    maxls_val   = np.zeros((masks.shape[0], 1))
    flatmask    = flatten_masks(masks)
    lspeaks     = [[] for k in range(masks.shape[0])]
    lspeakvals  = np.empty((0))
    epipeaks    = [[] for k in range(masks.shape[0])]
    epipeakvals = np.empty((0))
    rawtraces   = np.empty((masks.shape[0], 0))
    rawbckgnd   = np.empty((masks.shape[0], 0))
    for i, fvid in enumerate(fvids):
        vid     = cm.load(fvid)
        traces      = np.empty((masks.shape[0], vid.shape[0]))
        bval        = np.empty((masks.shape[0], vid.shape[0]))
        dff_i       = np.empty((masks.shape[0], vid.shape[0]))
        for j, mask in enumerate(masks):
            traces[j], bval[j]  = extract_neuron_trace_uniform(vid, mask, flatmask, 2)
            dff_i[j]            = trace_df_f(traces[j], bval[j])
            peaks, props        = scipy.signal.find_peaks(dff_i[j], distance=10, prominence=(0.1, None))
            if (peaks.size > 0):
                if lsepi[i]:
                    lspeaks[j].append(peaks)
                    lspeakvals      = np.append(lspeakvals, dff_i[j][peaks])
                else:
                    epipeaks[j].append(peaks)
                    epipeakvals     = np.append(epipeakvals, dff_i[j][peaks])
            if (max(dff_i[j]) > max_val[j]):
                max_idx[j]  = i
                max_val[j]  = max(dff_i[j])
                if lsepi[i]:
                    maxls_idx[j]    = i
                    maxls_val[j]    = max_val[j]
                else:
                    maxepi_idx[j]   = i
                    maxepi_val[j]   = max_val[j]

        dff     = np.concatenate([dff, dff_i], axis = 1)
        rawtraces = np.concatenate([rawtraces, traces], axis = 1)
        rawbckgnd = np.concatenate([rawbckgnd, bval], axis = 1)
        divs[i] = dff.shape[1]

    # Save generated values for post-post-processing
    masksout    = np.transpose(masks, axes=(2,1,0))
    np.save(os.path.join(mcdir, 'top_masks_out.npy'), masksout)
    np.save(os.path.join(mcdir, 'top_epi_sections.npy'), maxepi_idx)
    np.save(os.path.join(mcdir, 'top_epi_values.npy'), maxepi_val)
    np.save(os.path.join(mcdir, 'top_ls_sections.npy'), maxls_idx)
    np.save(os.path.join(mcdir, 'top_ls_values.npy'), maxls_val)
    np.save(os.path.join(mcdir, 'ls_peak_vals.npy'), lspeakvals)
    np.save(os.path.join(mcdir, 'epi_peak_vals.npy'), epipeakvals)
    np.save(os.path.join(mcdir, 'dff_traces.npy'), dff)
    np.save(os.path.join(mcdir, 'dff_div_points.npy'), divs)

    # Plot out the dF/F traces, and put vertical markers at the dividers
    # between video segments. User would have to manually label them as there's
    # no real way to determine what segments are what.
    nrow = 6
    ncol = 1
    dffig, ax    = plt.subplots(nrow, ncol, sharex = True, sharey = True)
    ll   = dff[0].shape[0]
    axrange = np.linspace(0, (ll-1)/10, num=ll)
    for i, dfft in enumerate(dff):
        if i == nrow:
            ax[i-1].set_xlabel('Time (seconds)')
            break
        ax[i].plot(axrange, dfft)
        ax[i].set_ylabel(str(i+1))
        for div in divs:
            ax[i].axvline(div/10, color='r', linestyle='--')
    dffig.suptitle('Neuron dF/F Curves')
    plt.show()

    tfig, ax    = plt.subplots(nrow, ncol, sharex = True, sharey = True)
    ll   = rawtraces[0].shape[0]
    axrange = np.linspace(0, (ll-1)/10, num=ll)
    for i, tr in enumerate(rawtraces):
        if i >= nrow:
            ax[i-1].set_xlabel('Time (seconds)')
            break
        ax[i].plot(axrange, np.add(tr, rawbckgnd[i]))
        ax[i].plot(axrange, rawbckgnd[i])
        ax[i].set_ylabel(str(i+1))
        for div in divs:
            ax[i].axvline(div/10, color='r', linestyle='--')
    tfig.suptitle('Neuron Raw Traces + Background')
    plt.show()

    """
    # Next do line plot with averages + error bars.
    #   Set up lines for a given neuron, showing increase or decrease of max
    #   intensity on that neuron between lightsheet and epi.
    #   
    intensity_lineplot  = np.concatenate([maxepi_val, maxls_val], axis=1).T
    avg_ls  = np.mean(maxls_val)
    std_ls  = np.std(maxls_val)/math.sqrt(maxls_val.shape[0])
    avg_epi = np.mean(maxepi_val)
    std_epi = np.std(maxepi_val)/math.sqrt(maxepi_val.shape[0])
    binplot, ax = plt.subplots()
    plt.plot(intensity_lineplot)
    ax.bar([0, 1], [avg_epi, avg_ls], yerr=[std_epi, std_ls], align='center', capsize=10, alpha=0.5)
    ax.set_ylabel('Peak dF/F')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Epi', 'Light-sheet'])
    ax.set_title('Contrast change, Epi vs. LS')
    plt.show()
    """
    # Histogram of spike intensities.
    histfig, ax = plt.subplots()
    if not (lspeakvals.size>0):
        lspeakvals = np.zeros(1)
    if not (epipeakvals.size>0):
        epipeakvals = np.zeros(1)
    binrange    = np.amax(np.concatenate([lspeakvals, epipeakvals]))
    binrange = 1.5 if binrange >1.5 else math.ceil(binrange*10)/10
    binset  = np.linspace(0, binrange, num=int(binrange*10+1))
    nls     = lspeakvals.shape[0]
    nepi    = epipeakvals.shape[0]
    epi_n, epi_bins, epi_patches    = ax.hist(epipeakvals, bins=binset, alpha=0.5, label='Epi-illumination', histtype='barstacked', ec='black', lw=0, color='#7f86c1')
    ls_n, ls_bins, ls_patches       = ax.hist(lspeakvals, bins=binset, alpha=0.5, label='Light-sheet', histtype='barstacked', ec='black', lw=0, color='#f48466')
    plt.legend(loc='upper right')
    histfig.suptitle('Lightsheet vs. Epi-illumination dF/F')
    plt.xlabel('dF/F')
    plt.ylabel('Spike Count')
    plt.show()

    # Plot the image with the contours (outlines of neurons), labeled
    Asparse     = scipy.sparse.csc_matrix(masksout.reshape((masksout.shape[1]*masksout.shape[0], masksout.shape[2])))
    lstop       = np.transpose(masksTopLS, axes=(2,1,0))
    epitop      = np.transpose(masksTopEpi, axes=(2,1,0))
    ovtop       = np.transpose(maskov, axes=(2,1,0))
    AsparseLS   = scipy.sparse.csc_matrix(lstop.reshape((lstop.shape[1]*lstop.shape[0], lstop.shape[2])))
    AsparseEpi  = scipy.sparse.csc_matrix(epitop.reshape((epitop.shape[1]*epitop.shape[0], epitop.shape[2])))
    AsparseOv   = scipy.sparse.csc_matrix(ovtop.reshape((ovtop.shape[1]*ovtop.shape[0], ovtop.shape[2])))
    vid         = cm.load(fvids[0])
    #Cn          = cm.local_correlations(vid.transpose(1,2,0))
    Cn          = np.zeros((vid.shape[1], vid.shape[2]))
    Cn[np.isnan(Cn)] = 0
    out=plt.figure()
    cm.utils.visualization.plot_contours(Asparse, Cn)
    out=plt.figure()
    cm.utils.visualization.plot_contours(AsparseLS, Cn)
    out=plt.figure()
    cm.utils.visualization.plot_contours(AsparseEpi, Cn)
    out=plt.figure()
    cm.utils.visualization.plot_contours(AsparseOv, Cn)

    scipy.io.savemat(os.path.join(mcdir, 'epi_histogram.mat'), {'n':epi_n, 'bins':epi_bins, 'patches':epi_patches})
    scipy.io.savemat(os.path.join(mcdir, 'ls_histogram.mat'), {'n':ls_n, 'bins':ls_bins, 'patches':ls_patches})
    scipy.io.savemat(os.path.join(mcdir, 'epi_spike_values.mat'), {'epispikes':epipeakvals})
    scipy.io.savemat(os.path.join(mcdir, 'ls_spike_values.mat'), {'lsspikes':lspeakvals})
    scipy.io.savemat(os.path.join(mcdir, 'df_over_f.mat'), {'data':dff, 'indices_between_ls_or_epi':divs})
    scipy.io.savemat(os.path.join(mcdir, 'rawtraces.mat'), {'data':rawtraces, 'indices_between_ls_or_epi':divs})
    scipy.io.savemat(os.path.join(mcdir, 'rawbackground.mat'), {'data':rawbckgnd, 'indices_between_ls_or_epi':divs})

if __name__ == "__main__":
    main()