import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_SAflow_bids(FOLDERPATH, subj, run, stage, cond=None):
    '''
    Constructs BIDS basename and filepath in the SAflow database format.
    '''
    if run == '1' or run == '8': # determine task based on run number
        task = 'RS'
    else:
        task = 'gradCPT'

    if 'epo' in stage or 'raw' in stage: # determine extension based on stage
        extension = '.fif.gz'
    elif 'PSD' in stage:
        extension = '.mat'
    elif 'sources' in stage:
        extension = '.hd5'
    elif 'events' in stage:
        extension = '.tsv'
    elif 'ARlog' in stage:
        extension = '.hdf5'

    if 'events' in stage:
        SAflow_bidsname = 'sub-{}_ses-recording_task-{}_run-0{}_{}{}'.format(subj, task, run, stage, extension)
    else:
        if cond == None: # build basename with or without cond
            SAflow_bidsname = 'sub-{}_ses-recording_task-{}_run-0{}_meg_{}{}'.format(subj, task, run, stage, extension)
        else:
            SAflow_bidsname = 'sub-{}_ses-recording_task-{}_run-0{}_meg_{}_{}{}'.format(subj, task, run, cond, stage, extension)

    SAflow_bidspath = os.path.join(FOLDERPATH, 'sub-{}'.format(subj), 'ses-recording', 'meg', SAflow_bidsname)
    return SAflow_bidsname, SAflow_bidspath



def array_topoplot(toplot, ch_xy, showtitle=False, titles=None, savefig=False, figpath=None, vmin=-1, vmax=1, cmap='magma', with_mask=False, masks=None, show=True):
    #create fig
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=5)
    fig, ax = plt.subplots(1,len(toplot), figsize=(20,20))
    for i, data in enumerate(toplot):
        if with_mask == False:
            image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax[i], show=False, contours=None, extrapolate='box', outlines='head')
        elif with_mask == True:
            image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax[i], show=False, contours=None, mask_params=mask_params, mask=masks[i], extrapolate='box', outlines='head')
        #option for title
        if showtitle == True:
            ax[i].set_title(titles[i], fontdict={'fontsize': 20, 'fontweight': 'heavy'})
    #add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    divider = make_axes_locatable(ax[-1])
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.tick_params(labelsize=14)
    #save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300)
    if show == True:
        plt.show()
        plt.close(fig=fig)
    else:
        plt.close(fig=fig)
    return fig

def create_pval_mask(pvals, alpha=0.05):
    mask = np.zeros((len(pvals),), dtype='bool')
    for i, pval in enumerate(pvals):
        if pval <= alpha:
            mask[i] = True
    return mask
