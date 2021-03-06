from scipy.io import loadmat, savemat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, ShuffleSplit, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
from pathlib import Path
import argparse
import os
from utils import get_SAflow_bids
from neuro import load_PSD_data
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FREQS_NAMES, FEAT_PATH
import pickle
import time
from joblib import Parallel, delayed
from itertools import product
from scipy.stats.mstats import zscore


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--features",
    default='PSD_VTC',
    type=str,
    help="Channels to compute",
)


args = parser.parse_args()

### ML single subject classification of IN vs OUT epochs
# - single-features
# - CV k-fold (maybe 10 ?)
# - LDA, RF, kNN ?
def prepare_data(PSD_data, FREQ, CHAN=None, norm=True):
    # TODO : ADD OPTION TO ZTRANSFORM
    '''
    Returns X, y and groups arrays from SAflow data for sklearn classification.
    FREQ is an integer
    CHAN is an int or a list of int
    '''
    # retain desired CHAN(s)
    for i, cond in enumerate(PSD_data):
        for j, subj in enumerate(cond):
            if CHAN != None:
                if norm:
                    PSD_data[i][j] = zscore(PSD_data[i][j][FREQ,CHAN,:])
                else:
                    PSD_data[i][j] = PSD_data[i][j][FREQ,CHAN,:]
            else:
                if norm:
                    PSD_data[i][j] = zscore(PSD_data[i][j][FREQ,:,:], axis=1) # CHECK IF IT IS THE RIGHT AXIS
                else:
                    PSD_data[i][j] = PSD_data[i][j][FREQ,:,:]
    X_list = []
    y_list = []
    groups_list = []
    for i, cond in enumerate(PSD_data):
        for j, subj in enumerate(cond):
            X_list.append(subj)
            if i == 0:
                y_list.append(np.zeros(len(subj)))
            elif i == 1:
                y_list.append(np.ones(len(subj)))
            groups_list.append(np.ones(len(subj))*j)
    X = np.concatenate((X_list), axis=0).reshape(-1, 1)
    y = np.concatenate((y_list), axis=0)
    groups = np.concatenate((groups_list), axis=0)
    return X, y, groups


def classif_singlefeat(X,y,groups, FREQ, CHAN):
    cv = LeaveOneGroupOut()
    inner_cv = LeaveOneGroupOut()
    p_grid = {}
    p_grid['gamma']= [1e-3, 1e-4]
    p_grid['C']= [1, 10, 100, 1000]
    n_iter_search=10
    svc = SVC()
    clf = RandomizedSearchCV(svc, param_distributions=p_grid,
                                   n_iter=n_iter_search, cv=inner_cv)
    results = classification(clf, cv, X, y, groups=groups, perm=1001, n_jobs=-1)
    print('Done')
    print('DA : ' + str(results['acc_score']))
    print('p value : ' + str(results['acc_pvalue']))
    return results

def optiSVMsf(CHAN, FREQ, FEAT_FILE, RESULTS_PATH):
    SAVEPATH = '{}/classif_{}_{}.mat'.format(RESULTS_PATH, FREQS_NAMES[FREQ], CHAN)
    if not os.path.exists(SAVEPATH):
        with open(FEAT_FILE, 'rb') as fp:
            PSD_data = pickle.load(fp)
        X, y, groups = prepare_data(PSD_data, FREQ, CHAN)
        print('Computing chan {} in {} band :'.format(CHAN, FREQS_NAMES[FREQ]))
        results = classif_singlefeat(X,y,groups, FREQ, CHAN)
        savemat(SAVEPATH, results)
    else:
        print('Already exists : {}'.format(SAVEPATH))

FEAT_PATH = '../features/'
FEAT_FILE = FEAT_PATH + args.features
RESULTS_PATH = '../results/single_feat/optiSVMsf_L1SO_' + args.features


if __name__ == "__main__":
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
        print('Results folder created at : {}'.format(RESULTS_PATH))
    else:
        print('{} already exists.'.format(RESULTS_PATH))
    Parallel(n_jobs=-1)(
        delayed(optiSVMsf)(CHAN, FREQ, FEAT_FILE, RESULTS_PATH) for CHAN, FREQ in product(range(270), range(len(FREQS_NAMES)))
    )
