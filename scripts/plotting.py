import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from hrf_matrix import HRFMatrix
from matplotlib import rc
from nilearn.masking import apply_mask
from sklearn.linear_model import lars_path as lars
from stability_selection import calculate_auc
from stability_selection import stability_selection


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})


def plot_task():

    time = np.linspace(0, 1024*0.5, 1024)
    task = np.zeros(len(time))
    task[np.arange(start=45, stop=1024*2, step=45*2)] = 1


    plt.figure(figsize=(25, 8))
    plt.stem(time, task, linefmt='blue', markerfmt='go',
             basefmt='blue')
    plt.xlabel(r'$Time (s)$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig('task.png', dpi=300)
    plt.close()


def plot_timeseries(data, innovation, r2):

    innovation[innovation == 0] = np.nan
    r2[r2 == 0] = np.nan
    time = np.linspace(0, 300*2, 300)

    plt.figure(figsize=(25, 8))
    plt.plot(time, data, color='black', linewidth=0.5)
    plt.stem(time, 0.1*r2, linefmt='green', markerfmt='go',
             basefmt='green', label='Activity inducing signal')
    plt.legend()
    plt.xlabel(r'$Time (s)$')
    plt.ylabel(r'$Amplitude (SPC)$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig('demo_r2.png', dpi=300)
    plt.close()

    plt.figure(figsize=(25, 8))
    plt.plot(time, data)
    plt.stem(time, 0.1*r2, linefmt='green', markerfmt='go',
             basefmt='green', label='Activity inducing signal')
    plt.stem(time, 0.1*innovation, linefmt='red',
             markerfmt='ro', basefmt='red', label='Innovation signal')
    plt.legend()
    plt.xlabel(r'$Time (s)$')
    plt.ylabel(r'$Amplitude (SPC)$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig('demo_timeseries.png', dpi=300)
    plt.close()


def plot_regularization_path(coef_path, true_idx, cmap=None, key=''):

    if cmap is None:
        cmap = ['black'] * coef_path.shape[0]

        # Red color for true events
        for index, item in enumerate(cmap):
            if true_idx[index] == 1:
                cmap[index] = 'red'

    plt.figure(figsize=(25, 12))

    for i in range(coef_path.shape[0]):
        # True events have a thicker line
        if true_idx[i] == 1 or true_idx[i] == -1:
            l_width = 2
        else:
            l_width = 0.5
        plt.plot(coef_path[i, :].T, color=cmap[i], linewidth=l_width)

    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$Amplitude$')
    plt.title(r'$Regularization \; path$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    locs, labels = plt.xticks()
    new_labels = [None] * len(labels)
    new_labels[0] = r'$\lambda_{max}$'
    new_labels[-1] = r'$\lambda_{min}$'
    ax = plt.gca()
    ax.set_xticklabels(new_labels)
    plt.savefig(f'demo_regul_path{key}.png', dpi=300)
    plt.close()


def plot_stability_path(coef_path, true_idx, cmap=None, key=''):

    if cmap is None:
        cmap = ['black'] * coef_path.shape[0]

        # Red color for true events
        for index, item in enumerate(cmap):
            if true_idx[index] == 1 or true_idx[index] == -1:
                cmap[index] = 'red'

    plt.figure(figsize=(25, 12))

    for i in range(coef_path.shape[0]):
        # True events have a thicker line
        if true_idx[i] == 1 or true_idx[i] == -1:
            l_width = 2
        else:
            l_width = 0.5
        plt.plot(coef_path[i, :].T, color=cmap[i], linewidth=l_width)

    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$Probability$')
    plt.title(r'$Stability \; path$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    locs, labels = plt.xticks()
    new_labels = [None] * len(labels)
    new_labels[0] = r'$\lambda_{max}$'
    new_labels[-1] = r'$\lambda_{min}$'
    ax = plt.gca()
    ax.set_xticklabels(new_labels)
    plt.savefig(f'demo_stabil_path{key}.png', dpi=300)
    plt.close()


def plot_stability_path_filled(coef_path):

    plt.figure(figsize=(25, 12))
    plt.fill_between(np.arange(0, coef_path.shape[0]), 0, coef_path,
                     color='red')
    plt.plot(coef_path, color='red', linewidth=1.5)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$Probability$')
    plt.title(r'$Stability \; path$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    locs, labels = plt.xticks()
    new_labels = [None] * len(labels)
    new_labels[0] = r'$\lambda_{max}$'
    new_labels[-1] = r'$\lambda_{min}$'
    ax = plt.gca()
    ax.set_xticklabels(new_labels)
    plt.savefig('demo_stabil_path_fill.png', dpi=300)
    plt.close()


def plot_auc(auc, true_idx, cmap, thr=False, key=''):
    true_idx[true_idx == 0] = np.nan
    time = np.linspace(0, 300*2, 300)
    idx = np.where((true_idx == 1) | (true_idx == -1))[0]*2

    plt.figure(figsize=(25, 8))
    plt.plot(time, auc, color='black', linewidth=0.5)
    plt.axhline(color='black', linewidth=0.5)
    for x, c in zip(idx, cmap):
        plt.vlines(x, 0, 0.5, color=c)
    plt.xlabel(r'$Time (s)$')
    plt.ylabel(r'$Probability$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    if thr:
        plt.title(r'$Thresholded \; AUC \; time \; series \; for \; a \; given \; voxel$')
        plt.savefig(f'auc_thr{key}.png', dpi=300)
    else:
        plt.title(r'$AUC \; time \; series \; for \; a \; given \; voxel$')
        plt.savefig(f'auc{key}.png', dpi=300)
    plt.close()


def plot_timeseries_colors(data, innovation, r2, cmap1, cmap2):
    innovation[innovation == 0] = np.nan
    r2[r2 == 0] = np.nan
    time = np.linspace(0, 300*2, 300)
    idx = np.where(r2 == 1)[0]*2

    plt.figure(figsize=(25, 8))
    plt.plot(time, data, color='black', linewidth=0.5)
    plt.axhline(color='black', linewidth=0.5)
    for x, c in zip(idx, cmap1):
        plt.vlines(x, 0, 0.2, color=c)
    plt.xlabel(r'$Time (s)$')
    plt.ylabel(r'$Amplitude (SPC)$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig('demo_r2_colors.png', dpi=300)
    plt.close()

    idx = np.where((innovation == 1) | (innovation == -1))[0]*2
    idx2 = np.where((innovation == 1) | (innovation == -1))[0]

    plt.figure(figsize=(25, 8))
    plt.plot(time, data, color='black', linewidth=0.5)
    plt.axhline(color='black', linewidth=0.5)
    for x, c, i in zip(idx, cmap2, idx2):
        if innovation[i] > 0:
            plt.vlines(x, 0, 0.2, color=c)
        elif innovation[i] < 0:
            plt.vlines(x, -0.2, 0, color=c)
    plt.xlabel(r'$Time (s)$')
    plt.ylabel(r'$Amplitude (SPC)$')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig('demo_innovation_colors.png', dpi=300)
    plt.close()


def colors(innovation=False):
    if innovation:
        colors_list = ['#063379', '#5593f6', '#06792f', '#55f68e', '#870cda',
                      '#dfb6fb', '#aa0924', '#f98699', '#da5b0c', '#f69355']
    else:
        colors_list = ['#063379', '#0c5bda', '#5593f6', '#06792f', '#0cda54', '#55f68e', '#4b0679',
                      '#870cda', '#b555f6', '#dfb6fb', '#49040f', '#aa0924', '#f32548', '#f98699',
                       '#da5b0c', '#f69355']
    return colors_list


def main(vox_idx=0):

    # Load data and normalize
    data = np.load('data/simulation_data.npy')

    data += np.random.normal(0, 0.1, (300,8))
    data = (data - np.mean(data, axis=0)) / np.mean(data, axis=0)
    data = data / np.max(data)
    nscans = data.shape[0]
    nvoxels = data.shape[1]

    r2 = np.load('data/simulation_r2.npy')
    innovation = np.load('data/simulation_innovation.npy')

    data_voxel = data[:, vox_idx]
    r2_voxel = r2[:, vox_idx]
    innovation_voxel = innovation[:, vox_idx]

    # Colormap
    cmap = ['black'] * nscans
    color_list = colors()
    counter = 0
    for index, item in enumerate(cmap):
        if r2_voxel[index] == 1:
            cmap[index] = color_list[counter]
            counter += 1

    # Plot time series of given voxel
    # plot_timeseries(data_voxel, innovation_voxel, r2_voxel)
    plot_timeseries_colors(data_voxel, innovation_voxel, r2_voxel, colors(), colors(True))

    # Plot regularization paths
    hrf_matrix = HRFMatrix(TR=2, TE=[0], nscans=nscans, r2only=True,
                           has_integrator=False, is_afni=True)
    hrf_matrix.generate_hrf()
    hrf = hrf_matrix.X_hrf_norm

    nlambdas = nscans + 1

    _, _, coef_path = lars(hrf, np.squeeze(data_voxel),
                           method='lasso', Gram=np.dot(hrf.T, hrf),
                           Xy=np.dot(hrf.T, np.squeeze(data_voxel)),
                           max_iter=nlambdas-1, eps=1e-9)

    plot_regularization_path(coef_path, r2_voxel, cmap, key='_spk')

    # Plot stability paths
    coef_path_stability, lambdas = stability_selection(hrf, data_voxel, nsurrogates=100)

    plot_stability_path(coef_path_stability, r2_voxel, cmap, key='_spk')

    auc = calculate_auc(coef_path_stability, lambdas)
    plot_auc(auc, r2_voxel, colors(), key='_spk')

    auc_thr = auc.copy()
    auc_thr[auc_thr < 0.5] = 0
    plot_auc(auc_thr, r2_voxel, colors(), thr=True, key='_spk')

    # Colormap
    cmap = ['black'] * nscans
    color_list = colors(True)
    counter = 0
    for index, item in enumerate(cmap):
        if innovation_voxel[index] == 1 or innovation_voxel[index] == -1:
            cmap[index] = color_list[counter]
            counter += 1

    hrf_matrix = HRFMatrix(TR=2, TE=[0], nscans=nscans, r2only=True,
                           has_integrator=True, is_afni=True)
    hrf_matrix.generate_hrf()
    hrf = hrf_matrix.X_hrf_norm

    nlambdas = nscans + 1

    _, _, coef_path = lars(hrf, np.squeeze(data_voxel),
                                 method='lasso', Gram=np.dot(hrf.T, hrf),
                                 Xy=np.dot(hrf.T, np.squeeze(data_voxel)),
                                 max_iter=nlambdas-1, eps=1e-9)

    plot_regularization_path(coef_path, innovation_voxel, cmap, key='_int')

    # Plot stability paths
    coef_path_stability, lambdas = stability_selection(hrf, data_voxel, nsurrogates=100)

    plot_stability_path(coef_path_stability, innovation_voxel, cmap, key='_int')

    auc = calculate_auc(coef_path_stability, lambdas)
    plot_auc(auc, innovation_voxel, colors(True), key='_int')

    auc_thr = auc.copy()
    auc_thr[auc_thr < 0.3] = 0
    plot_auc(auc_thr, innovation_voxel, colors(True), thr=True, key='_int')


if __name__ == "__main__":
    main()
