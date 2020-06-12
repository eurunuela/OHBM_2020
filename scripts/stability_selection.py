import numpy as np
from sklearn.linear_model import lars_path


def subsampling(nscans, mode=1, nTE=0):
    # Subsampling for Stability Selection
    if mode == 1:  # different time points are selected across echoes
        subsample_idx = np.sort(np.random.choice(range(nscans), int(0.6*nscans), 0)) # 60% of timepoints are kept
        if nTE > 1:
            for i in range(nTE-1):
                subsample_idx = np.concatenate((subsample_idx, np.sort(np.random.choice(range((i+1)*nscans,(i+2)*nscans), \
                                                int(0.6*nscans), 0))))
    elif mode > 1:  # same time points are selected across echoes
        subsample_idx = np.sort(np.random.choice(range(nscans), int(0.6*nscans), 0)) # 60% of timepoints are kept

    return(subsample_idx)


def stability_selection(hrf, data, nsurrogates):

    nscans = data.shape[0]
    nlambdas = nscans + 1

    lambdas = np.zeros((nsurrogates, nlambdas), dtype=np.float32)
    coef_path = np.zeros((nsurrogates, nscans, nlambdas), dtype=np.float32)
    sur_idxs = np.zeros((nsurrogates, int(0.6*nscans)))

    for surrogate_idx in range(nsurrogates):

        idxs = subsampling(nscans)
        sur_idxs[surrogate_idx, :] = idxs
        y_sub = data[idxs]
        X_sub = hrf[idxs, :]
        max_lambda = abs(np.dot(X_sub.T, y_sub)).max()

        # LARS path
        lambdas_temp, _, coef_path_temp = lars_path(X_sub, np.squeeze(y_sub), method='lasso', Gram=np.dot(X_sub.T, X_sub),
                                                    Xy=np.dot(X_sub.T, np.squeeze(y_sub)), max_iter=nlambdas-1, eps=1e-9)
        lambdas[surrogate_idx, :len(lambdas_temp)] = lambdas_temp
        n_coefs = (coef_path_temp != 0).shape[1]
        coef_path[surrogate_idx, :, :n_coefs] = coef_path_temp != 0

    # Sorting and getting indexes
    lambdas_merged = lambdas.copy()
    lambdas_merged = lambdas_merged.reshape((nlambdas * nsurrogates,))
    sort_idxs = np.argsort(-lambdas_merged)
    lambdas_merged = -np.sort(-lambdas_merged)
    nlambdas_merged = len(lambdas_merged)
    stability_path = np.zeros((nscans, nsurrogates * nlambdas), dtype=np.float64)

    for surrogate_idx in range(nsurrogates):
        if surrogate_idx == 0:
            first = 0
            last = nlambdas - 1
        else:
            first = last + 1
            last = first + nlambdas - 1

        same_lambda_idxs = np.where((first <= sort_idxs) & (sort_idxs <= last))[0]

        # Find indexes of changes in value (0 to 1 changes are expected).
        nonzero_change_scans, nonzero_change_idxs = np.where(np.squeeze(coef_path[surrogate_idx, :, :-1]) != np.squeeze(coef_path[surrogate_idx, :, 1:]))
        nonzero_change_idxs = nonzero_change_idxs + 1

        coef_path_squeezed = np.squeeze(coef_path[surrogate_idx, :, :])
        coef_path_merged = np.full((nscans, nlambdas * nsurrogates), False, dtype=bool)
        coef_path_merged[:, same_lambda_idxs] = coef_path_squeezed.copy()

        for i in range(len(nonzero_change_idxs)):
            coef_path_merged[nonzero_change_scans[i], same_lambda_idxs[nonzero_change_idxs[i]]:] = True

        stability_path += coef_path_merged

    stability_path /= nsurrogates

    return(stability_path, lambdas_merged)


def calculate_auc(coef_path, lambdas):

    nscans = coef_path.shape[0]
    auc = np.zeros((nscans, ), dtype=np.float64)
    nlambdas = len(lambdas)
    lambda_sum = np.sum(lambdas)

    for lambda_idx in range(nlambdas):
        auc += coef_path[:,lambda_idx]*lambdas[lambda_idx]/lambda_sum

    return auc