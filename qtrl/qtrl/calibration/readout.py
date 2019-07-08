# Copyright (c) 2018-2019, UC Regents

import numpy as np
from ..processing import find_resonator_names
from sklearn.decomposition import PCA
from sklearn import mixture


def find_rotation(measurement, display=True):
    """uses PCA to create a list of rotation angles to get all IQ plots along 1 axis"""
    res_names = find_resonator_names(measurement)

    angles = []
    for res in res_names:
        pca = PCA(2, whiten=True)
        pca.fit(measurement[res]['Heterodyne'].reshape(2, -1).T)

        angles.append(np.arctan2(*pca.components_[1]))
        if display:
            print("- ", angles[-1])
    return angles


def find_GMM(measurement, n_gaussians=2, input_name='Heterodyne', herald_trigger=0):
    """Fit gaussian mixture model to all resonators in current measurement
    Example Use:
        results = find_GMM(dat)
        results['result_name'] = 'GMM'
        cfg.ADC['processing/00_common/05_GMM/kwargs'] = results
        results
    """
    res_names = find_resonator_names(measurement)

    means = []
    covs = []
    for res in res_names:
        samples = measurement[res][input_name].reshape(2, -1)
        gmix = mixture.GaussianMixture(n_components=n_gaussians, covariance_type='spherical')
        gmix.fit(samples.T)

        order = np.argsort(gmix.means_[:, 0])
        gmix.means_ = gmix.means_[order]
        gmix.covariances_ = gmix.covariances_[order]

        # If there are more than 1 triggers in the data, we can do some sorting
        if measurement[res][input_name].shape[-1] > 1:
            herald_results = measurement[res][input_name][..., herald_trigger].reshape(2, -1)
            pred_results = gmix.predict(herald_results.T)

            if np.mean(pred_results == 0) > 0.5:
                # Then the order is correct, do nothing
                pass
            else:

                ordering = np.argsort([np.mean(pred_results == n) for n in range(n_gaussians)])[::-1]

                print(f'Swapping order of {res} to order {ordering}')
                # Re-order the blobs as appropriate
                gmix.means_ = gmix.means_[ordering]
                gmix.covariances_ = gmix.covariances_[ordering]

        means.append(np.around(gmix.means_.reshape(-1), 1).tolist())
        covs.append(np.around(gmix.covariances_, 1).tolist())

    results = dict()
    results['covariances'] = covs
    results['means'] = means

    return results
