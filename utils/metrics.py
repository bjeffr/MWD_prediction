import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def mean_absolute_errors(Y_true, Y_pred):
    errors = []
    for i in range(Y_pred.shape[1]):
        errors.append(mean_absolute_error(Y_true[:, i], Y_pred[:, i]))

    return errors


def pdi_mean_relative_error(y_true, y_pred):
    abs_err = abs(y_pred - y_true)
    rel_err = abs_err / (y_true - 1)

    return np.mean(rel_err)


def mean_relative_errors(Y_true, Y_pred):
    errors = []
    for i in range(Y_pred.shape[1]):
        if i in (0, 2):
            errors.append(mean_absolute_percentage_error(Y_true[:, i], Y_pred[:, i]))
        elif i in (1, 3):
            errors.append(pdi_mean_relative_error(Y_true[:, i], Y_pred[:, i]))

    return errors


def ds_size_rel_errors(dir, sample_sizes, labels):
    rel_errs = []
    for n_samples in sample_sizes:
        Y_test = np.load(f'data/predictions/ds_size/{dir}/Y_test.npy', allow_pickle=False)
        Y_pred = np.load(f'data/predictions/ds_size/{dir}/Y_pred_{n_samples}.npy', allow_pickle=False)
        row = [n_samples]
        row.extend(mean_relative_errors(Y_test, Y_pred))
        rel_errs.append(row)

    columns = ['n_samples']
    columns.extend(labels)
    df_rel_errs = pd.DataFrame(rel_errs, columns=columns)
    df_rel_errs.set_index('n_samples', inplace=True)

    return df_rel_errs
