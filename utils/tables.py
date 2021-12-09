import pandas as pd


def sci_notation(number, sig_fig=2):
    ret_string = '{0:.{1:d}e}'.format(number, sig_fig)
    a, b = ret_string.split('e')
    b = int(b)

    return '$' + a + r'\times10^{' + str(b) + '}$'


def summary_stats_1c(targets, features):
    unique_count = []
    for col in targets.columns:
        unique_count.append(len(targets[col].unique()))
    for col in features.columns:
        unique_count.append(len(features[col].unique()))
    unique_count = pd.DataFrame([unique_count], index=['unique count'], columns=['M_W', 'PDI', 'G\'', 'G\'\''])

    df = pd.concat([targets.describe(), features.describe()], axis=1)
    indices = list(df.index)
    df = df.append(unique_count)
    indices.insert(1, 'unique count')
    df = df.reindex(indices)
    df.index = ['count', 'unique count', 'mean', 'std', 'min', r'1\textsuperscript{st} quartile', 'median', r'3\textsuperscript{rd} quartile', 'max']
    df = df.astype(str)

    for index, row in df.iterrows():
        if index in ['count', 'unique count']:
            for col in df.columns:
                df.at[index, col] = f'{float(row[col]):,.0f}'
        else:
            df.at[index, 'M_W'] = f"{float(row['M_W']):,.0f}"
            df.at[index, 'PDI'] = f"{float(row['PDI']):.2f}"
            df.at[index, 'G\''] = sci_notation(float(row['G\'']), 2)
            df.at[index, 'G\'\''] = sci_notation(float(row['G\'\'']), 2)

    return df


def summary_stats_targets_2c(targets):
    unique_count = []
    for col in targets.columns:
        unique_count.append(len(targets[col].unique()))
    unique_count = pd.DataFrame([unique_count], index=['unique count'], columns=['M_W_S', 'PDI_S', 'M_W_L', 'PDI_L', 'phi_L'])

    df = targets.describe()
    indices = list(df.index)
    df = df.append(unique_count)
    indices.insert(1, 'unique count')
    df = df.reindex(indices)
    df.index = ['count', 'unique count', 'mean', 'std', 'min', r'1\textsuperscript{st} quartile', 'median', r'3\textsuperscript{rd} quartile', 'max']
    df = df.astype(str)

    for index, row in df.iterrows():
        if index in ['count', 'unique count']:
            for col in df.columns:
                df.at[index, col] = f'{float(row[col]):,.0f}'
        else:
            df.at[index, 'M_W_S'] = f"{float(row['M_W_S']):,.0f}"
            df.at[index, 'PDI_S'] = f"{float(row['PDI_S']):.2f}"
            df.at[index, 'M_W_L'] = f"{float(row['M_W_L']):,.0f}"
            df.at[index, 'PDI_L'] = f"{float(row['PDI_L']):.2f}"
            df.at[index, 'phi_L'] = f"{float(row['phi_L']):.3f}"

    return df


def summary_stats_features_2c(features):
    unique_count = []
    for col in features.columns:
        unique_count.append(len(features[col].unique()))
    unique_count = pd.DataFrame([unique_count], index=['unique count'], columns=['G\'', 'G\'\''])

    df = features.describe()
    indices = list(df.index)
    df = df.append(unique_count)
    indices.insert(1, 'unique count')
    df = df.reindex(indices)
    df.index = ['count', 'unique count', 'mean', 'std', 'min', r'1\textsuperscript{st} quartile', 'median', r'3\textsuperscript{rd} quartile', 'max']
    df = df.astype(str)

    for index, row in df.iterrows():
        if index in ['count', 'unique count']:
            for col in df.columns:
                df.at[index, col] = f'{float(row[col]):,.0f}'
        else:
            df.at[index, 'G\''] = sci_notation(float(row['G\'']), 2)
            df.at[index, 'G\'\''] = sci_notation(float(row['G\'\'']), 2)

    return df
