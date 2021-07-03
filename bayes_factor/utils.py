import numpy as np
import pandas as pd

def get_next_column_dtype(data, dtype):
    if data is None:
        return 0
    else:
        dtype_columns = [int(col.split('_')[1]) for col in data.columns if dtype in col]
        if len(dtype_columns) > 0:
            return max(dtype_columns)+1
        else:
            return 0

def set_column_names(data, dtype, columns):
    next_column = get_next_column_dtype(data, dtype)
    columns = [f'{dtype}_{i}' for i in range(next_column, columns+next_column)] if type(columns) == int else columns
    return columns

def sample(new_data, data, dtypes, mask, new_dtype, columns, column_names):
    if dtypes is None:
        dtypes = {col: new_dtype for col in new_data.columns}
    else:
        for col in new_data.columns:
            dtypes[col] = new_dtype
    if data is None:
        data = new_data
    elif mask is not None:
        data.loc[mask, new_data.columns] = new_data.loc[mask]
    else:
        data = pd.concat([data, new_data], axis=1)

    if column_names is None:
        column_names = set_column_names(data, new_dtype, columns)
    data.columns = column_names
    return data, dtypes

def sample_binary(p, rows, columns, data=None, dtypes=None, mask=None, column_names=None):
    new_data = pd.DataFrame(np.random.binomial(1, p, size=(rows, columns if type(columns) == int else len(columns))))
    data, dtypes = sample(new_data, data, dtypes, mask, 'binary', columns, column_names)
    return data, dtypes

def sample_nominal(p, rows, columns, data=None, dtypes=None, mask=None, column_names=None):
    new_data = pd.DataFrame(np.random.choice(np.arange(len(p)), p=p, size=(rows, columns if type(columns) == int else len(columns)))).astype(int).astype(str)
    data, dtypes = sample(new_data, data, dtypes, mask, 'nominal', columns, column_names)
    return data, dtypes

def sample_ordinal(p, rows, columns, data=None, dtypes=None, mask=None, column_names=None):
    new_data = pd.DataFrame(np.random.choice(np.arange(len(p)), p=p, size=(rows, columns if type(columns) == int else len(columns))))
    data, dtypes = sample(new_data, data, dtypes, mask, 'ordinal', columns, column_names)
    return data, dtypes

def sample_numeric(mean, std, rows, columns, data=None, dtypes=None, mask=None, regressor_col=None, slope=0, column_names=None):
    if regressor_col is not None and data is not None:
        y = (data[regressor_col] * slope)[:,None]
    else:
        y = 0
    new_data = pd.DataFrame(np.random.normal(mean, std, size=(rows, columns if type(columns) == int else len(columns))) + y)
    data, dtypes = sample(new_data, data, dtypes, mask, 'numeric', columns, column_names)
    return data, dtypes