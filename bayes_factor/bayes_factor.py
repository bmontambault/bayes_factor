import numpy as np
import pandas as pd
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri
import rpy2.rinterface_lib.callbacks
import rpy2.robjects as robjects

numpy2ri.activate()
pandas2ri.activate()
RBayesFactor=importr('BayesFactor', suppress_messages=True)

class BayesFactor(object):

    def __init__(self, dtypes):
        self.dtypes = dtypes
        self.capture_r_output()

    def capture_r_output(self):
        self.stdout = []
        self.stderr = []
        def add_to_stdout(line): self.stdout.append(line)
        def add_to_stderr(line): self.stderr.append(line)
        self.stdout_orig = rpy2.rinterface_lib.callbacks.consolewrite_print
        self.stderr_orig = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
        rpy2.rinterface_lib.callbacks.consolewrite_print     = add_to_stdout
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = add_to_stderr

    def ttest(self, data, y_field=None, x_field=None, mask=None):
        if mask is None:
            mask = data[x_field].astype(bool)
        if mask.sum() <= 1 or (~mask).sum() <= 1:
            return 0
        res = RBayesFactor.ttestBF(x=data.loc[mask][y_field].values, y=data.loc[~mask][y_field].values)
        bf = res.slots['bayesFactor']['bf'][0]
        return bf

    def anova(self, data, y_field, x_field):
        rdata = pandas2ri.DataFrame(data[[x_field, y_field]])
        x_col = robjects.vectors.FactorVector(rdata.rx2(x_field))
        x_col_index = list(rdata.colnames).index(x_field)
        rdata[x_col_index] = x_col
        formula = Formula(f"{y_field} ~ {x_field}")
        res = RBayesFactor.anovaBF(formula=formula, data=rdata)
        bf = res.slots['bayesFactor']['bf'][x_field]
        return bf

    def regression(self, data, y_field, x_field):
        formula = Formula(f"{y_field} ~ {x_field}")
        res = RBayesFactor.regressionBF(formula=formula, data=data)
        bf = res.slots['bayesFactor']['bf'][x_field]
        return bf

    def bayes_factor(self, data, y_field, x_field=None, mask=None, verbose=False):
        if x_field is None:            
            x_type = None
        else:
            x_type = self.dtypes[x_field]
        y_type = self.dtypes[y_field]
        if verbose:
            print(x_type, y_type)
        if y_type == 'numeric':
            if x_type is None:
                return self.ttest(data, y_field=y_field, mask=mask)
            elif x_type == 'binary':
                return self.ttest(data[mask], x_field, y_field)
            elif x_type == 'nominal':
                return self.anova(data[mask], x_field, y_field)
            elif x_type in ['ordinal', 'numeric']:
                return self.regression(data[mask], x_field, y_field)