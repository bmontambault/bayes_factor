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
    """This is the class used to calculate all bayes factors. In addition to methods for running each available test,
    this class also included a method for automatically infering the correct test to run based on data types.

    :param dtypes: Data types for each column in the dataset
    :type dtypes: dict
    """

    def __init__(self, dtypes):
        """Constructor method
        """

        self.dtypes = dtypes
        self.capture_r_output()

    def capture_r_output(self):
        """Method for capturing output from R.
        """

        self.stdout = []
        self.stderr = []
        def add_to_stdout(line): self.stdout.append(line)
        def add_to_stderr(line): self.stderr.append(line)
        self.stdout_orig = rpy2.rinterface_lib.callbacks.consolewrite_print
        self.stderr_orig = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
        rpy2.rinterface_lib.callbacks.consolewrite_print     = add_to_stdout
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = add_to_stderr

    def ttest(self, data, y_field, x_field=None, mask=None):
        """Returns the Bayes factor for a t-test run on a given data frame for defined x and y columns. The column
        designated as the x should be able to be cast as a boolean indicating the two groups in a independent-sample
        t-test.

        :param data: The dataset
        :type data: pd.DataFrame
        :param y_field: Column in the dataset whose means will be compared between groups
        :type y_field: str
        :param x_field: Column in the dataset that will be used to distinguish between groups
        :type x_field: str
        :param mask: Boolean array of length data.shape[0] that will be used to distinguish between groups if x_field is set to None
        :type mask: np.array, pd.Series
        :return: Bayes factor
        :rtype: float
        """

        if mask is None:
            mask = data[x_field].astype(bool)
        if mask.sum() <= 1 or (~mask).sum() <= 1:
            return 0
        res = RBayesFactor.ttestBF(x=data.loc[mask][y_field].values, y=data.loc[~mask][y_field].values)
        bf = res.slots['bayesFactor']['bf'][0]
        return bf

    def anova(self, data, y_field, x_field):
        """Returns the Bayes factor for an ANOVA test run on a given data frame for defined x and y columns. Each unique value 
        in the column designated as x will indicate a group in a one-way ANOVA.

        :param data: The dataset
        :type data: pd.DataFrame
        :param y_field: Column in the dataset whose means will be compared between groups
        :type y_field: str
        :param x_field: Column in the dataset that will be used to distinguish between groups
        :type x_field: str
        :return: Bayes factor
        :rtype: float
        """

        rdata = pandas2ri.DataFrame(data[[x_field, y_field]])
        x_col = robjects.vectors.FactorVector(rdata.rx2(x_field))
        x_col_index = list(rdata.colnames).index(x_field)
        rdata[x_col_index] = x_col
        formula = Formula(f"{y_field} ~ {x_field}")
        res = RBayesFactor.anovaBF(formula=formula, data=rdata)
        bf = res.slots['bayesFactor']['bf'][x_field]
        return bf

    def regression(self, data, y_field, x_field):
        """Returns the Bayes factor for a regression test run on a given data frame for defined x and y columns. The x column and
        y column indicate the indepedent and dependent variable respectively.

        :param data: The dataset
        :type data: pd.DataFrame
        :param y_field: Column in the dataset that will be used as the dependent variable
        :type y_field: str
        :param x_field: Column in the dataset that will be used as the independent variable
        :type x_field: str
        :return: Bayes factor
        :rtype: float
        """

        formula = Formula(f"{y_field} ~ {x_field}")
        res = RBayesFactor.regressionBF(formula=formula, data=data)
        bf = res.slots['bayesFactor']['bf'][x_field]
        return bf

    def bayes_factor(self, data, y_field, x_field=None, mask=None, verbose=False):
        """Returns the Bayes factor for a test determined by the data types of the given x and y columns.

        :param data: The dataset
        :type data: pd.DataFrame
        :param y_field: Column in the dataset that will be used to distinguish between groups or as the dependent variable
        :type y_field: str
        :param x_field: Column in the dataset that will be used to compare means or as the independent variable
        :type x_field: str
        :param mask: Boolean array of length data.shape[0] that will be used to distinguish between groups if x_field is set to None
        :type mask: np.array, pd.Series
        :param verbose: Option to print messages
        :type verbose: bool
        :return: Bayes factor
        :rtype: float
        """

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