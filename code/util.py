import pandas as pd
import numpy as np

class DataHandler():
    """
    A discrete distribution over possible labels

    Parameters
    ----------
    state_name
    labels
        A sequence of strings; the allowable labels
    probabilities
        A sequence of the same length, with values adding to one, giving probabilities for each of the label strings
    """

    def __init__(self, filepath: str,):
        self.filepath = filepath
        self.observed_df = pd.read_csv(self.filepath, sep=',', usecols=['UC(X)'])
        self.size = self.observed_df.shape[0]
        self.observed_df['GT'] = ''
        self._oracle_df = pd.read_csv(self.filepath, sep=',')
        self._p_gt = self._oracle_df[self._oracle_df['GT'] == True].shape[0] / self.size

    def get_observer_df(self):
        return self.observed_df
        
    def __repr__(self):
        return 'This is a data handler object.'

    def load_features(self, features):
        self.observed_df[features] = self._oracle_df[features]
    
    def get_oracle_labels(self, rows=[1]):
        self.observed_df.loc[rows, 'GT'] = self._oracle_df.loc[rows, 'GT']
    
    def get_sample_for_labeling(self, n_item=100, strategy='random'):
        unlabeled_subset = self.observed_df[self.observed_df['GT'] == '']
        if strategy == 'random':
            return unlabeled_subset.sample(n=n_item).index
#         elif strategy == 'purposive':
#         size = unlabeled_subset.shape[0]

    def get_labeled_sample(self):
        return self.observed_df[self.observed_df['GT'] != '']
    
    def count_gt(self):
        return (self.observed_df['GT'].values != '').sum() 

    def draw_labels(self, n: int):
        """
        Make n iid draws of discrete labels from the distribution

        Parameters
        ----------
        n
            How many labels to draw from the distribution

        Returns
        -------
            a single item or a numpy array
        """
        
        probabilities = probabilities / probabilities.sum()
        
        return np.random.choice(
            self.labels,
            n,
            p=probabilities
        )