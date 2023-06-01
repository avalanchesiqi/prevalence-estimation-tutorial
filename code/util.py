import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


class MixtureModel(stats.rv_continuous):
    def __init__(self, submodels, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.num_model = len(submodels)
        weight_sum = sum(weights)
        self.weights = [weight/weight_sum for weight in weights]
        self.num_theor_slice = 100
        self.theor_cx_axis = np.linspace(0, 1, self.num_theor_slice + 1)
        self.num_empir_bin = 20
        self.empir_cx_axis = np.linspace(0, 1, self.num_empir_bin + 1)

    def _pdf(self, x):
        pdf = self.weights[0] * self.submodels[0].pdf(x)
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += weight * submodel.pdf(x)
        return pdf

    def rvs(self, size):
        submodel_choices = np.random.choice(range(self.num_model), size=size, p=self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs
    
    def plot_pdf_and_hist(self, size, color='k'):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes = axes.ravel()
        
        rv_pdf = self.pdf(self.theor_cx_axis)
        for slice_idx in range(self.num_theor_slice):
            transpancy = (self.theor_cx_axis[slice_idx] + self.theor_cx_axis[slice_idx+1]) / 2
            axes[0].fill_between([self.theor_cx_axis[slice_idx], self.theor_cx_axis[slice_idx+1]], 
                                 [0, 0], 
                                 [rv_pdf[slice_idx], rv_pdf[slice_idx+1]], 
                                 facecolor=color, alpha=transpancy, lw=0)
        
        axes[0].plot(self.theor_cx_axis, rv_pdf, c=color, lw=2, zorder=50)
        axes[0].set_ylabel('$P(UC(X))$', fontsize=16)

        rv_scores = self.rvs(size)
        hist, _ = np.histogram(rv_scores, bins=self.empir_cx_axis)
                
        for bin_idx in range(self.num_empir_bin):            
            transpancy = (self.empir_cx_axis[bin_idx] + self.empir_cx_axis[bin_idx+1]) / 2
            axes[1].fill_between([self.empir_cx_axis[bin_idx], self.empir_cx_axis[bin_idx+1]], 
                                 [0, 0], 
                                 [hist[bin_idx], hist[bin_idx]], 
                                 facecolor=color, alpha=transpancy, lw=0)

        axes[1].plot((self.empir_cx_axis[1:] + self.empir_cx_axis[:-1]) / 2, 
                     hist, c=color, lw=2, zorder=50)
        axes[1].set_ylabel('frequency', fontsize=16)
            
        for ax in axes:
            ax.set_xlabel('$UC(X)$', fontsize=16)
            ax.set_xlim([-0.02, 1.02])
            ax.set_xticks([0, 0.5, 1])
            ax.set_ylim(ymin=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=14)
        
        plt.tight_layout()


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
    
    def reset_oracle(self):
        self.observed_df['GT'] = ''
    
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