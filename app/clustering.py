from typing import TypeVar, Any, Dict, Union

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn import metrics

from prettytable import PrettyTable
import matplotlib.pyplot as plt

CLSTR = TypeVar("CLSTR", bound=ClusterMixin)
TRFM = TypeVar("TRFM", bound=TransformerMixin)
DF = TypeVar("DF", pd.DataFrame, np.ndarray)


class Clusterizer:
    """
    General Clusteriser that implements pipeline for all sklearn 
    clusterisers together with dimensionality reduction.
    
    Args:
        df - pd.DataFrame or np.ndarray as input data
        
        
    Example of usage:
    ```
    CONFIG_KMEANS = {
    "model_type" : KMeans,
    "model_kwargs" : {
                        "n_clusters" : 3,
                        "init" : "k-means++",
                        "n_init" : 2,
                        },
    "reducer_type" : PCA,
    "reducer_kwargs" : {
                        "n_components" : 2,
                        },
    "reduce_first" : False,
    
    "apply_standardisation" : {
                                "reducer" : False,
                                "clusterizer" : False
                                }
    }
    
    kmeans = Clusterizer(df)
    kmeans.run(**CONFIG_KMEANS)
    kmeans.plot_results()
    ```
    """
    def __init__(self, df: DF):
        self.df = df.copy()
        self.data_to_clusterise = None

    def _clusterize(
        self, data: pd.DataFrame, model_type: CLSTR, model_kwargs: Dict[str, Any]
    ) -> CLSTR:
        # Check for standardisation
        do_standardisation = self.apply_standardisation.get("clusterizer", False)

        if do_standardisation:
            data = self._standardise_data(data)

        # Compute clusterization
        model_out = model_type(**model_kwargs).fit(data)

        return model_out

    def _reduce_dim(
        self, data, reducer_type: TRFM, reducer_kwargs: Dict[str, Any]
    ) -> DF:
        # Check for standardisation
        do_standardisation = self.apply_standardisation.get("clusterizer", False)

        if do_standardisation:
            data = self._standardise_data(data)

        # Compute dimensionality reduction
        data_out = reducer_type(**reducer_kwargs).fit_transform(data)
        return data_out

    def _get_name_class(self, _class: Union[CLSTR, TRFM]) -> str:
        l_index = str(_class).rfind(".")
        r_index = str(_class).rfind("'")
        return str(_class)[(l_index + 1) : r_index]

    def _standardise_data(self, data: pd.DataFrame) -> DF:
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def run(
        self,
        model_type: CLSTR,
        model_kwargs: Dict[str, Any],
        reducer_type: TRFM,
        reducer_kwargs: Dict[str, Any],
        apply_standardisation: Dict[str, bool],
        reduce_first: bool,
    ):
        """
        Executes the clustering pipeline
        
        Args:
            model_type - class of clusterisation model from sklearn.cluster 
            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
            
            model_kwargs - a dictionary of arguments used for configuration of the clustering method
            
            reducer_type - class of decomposition from sklearn.decomposition
            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
            
            reducer_kwargs - a dictionary of arguments used for dimensionality reducer
            
            apply_standardisation - a dictionary of two keys: "clusterizer" and "reducer" with boolean
            values for application of standardisation of dataframe before application of their respective
            methods
            
            reduce_first - boolean value for an action of firstly applying the dimensionality reduction and
            then applying the clustering, or without it
        """
        # Get name of used model for plotting
        self.name_clusteriser = self._get_name_class(model_type)

        # Get name of used reducer for plotting
        self.name_reducer = self._get_name_class(reducer_type)

        # Save standartisation config
        self.apply_standardisation = apply_standardisation

        # Reduce dimensionality
        self.reduced_data = self._reduce_dim(self.df, reducer_type, reducer_kwargs)

        # Set reduce_first flag
        self.reduce_first = reduce_first

        # Clusterise data
        if self.reduce_first:
            self.data_to_clusterise = self.reduced_data
        else:
            self.data_to_clusterise = self.df

        self.clusteriser = self._clusterize(
            self.data_to_clusterise, model_type, model_kwargs
        )

        return self.clusteriser

    def plot_results(self):
        if self.data_to_clusterise is None:
            raise Exception('Cannot plot results, execute the method "run" first!')

        plt.scatter(
            self.reduced_data[:, 0], self.reduced_data[:, 1], c=self.clusteriser.labels_
        )
        plt.title(
            f"2D Projection of {self.name_reducer} for {self.name_clusteriser} clustering"
        )
        plt.show()

    def compute_metrics(self):
        if self.data_to_clusterise is None:
            raise Exception('Cannot compute metrics, execute the method "run" first!')
        # Create table
        pretty_table = PrettyTable(['Clustering', 'Reducer', 'reduce_first', 'DBI', 'Silhouette'])
        pretty_table.float_format = '.2'
        
        # Compute dbi score
        dbi = metrics.davies_bouldin_score(self.data_to_clusterise, self.clusteriser.labels_)

        # Compute Silhoutte Score (closer to 1 is better)
        ss = metrics.silhouette_score(self.data_to_clusterise, self.clusteriser.labels_, metric='euclidean')

        pretty_table.add_row([self.name_clusteriser, self.name_reducer, self.reduce_first, dbi, ss])

        return pretty_table
