from typing import TypeVar, Any, Dict, Union

from fastapi import FastAPI

from sklearn.utils import shuffle

import pandas as pd

from app.clustering import Clusterizer
from app.embeddings import DocEmbeddings

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

app = FastAPI()

@app.post("/")


# CONFIG_KMEANS = {
#     "model_type" : "KMeans",
#     "model_kwargs" : {
#                         "n_clusters" : 3,
#                         "init" : "k-means++",
#                         "n_init" : 2,
#                         },
#     "reducer_type" : "PCA",
#     "reducer_kwargs" : {
#                         "n_components" : 2,
#                         "perplexity" : 0.1,
#                         "early_exaggeration" : 10.0,
#                         "metric" : "cosine"
#                         },
#     "reduce_first" : True,
    
#     "apply_standardisation" : {
#                                 "reducer" : False,
#                                 "clusterizer" : False
#                                 }
# }
@app.get("/get-clustering/")
async def get_clustering(model_type : str, 
                         reducer_type : str,
                         reduce_first : bool,
                        #  apply_standardisation : Dict[str, bool],
                         type_tokenizer : str,
                         link_data : str):


    CONFIG = {"model_type" : model_type,
    "model_kwargs" : {
                        "n_clusters" : 3,
                        "init" : "k-means++",
                        "n_init" : 2,
                        },
    "reducer_type" : reducer_type,
    "reducer_kwargs" : {
                        "n_components" : 2,
                        },
    "reduce_first" : reduce_first,
    "apply_standardisation" : {
                                "reducer" : False,
                                "clusterizer" : False
                                }
    }
    # Adjust config
    model_type = {"Kmeans" : KMeans}
    reducer_type = {'PCA' : PCA}


    CONFIG["model_type"] = model_type[CONFIG["model_type"]]
    CONFIG["reducer_type"] = reducer_type[CONFIG["reducer_type"]]


    df = pd.read_json(link_data)
    my_emb = DocEmbeddings(shuffle(df.iloc[:20]))
    prepared_embeddings = my_emb.encode_text(type_tokenizer=type_tokenizer)

    kmeans = Clusterizer(prepared_embeddings)
    kmeans.run(**CONFIG)
    metrics = kmeans.compute_metrics()
    return metrics.get_string()