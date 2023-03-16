import string

import numpy as np
import pandas as pd

import pymorphy2 as py
from transformers import BertModel, AutoTokenizer
import gensim.downloader as api
import torch

class DocEmbeddings:
    """
    Language embedder for documents that prepares words either
    via word2vec or via BertTokenizer and then averages it for each separate
    document

    Example of usage:
    ```
    my_emb = MyDocEmbeddings(shuffle(df.iloc[:1000]))
    prepared_embeddings_bert = my_emb.encode_text(type_tokenizer="bert")
    ```

    """
    # Модели для Word2Vec
    tagger = py.MorphAnalyzer()
    w2v_pretrained_kv = api.load('word2vec-ruscorpora-300')

    # Модели для Bert Tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    bert_model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

    label_mapper = {
        'negative' : 0,
        'neutral' : 1,
        'positive' : 2
    }

    def __init__(self, df : pd.DataFrame):
        self.df = df.copy()

    def _delete_punctuation(self, text : str):
        """
        Удаляет пунктуацию из текста

        Удаленные знаки пунктуации:
        !"#$%&'()*+,-./:;<=>?@[\]^_`{|\}~
        """
        return str(text).translate(str.maketrans('', '', string.punctuation))

    def _tag_word(self, word):
        try:    
            parsed = self.tagger.parse(word)[0]
            # Падеж .case; выделяется у существительных, полных прилагательных, полных причастий, числительных и местоимений
            return word + '_' + parsed.tag.POS
        except:
            return word

    def _tag_text(self, text):
        return [self._tag_word(w) for w in text.split()]


    def _encode_text_v2w(self, text):
        # Prepare each word
        text = self._delete_punctuation(text).lower()
        tokenized_text = self._tag_text(text)

        vectors = np.zeros((300,))
        for word in tokenized_text:
            try:
                vectors += self.w2v_pretrained_kv[word]
            except KeyError:
                continue # do not add vector if there is no word in vocabulary

        return vectors.reshape(1, 300) / len(tokenized_text)


    def _encode_text_bert(self, text):
        input_ids = self.bert_tokenizer(text)['input_ids']

        last_hidden_state = self.bert_model(input_ids=torch.LongTensor(input_ids).unsqueeze(0))['last_hidden_state']

        return last_hidden_state.squeeze().mean(0)



    def encode_text(self, type_tokenizer="bert"):
        """
        Encodes documents by computing the embedding of each word and then
        averaging it
        """
        self.df['label'] = self.df['sentiment'].apply(lambda x: self.label_mapper[x])
        
        # Iterate through the dataset
        if type_tokenizer == "word2vec":
            embeddings = None
            for doc in self.df['text']:
                if embeddings is None:
                    embeddings = self._encode_text_v2w(doc)
                else:
                    embeddings = np.append(embeddings, self._encode_text_v2w(doc), axis=0)

        elif type_tokenizer == 'bert':
            embeddings = None
            for doc in self.df['text']:
                if embeddings is None:
                    embeddings = self._encode_text_v2w(doc)
                else:
                    embeddings = np.append(embeddings, self._encode_text_v2w(doc), axis=0)

        else:
            raise Exception("Only 2 types of tokenizer are available: bert and word2vec")

        return embeddings