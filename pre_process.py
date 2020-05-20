import math
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from itertools import combinations
import networkx
from tqdm import tqdm

import util

IMDB_FILE = 'data/labeledTrainData.tsv'


def load_data(file, max_num=None):
    if util.is_exist("data.pkl"):
        print("loaded")
        return util.load_pickle("data.pkl")
    labels = []
    reviews = []
    ids = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        max_line = len(lines) if max_num is None else max_num + 1
        # Ignore Header Line
        for line in lines[1:max_line]:
            id, label, review = line.split("\t")
            labels.append(label)
            # remove quotation marks around
            review = review[1:len(review) - 1]
            reviews.append(review)
            # remove quotation marks around
            id = id[1:len(id) - 1]
            ids.append(id)

    reviews = [cleanText(r) for r in reviews]

    util.save_pickle((ids, labels, reviews), "data.pkl")
    return ids, labels, reviews


def cleanText(text):
    text = text.lower()
    # remove html tags
    text = re.sub('<.*?>', '', text)
    # remove punctuations
    text = re.sub('[^a-z\']', ' ', text)
    # remove extra space
    text = re.sub('[ ]+', ' ', text)
    return text


def calc_tf_idf(reviews, min_df=0.01):
    if util.is_exist("tf-idf.pkl"):
        print("loaded")
        return util.load_pickle("tf-idf.pkl")
    vectorizer = TfidfVectorizer(input="content", stop_words=stopwords.words("english"), min_df=min_df, max_df=0.5)
    vectorizer.fit(reviews)
    tfidf = vectorizer.transform(reviews).toarray()
    vocab = vectorizer.get_feature_names()
    util.save_pickle((tfidf, vocab), "tf-idf.pkl")
    return tfidf, vocab


# 可优化,预先转为index
def calc_pmi(reviews, vocabs, window_size=10):
    if util.is_exist("pmi.pkl"):
        print("loaded")
        return util.load_pickle("pmi.pkl")

    word2index = {word: index for index, word in enumerate(vocabs)}

    W = 1  # 防止出现p = 1 的情况
    W_i = np.zeros(len(vocabs), dtype=np.int32)
    W_ij = np.identity(len(vocabs), dtype=np.int32)

    vocabs = set(vocabs)
    words_list = [[w for w in r.split() if w in vocabs] for r in reviews]

    for word_seq in tqdm(words_list, total=len(words_list)):

        for i in range(max(len(word_seq) - window_size, 1)):
            W += 1
            word_set = set(word_seq[i:i + window_size])
            for w in word_set:
                W_i[word2index[w]] += 1

            for w1, w2 in combinations(word_set, 2):
                i1 = word2index[w1]
                i2 = word2index[w2]

                W_ij[i1][i2] += 1
                W_ij[i2][i1] += 1

    p_i = W_i / W
    p_ij = W_ij / W
    val = np.zeros(p_ij.shape, dtype=np.float64)

    for i in range(len(p_i)):
        for j in range(len(p_i)):
            if p_ij[i, j] != 0 and p_i[i] * p_i[j] != 0:
                val[i, j] = math.log(p_ij[i, j] / (p_i[i] * p_i[j]))
    util.save_pickle(val, "pmi.pkl")
    return val


def build_graph(ids, vocabs, pmi, tfidf):
    if util.is_exist("graph.pkl"):
        print("loaded")
        return util.load_pickle("graph.pkl")
    G = networkx.Graph()
    G.add_nodes_from(ids)
    G.add_nodes_from(vocabs)

    cn2 = lambda x: x * (x - 1) // 2
    print("Calculating word_word edges")
    for (i, w1), (j, w2) in tqdm(combinations(enumerate(vocabs), 2), total=cn2(len(vocabs))):
        if pmi[i][j] > 0:
            G.add_edge(w1, w2, weight=pmi[i][j])

    print("Calculating doc_word edges")
    for i, review_id in tqdm(enumerate(ids), total=len(ids)):
        for j, word in enumerate(vocabs):
            G.add_edge(review_id, word, weight=tfidf[i][j])

    print("Calculating doc_doc edges")
    for review_id in tqdm(ids, total=len(ids)):
        G.add_edge(review_id, review_id, weight=1)

    util.save_pickle(G, "graph.pkl")
    return G


def preprocess():
    if util.is_exist("preprocessed.pkl"):
        print("loading")
        return util.load_pickle("preprocessed.pkl")
    _, labels, _ = util.load_pickle("data.pkl")
    labels = np.array(labels, dtype=np.int32)
    G = util.load_pickle("graph.pkl")

    print("calc adjacent matrix")
    A = networkx.to_numpy_matrix(G, weight="weight")

    print("calc degree matrix")
    degrees = [d ** -0.5 if d != 0 else 0 for _, d in G.degree]

    print("normalize adjacent matrix")
    '''
    degrees = np.diag(degrees)
    A_hat = degrees @ A @ degrees
    '''
    # decrease memory allocation
    A_hat = A
    for i in tqdm(range(A.shape[0]), total=A.shape[0]):
        for j in range(A.shape[1]):
            A_hat[i, j] *= degrees[i] * degrees[j]

    print("calc feature matrix")
    X = np.eye(G.number_of_nodes())  # Features are just identity matrix
    util.save_pickle((X, A_hat, labels), "preprocessed.pkl")
    return X, A_hat, labels


# 计算特征,计算tf-idf, 计算 pmi,
def main():
    print("Loading Data...")
    ids, labels, reviews = load_data(IMDB_FILE, 5000)
    print("TF_IDF...")
    tfidf, vocabs = calc_tf_idf(reviews)
    print("Reviews: {}, Words: {}".format(len(reviews), len(vocabs)))
    print("PMI...")
    pmi = calc_pmi(reviews, vocabs)
    print("Building Graph...")
    build_graph(ids, vocabs, pmi, tfidf)
    print("Calculating Graph Structure...")
    preprocess()
    print("Preprocess Done.")


if __name__ == '__main__':
    main()
