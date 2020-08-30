# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:07:01 2020

@author: Hui Z
"""

from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import math

def get_doc(nlp, file_name, encoding_='utf-8'):
    return nlp(open(file_name, 'r', encoding=encoding_).read())

def get_sentences(doc):
    return [sent for sent in doc.sents]

def get_tokens_of_sentence(sentence):
    return [token for token in sentence]

def token_to_vector(token):
    return token.vector

def sentence_to_vectors(sentence):
    return [token_to_vector(token) for token in 
            get_tokens_of_sentence(sentence) 
            if not (token.is_punct or token.is_space)]

def sentence_mean_vector(sentence):
    sentence_vectors = sentence_to_vectors(sentence)
    if len(sentence_vectors) == 0:
        return np.zeros(300, np.float32)
    else:
        return np.mean(sentence_vectors, axis=0)
    
def sentences_similarity(sentence_l, sentence_r):
    sentence_l_mean_vector = sentence_mean_vector(sentence_l)
    sentence_r_mean_vector = sentence_mean_vector(sentence_r)
    has_empty_mean_vector = (np.dot(sentence_l_mean_vector, sentence_l_mean_vector) == 0) or (np.dot(sentence_r_mean_vector, sentence_r_mean_vector) == 0)
    if has_empty_mean_vector:
        return 0.0
    else:
        return 1 - cosine_distance(sentence_l_mean_vector, sentence_r_mean_vector)
    
def similar_sentence_pairs(sentences, threshold):
    result = []
    for idx1 in range(len(sentences)):
        for idx2 in range(idx1+1, len(sentences)):
            if sentences_similarity(
                    sentences[idx1], 
                    sentences[idx2]) > threshold:
                result.append((sentences[idx1], sentences[idx2]))
                
    return result

def similar_matrix(sentences, threshold):
    matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(idx1+1, len(sentences)):
            if sentences_similarity(sentences[idx1], sentences[idx2]) > threshold:
                matrix[idx1][idx2] += 1
                
    return matrix

def summarize(nlp, file_name, threshold=0.95, top_most=10):
    doc = get_doc(nlp, file_name)
    sents = get_sentences(doc)
    matrix = similar_matrix(sents, threshold)
    sum_by_row = [np.sum(row) for row in matrix]
    sorted_with_index = sorted(zip(sum_by_row, range(len(sum_by_row))), reverse=True)[:top_most]
    indices = sorted([e[1] for e in sorted_with_index])
    return [sents[i] for i in indices]

def summarize_pretty(nlp, file_name, threshold=0.95, top_most=10):
    sents = summarize(nlp, file_name, threshold, top_most)
    for i in range(len(sents)):
        print("{0}\n".format(sents[i].text.strip()))
        
def summarize_as_adj_edges(sents, threshold=0.95):
    vert_list = [i for i in range(len(sents))]
    edge_list = []
    for idx1 in range(len(sents)):
        for idx2 in range(len(sents)):
            if idx1 != idx2 and sentences_similarity(
                    sents[idx1], 
                    sents[idx2]) > threshold:
                edge_list.append((idx1, idx2))
                
    return (vert_list, edge_list)

def summarize_as_adj_G(sents, threshold=0.95):
    graph = summarize_as_adj_edges(sents, threshold)
    G = nx.Graph()
    for vert in graph[0]:
        G.add_node(vert)
        
    for edge in graph[1]:
        G.add_edge(edge[0], edge[1])
        
    return G
    
    
def summarize_with_adj_grahp(nlp, file_name, threshold=0.95):
    doc = get_doc(nlp, file_name)
    sents = get_sentences(doc)
    adj_graph = summarize_as_adj_G(sents, threshold)
    sub_graphs = nx.connected_components(adj_graph)
    max_sub_graph = max(sub_graphs, key=lambda x:len(x))
    return [sents[i] for i in max_sub_graph]