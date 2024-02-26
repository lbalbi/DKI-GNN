import numpy
import random
import os
import pandas as pd
from operator import itemgetter

import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import json

from pyrdf2vec.graphs import kg
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec

from pyrdf2vec.samplers import (  # isort: skip
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler,
    RandomSampler,)
from pyrdf2vec.walkers import RandomWalker, WeisfeilerLehmanWalker


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)



########################################################################################################################
##############################################         Construct KGs        ############################################
########################################################################################################################
def process_correspondence_file(correspondence_file_path):
    dic_nodeidx2ID, dic_ID2nodeidx  = {}, {}
    with open(correspondence_file_path, 'r') as csv_file:
        csv_file.readline()
        for line in csv_file:
            node_idx, ID = line.strip().split(',')
            if "GO:" in ID:
                dic_nodeidx2ID[node_idx] = ID.replace("GO:", "http://purl.obolibrary.org/obo/GO_")
                # dic_ID2nodeidx[ID.replace("GO:", "http://purl.obolibrary.org/obo/GO_")] = node_idx
            else:
                dic_nodeidx2ID[node_idx] = ID
                # dic_ID2nodeidx[ID] = node_idx

    return dic_nodeidx2ID

def construct_kg(ontology_file_path, annotations_file_path, correspondence_file_path):

    dic_nodeidx2ID = process_correspondence_file(correspondence_file_path)
    ents = [dic_nodeidx2ID[key] for key in dic_nodeidx2ID if "GO" not in dic_nodeidx2ID[key]]  ## proteinas

    # Include Ontology
    g = rdflib.Graph()
    g.parse(ontology_file_path, format='xml')

    for ent in ents:
        g.add((rdflib.term.URIRef(ent),RDF.type,rdflib.term.URIRef("http://www.w3.org/2002/07/owl#NamedIndividual")))

    # Include Ontology Annotations
    with open(annotations_file_path , 'r') as file_annot:
        file_annot.readline()
        for annot in file_annot:
            idx1, idx2 = annot.strip().split(",")
            g.add((rdflib.term.URIRef(dic_nodeidx2ID[idx1]), rdflib.term.URIRef('http://hasAnnotation') , rdflib.term.URIRef(dic_nodeidx2ID[idx2])))


    return g, ents


########################################################################################################################
##############################################      Compute Embeddings      ############################################
########################################################################################################################

def calculate_embeddings(g, ents, path_output,  size_value, type_word2vec, n_walks , walk_depth, walker_type, sampler_type):

    graph = kg.rdflib_to_kg(g)
    del g

    if type_word2vec == 'CBOW':
        sg_value = 0
    if type_word2vec == 'skip-gram':
        sg_value = 1

    print('----------------------------------------------------------------------------------------')
    print('Vector size: ' + str(size_value))
    print('Type Word2vec: ' + type_word2vec)

    if sampler_type.lower() == 'uniform':
        sampler = UniformSampler()
    elif sampler_type.lower() == 'predfreq':
        sampler = PredFreqSampler()
    elif sampler_type.lower() == 'objfreq':
        sampler = ObjFreqSampler()
    elif sampler_type.lower() == 'objpredfreq':
        sampler = ObjPredFreqSampler()
    elif sampler_type.lower() == 'pagerank':
        sampler = PageRankSampler()
    elif sampler_type.lower() == 'random':
        sampler = RandomSampler()

    if walker_type.lower() == 'random':
        walker = RandomWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'wl':
        walker = WeisfeilerLehmanWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'anonymous':
        walker = AnonymousWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'halk':
        walker = HalkWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'ngram':
        walker = NGramWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'walklet':
        walker = WalkletWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)

    del sampler
    w2v = Word2Vec(vector_size=size_value, sg=sg_value)
    transformer = RDF2VecTransformer(w2v, walkers=[walker])
    print("transformer")
    embeddings = transformer.fit_transform(graph, ents)
    print("transformed")
    with open(path_output + 'Embeddings_RDF2VEC_' + str(type_word2vec) + '_' + walker_type + '_' + str(size_value) + '.txt', 'w') as file:
        file.write("{")
        first = False
        for i in range(len(ents)):
            if first:
                file.write(", '%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
            else:
                file.write("'%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
                first = True
            file.flush()
        file.write("}")



########################################################################################################################
##############################################        Call Embeddings       ############################################
########################################################################################################################

def run_embedddings(ontology_file_path, annotations_file_path, correspondence_file_path, vector_size, type_word2vec, n_walks, walk_depth, walker_type, sampler_type, path_output):

    g, ents = construct_kg(ontology_file_path, annotations_file_path, correspondence_file_path)
    calculate_embeddings(g, ents, path_output, vector_size, type_word2vec, n_walks, walk_depth, walker_type, sampler_type)


import argparse
if __name__== '__main__':
    #################################### Parameters ####################################
    parser = argparse.ArgumentParser(description='OGBL-PPA rdf2vec')
    parser.add_argument('--vector_size', type=int, default=50)
    parser.add_argument('--n_walks', type=int, default=80)
    parser.add_argument('--type_word2vec', type=str, default="skip-gram")
    parser.add_argument('--walk_depth', type=int, default=4)
    parser.add_argument('--walker_type', type=str, default="wl")
    parser.add_argument('--sampler_type', type=str, default="uniform")
    parser.add_argument('--ontology_file_path', type=str, action="store", default="./go.owl")
    parser.add_argument('--annotations_file_path', type=str, action="store", default="./ogbn_proteins_CC_annotations.csv")
    parser.add_argument('--correspondence_file_path', type=str, action="store", default="./ogbn_proteins_nodeidx2proteinid.csv")
    parser.add_argument('--path_output', type=str, action="store", default="Embed/")
    args = parser.parse_args()
    print(args)
    run_embedddings(args.ontology_file_path, args.annotations_file_path, args.correspondence_file_path,
                    args.vector_size, args.type_word2vec, args.n_walks, args.walk_depth, args.walker_type, args.sampler_type, args.path_output)
