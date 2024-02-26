import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
import os
import json
import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import sys

from OpenKE import config
from OpenKE import models


#####################
##    Functions    ##
#####################

def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: path-like object representing a file system path;
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def construct_model(dic_nodes, dic_relations, list_triples, path_output, model_embedding, vector_size):
    """
    Construct and embedding model and compute embeddings.
    :param dic_nodes: dictionary with KG nodes and respective ids;
    :param dic_relations: dictionary with type of relations in the KG and respective ids;
    :param list_triples: list with triples of the KG;
    :param path_output: OpenKE path;
    :param n_embeddings: dimension of embedding vectors;
    :param models_embeddings: list of embedding models;
    :param vector_size: int representing the size of the embeddings;
    :return: write a json file with the embeddings for all nodes and relations.
    """
    entity2id_file = open(path_output + "entity2id.txt", "w")
    entity2id_file.write(str(len(dic_nodes))+ '\n')
    for entity , id in dic_nodes.items():
        entity = entity.replace('\n' , ' ')
        entity = entity.replace(' ' , '__' )
        entity = entity.encode('utf8')
        entity2id_file.write(str(entity) + '\t' + str(id)+ '\n')
    entity2id_file.close()

    relations2id_file = open(path_output + "relation2id.txt", "w")
    relations2id_file.write(str(len(dic_relations)) + '\n')
    for relation , id in dic_relations.items():
        relation  = relation.replace('\n' , ' ')
        relation = relation.replace(' ', '__')
        relation = relation.encode('utf8')
        relations2id_file.write(str(relation) + '\t' + str(id) + '\n')
    relations2id_file.close()

    train2id_file = open(path_output + "train2id.txt", "w")
    train2id_file.write(str(len(list_triples)) + '\n')
    for triple in list_triples:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[2]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()

    # Input training files from data folder.
    con = config.Config()
    con.set_in_path(path_output)
    con.set_dimension(vector_size)

    print('--------------------------------------------------------------------------------------------------------------------')
    print('MODEL: ' + model_embedding)

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(path_output + model_embedding + "/model.vec.tf", 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(path_output + model_embedding + "/embedding.vec.json")

    if model_embedding == 'ComplEx':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(0.5)
        con.set_lmbda(0.05)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        #Set the knowledge embedding model
        con.set_model(models.ComplEx)

    elif model_embedding == 'distMult':
        con.set_work_threads(8)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.5)
        con.set_lmbda(0.05)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.DistMult)

    elif model_embedding == 'HOLE':
        con.set_work_threads(4)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.1)
        con.set_bern(0)
        con.set_margin(0.2)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.HolE)

    elif model_embedding == 'RESCAL':
        con.set_work_threads(4)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.1)
        con.set_bern(0)
        con.set_margin(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.RESCAL)

    elif model_embedding == 'TransD':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(1.0)
        con.set_margin(4.0)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransD)

    elif model_embedding == 'TransE':
        con.set_work_threads(8)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransE)

    elif model_embedding == 'TransH':
        con.set_work_threads(8)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransH)

    elif model_embedding == 'TransR':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(1.0)
        con.set_lmbda(4.0)
        con.set_margin(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransR)

    # Train the model.
    con.run()


def write_embeddings(path_model_json, path_embeddings_output, ents, dic_nodes):
    """
    Writing embeddings.
    :param path_model_json: json file with the embeddings for all nodes and relations;
    :param path_embeddings_output: embedding file path;
    :param ents: list of entities for which embeddings will be saved;
    :param dic_nodes: dictionary with KG nodes and respective ids;
    :return: writes an embedding file with format "{ent1:[...], ent2:[...]}".
    """
    with open(path_model_json, 'r') as embeddings_file:
        data = embeddings_file.read()
    embeddings = json.loads(data)
    embeddings_file.close()

    ensure_dir(path_embeddings_output)
    with open(path_embeddings_output, 'w') as file_output:
        file_output.write("{")
        first = False
        for i in range(len(ents)):
            ent = ents[i]
            if first:
                if "ent_embeddings" in embeddings:
                    file_output.write(", '%s':%s" % (str(ent), str(embeddings["ent_embeddings"][dic_nodes[str(ent)]])))
                else:
                    file_output.write(
                        ", '%s':%s" % (str(ent), str(embeddings["ent_re_embeddings"][dic_nodes[str(ent)]])))
            else:
                if "ent_embeddings" in embeddings:
                    file_output.write("'%s':%s" % (str(ent), str(embeddings["ent_embeddings"][dic_nodes[str(ent)]])))
                else:
                    file_output.write(
                        "'%s':%s" % (str(ent), str(embeddings["ent_re_embeddings"][dic_nodes[str(ent)]])))
                first = True
        file_output.write("}")
    file_output.close()


########################################################################################################################
##############################################        Call Embeddings       ############################################
########################################################################################################################
def process_correspondence_file(correspondence_file_path):
    dic_nodeidx2ID, dic_ID2nodeidx  = {}, {}
    with open(correspondence_file_path, 'r') as csv_file:
        csv_file.readline()
        for line in csv_file:
            node_idx, ID = line.strip().split(',')
            if "GO:" in ID:
                dic_nodeidx2ID[node_idx] = ID.replace("GO:", "http://purl.obolibrary.org/obo/GO_")
            else:
                dic_nodeidx2ID[node_idx] = ID
    return dic_nodeidx2ID


def construct_kg(ontology_file_path, annotations_file_path, correspondence_file_path):

    dic_nodeidx2ID = process_correspondence_file(correspondence_file_path)
    ents = [dic_nodeidx2ID[key] for key in dic_nodeidx2ID if "GO" not in dic_nodeidx2ID[key]]

    # Include Ontology
    g = rdflib.Graph()
    g.parse(ontology_file_path, format='xml')
    for ent in ents:
        g.add((rdflib.term.URIRef(ent), RDF.type,rdflib.term.URIRef("http://www.w3.org/2002/07/owl#NamedIndividual")))

    # Include Ontology Annotations
    with open(annotations_file_path , 'r') as file_annot:
        file_annot.readline()
        for annot in file_annot:
            idx1, idx2 = annot.strip().split(",")
            g.add((rdflib.term.URIRef(dic_nodeidx2ID[idx1]), rdflib.term.URIRef('http://hasAnnotation') , rdflib.term.URIRef(dic_nodeidx2ID[idx2])))

    return g, ents


def buildIds(g):
    """
    Assigns ids to KG nodes and KG relations.
    :param g: knowledge graph;
    :return: 2 dictionaries and one list. "dic_nodes" is a dictionary with KG nodes and respective ids. "dic_relations" is a dictionary with type of relations in the KG and respective ids. "list_triples" is a list with triples of the KG.
    """
    dic_nodes = {}
    id_node = 0
    id_relation = 0
    dic_relations = {}
    list_triples = []

    for (subj, predicate, obj) in g:
        if str(subj) not in dic_nodes:
            dic_nodes[str(subj)] = id_node
            id_node = id_node + 1
        if str(obj) not in dic_nodes:
            dic_nodes[str(obj)] = id_node
            id_node = id_node + 1
        if str(predicate) not in dic_relations:
            dic_relations[str(predicate)] = id_relation
            id_relation = id_relation + 1
        list_triples.append([dic_nodes[str(subj)], dic_relations[str(predicate)], dic_nodes[str(obj)]])

    return dic_nodes, dic_relations, list_triples


def run_embedddings(ontology_file_path, annotations_file_path, correspondence_file_path, vector_size, path_embedding, embeddings):
    print(embeddings)
    g, ents = construct_kg(ontology_file_path, annotations_file_path, correspondence_file_path)
    dic_nodes, dic_relations, list_triples = buildIds(g)
    path_output = 'OpenKE/'

    for model_embedding in embeddings:
        print(model_embedding)
        ensure_dir(path_output + model_embedding)
        construct_model(dic_nodes, dic_relations, list_triples, path_output, model_embedding, vector_size)
        path_model_json = path_output + model_embedding + "/embedding.vec.json"
        path_embeddings_output = path_embedding + 'Embeddings_' + model_embedding + '_' + str(vector_size) + '.txt'
        write_embeddings(path_model_json, path_embeddings_output, ents, dic_nodes)


import argparse
if __name__== '__main__':
    parser = argparse.ArgumentParser(description='OGBL-PPA opa2vec')
    parser.add_argument('--embeddings', action="store", default=['TransE',"ComplEx", "distMult"])
    parser.add_argument('--vector_size', type=int, default=50)
    parser.add_argument('--ontology_file_path', type=str, action="store", default="./go.owl")
    parser.add_argument('--path_embedding', type=str, action="store", default="./")
    parser.add_argument('--annotations_file_path', type=str, action="store", default="./ogbn_proteins_CC_annotations.csv")
    parser.add_argument('--correspondence_file_path', type=str, action="store", default="./ogbn_proteins_nodeidx2proteinid.csv")
    args = parser.parse_args()
    print(args)
    import ast
    run_embedddings(args.ontology_file_path, args.annotations_file_path, args.correspondence_file_path, args.vector_size, args.path_embedding, ast.literal_eval(args.embeddings))



