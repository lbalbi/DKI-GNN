import torch,numpy
import pandas as pd
import rdflib, ast, torch
from tdc.multi_pred import PPI
from _2_eval_sim import *

def read_gaf_(a_file):
    data = pd.read_csv(a_file, skiprows=31, on_bad_lines="skip", header=None, index_col=False, sep="\t")
    dict_data = {}
    for i in range(len(data[0])):
        try:
            dict_data[data[1][i]].append(data[4][i])
        except:
            dict_data[data[1][i]] = [data[4][i]]
    return dict_data

def read_go_(file):
    g=rdflib.Graph()
    G = g.parse(file)
    lista = []
    for (i,j,k) in G:
        if "GO_" in i and "GO_" in k:
            lista.append([str(i).replace("http://purl.obolibrary.org/obo/GO_","GO:"),str(k).replace("http://purl.obolibrary.org/obo/GO_","GO:")])
    df = pd.DataFrame(lista)
    return df

def get_annotations_(prot_list,correspondence, a_dict):
    dict_corr,annot_list = {},[]
    for i in range(len(correspondence["From"])):
        try: dict_corr[correspondence["From"][i]].append(correspondence["Entry"][i])
        except: dict_corr[correspondence["From"][i]] = [correspondence["Entry"][i]]
    for i in range(len(prot_list[1])):
        if dict_corr.get(prot_list[1][i]) != None:
            for j in dict_corr[prot_list[1][i]]:
                if a_dict.get(j) != None:
                    for k in a_dict[j]:
                        annot_list.append([prot_list[1][i],k])
    annot_list = pd.DataFrame(annot_list)
    return annot_list

def read_data():
    data = PPI(name = 'HuRI')
    data = data.neg_sample(frac = 1)
    split = data.get_split()
    testdata = split["test"][["Protein1_ID", "Protein2_ID","Y"]].rename(columns = {"Protein1_ID":"Ent1","Protein2_ID":"Ent2","Y":"Label"})
    testpairs,testlabels,allproteins = testdata[["Ent1","Ent2"]].values, testdata[["Label"]].values,[]
    alldata = pd.concat([split["test"][["Protein1_ID", "Protein2_ID","Y"]].rename(columns = {"Protein1_ID":"Ent1","Protein2_ID":"Ent2","Y":"Label"}),split["train"][["Protein1_ID", "Protein2_ID","Y"]].rename(columns = {"Protein1_ID":"Ent1","Protein2_ID":"Ent2","Y":"Label"})], axis=0, ignore_index=True)
    alldata = pd.concat([alldata,split["valid"][["Protein1_ID", "Protein2_ID","Y"]].rename(columns = {"Protein1_ID":"Ent1","Protein2_ID":"Ent2","Y":"Label"})], axis=0, ignore_index=True)
    allpairs,alllabels = alldata[["Ent1","Ent2"]].values, alldata[["Label"]].values

    for (i,j) in allpairs:
        allproteins.append(i);allproteins.append(j)
    allproteins = list(set(allproteins))
    allproteins = pd.concat([pd.DataFrame([i for i in range(len(allproteins))]), pd.DataFrame(allproteins)], axis=1, ignore_index=True)
    dict_proteins = {}
    for (i,j) in allproteins.itertuples(index=False):
        dict_proteins[j] = i

    testpairs = pd.DataFrame([(dict_proteins[i],dict_proteins[j]) for (i,j) in testpairs])
    allpairs = pd.DataFrame([(dict_proteins[i],dict_proteins[j]) for (i,j) in allpairs])
    return testpairs, allpairs, allproteins, testlabels, alllabels
    
def all_idxs_(proteins,go):
    go_list = []
    for i in range(len(go[0])):
        go_list.append(go[0][i]);go_list.append(go[1][i])
    df_go = pd.DataFrame(list(set(go_list)))
    df_go_full = pd.concat([pd.DataFrame([i+len(proteins[0]) for i in range(len(df_go))]),df_go],axis=1, ignore_index=True)
    df_go_full = pd.concat([proteins,df_go_full],axis=0, ignore_index=True)
    return df_go_full

def trans_annotations_(annotations,idxs):
    dict_idxs, list_annots = {}, []
    try:
        for i in range(len(idxs[0])):
            dict_idxs[idxs[1][i]] = idxs[0][i]
    except:
        for i in range(len(idxs["node idx"])):
            dict_idxs[idxs["protein id"][i]] = idxs["node idx"][i]
    for k in range(len(annotations[0])):
        list_annots.append([dict_idxs[annotations[0][k]], dict_idxs[annotations[1][k]]])
    return pd.DataFrame(list_annots)

def process_embedding_file(embeddings_in_file,proteins):
    with open('{}'.format(embeddings_in_file), 'r') as f:
        mylist = ast.literal_eval(f.read())
    f.close()
    dict1,lista = {},[]
    all_idxs = pd.read_csv("nodeidx2proteinid.csv", index_col=False, nrows=len(proteins[0]))
    for i in range(len(all_idxs["node idx"])):
        dict1[all_idxs["node idx"][i]] = all_idxs["protein id"][i]
    for i in range(len(proteins[0])):
        if dict1.get(i) != None:
            if mylist.get(dict1[i]) != None:
                lista.append(mylist[dict1[i]])
    x = pd.DataFrame(lista)
    x = torch.from_numpy(x.values.astype(numpy.float32)).float()
    return x


import gzip
import shutil
def zip_file(file_name):
    with open(file_name, 'rb') as f_in:
        with gzip.open(file_name + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

import os
def make_ogb_dir():
    if not os.path.exists("dataset"):
        list_folders = ["raw", "processed", "mapping", "throughput", "dataset"]
        raw_files = ["edge.csv.gz", "node-feat.csv.gz", "num-node-list.csv.gz", "num-edge-list.csv.gz"]
        mapping_files = ["nodeidx2proteinid.csv.gz", "featdim2speciesid.csv.gz"]
        split_files = ["train.pt", "test.pt", "valid.pt"]
        for k in list_folders:
            os.makedirs(k)
        for k in raw_files:
            shutil.move("{}".format(k), "raw/{}".format(k))
        for k in mapping_files:
            shutil.move("{}".format(k), "mapping/{}".format(k))
        for k in split_files:
            shutil.move("{}".format(k), "throughput/{}".format(k))
        shutil.move("./throughput", "./ogbl_ppa/split/throughput")
        shutil.move("./raw", "./ogbl_ppa/raw")
        shutil.move("./mapping", "./ogbl_ppa/mapping")
        shutil.move("./processed", "./ogbl_ppa/processed")
        shutil.move("./ogbl_ppa", "./dataset/ogbl_ppa")
        with open("dataset/ogbl_ppa/RELEASE_v2.txt", "w") as f_:
            f_.write("## huri_dataset")
        f_.close()
        # if having issues with dataset folder location comment line below to keep dataset in current directory
        shutil.move("./dataset", "./Link_Prediction/dataset")


#######
def main_func():
    testpairs, allpairs, allproteins, testlabels, alllabels = read_data()
    go_df = read_go_("go.owl")
    dict_annots = read_gaf_("goa_human.gaf")
    correspondence = pd.read_csv("id_correspondence_true_neg.csv", index_col=False, sep=";").loc[:,["From","Entry"]]
    annot_list = get_annotations_(allproteins,correspondence,dict_annots)
    allidxs = all_idxs_(allproteins,go_df)
    allidxs.to_csv("nodeidx2proteinid.csv", header=["node idx", "protein id"], index=False)
    zip_file("nodeidx2proteinid.csv")
    annotations = trans_annotations_(annot_list,allidxs)
    annotations.to_csv("annotations.csv",index=False,header=False);allproteins.to_csv("allproteins_.csv",index=False,header=False)
    pd.DataFrame(testlabels).to_csv("testlabels_.csv",index=False,header=False);pd.DataFrame(alllabels).to_csv("alllabels_.csv", header=False, index=False)
    allpairs.to_csv("allpairs_.csv",index=False,header=False);testpairs.to_csv("testpairs_.csv",index=False,header=False)

    ###########################################################################
    edge = pd.read_csv("allpairs_.csv", header=None, index_col=False)
    edge = edge.iloc[:,:2]
    edge.to_csv("edge_.csv", index=False, header=False)
    nodes = pd.read_csv("nodeidx2proteinid.csv", index_col=False)
    edge = pd.read_csv("edge_.csv", header=None, index_col=False)
    labels = pd.read_csv("alllabels_.csv", header=None, index_col=False)

    dict_ = {}
    for i in range(len(edge)):
        dict_[(edge[0][i],edge[1][i])] = labels[0][i]
    edge = edge.sample(frac=1, random_state=0).reset_index(drop=True)
    edge_ = list(set(list(edge[0])))
    valid_list = edge_[:1*(len(edge_)//32)]
    test_list = edge_[1*(len(edge_)//32):2*(len(edge_)//32)]
    train_list = edge_[2*(len(edge_)//32):]
    train_pos,test_neg,test_pos,valid_neg,valid_pos = [],[],[],[],[]

    for i in range(len(edge)):
        if edge[0][i] in valid_list:
            if dict_[(edge[0][i],edge[1][i])] == 1:
                valid_pos.append([edge[0][i],edge[1][i]])
            if dict_[(edge[0][i],edge[1][i])] == 0:
                valid_neg.append([edge[0][i],edge[1][i]])
        if edge[0][i] in test_list:
            if dict_[(edge[0][i],edge[1][i])] == 1:
                test_pos.append([edge[0][i],edge[1][i]])
            if dict_[(edge[0][i],edge[1][i])] == 0:
                test_neg.append([edge[0][i],edge[1][i]])
        if edge[0][i] in train_list:
            if dict_[(edge[0][i],edge[1][i])] == 1:
                train_pos.append([edge[0][i],edge[1][i]])
    dict_train, dict_test,dict_valid = {},{},{}
    dict_train["edge"] = numpy.array(train_pos)
    dict_test["edge"], dict_test["edge_neg"] = numpy.array(test_pos), numpy.array(test_neg)
    dict_valid["edge"], dict_valid["edge_neg"] = numpy.array(valid_pos), numpy.array(valid_neg)
    torch.save(dict_train, "train.pt"), torch.save(dict_test, "test.pt"), torch.save(dict_valid, "valid.pt")

    list_feats = []
    for k in range(len(nodes["node idx"])):
        list_feats.append(k)
    feats = pd.DataFrame(list_feats)
    feats.to_csv("node-feat.csv", index=False, header=False)
    zip_file("node-feat.csv")
    train = torch.load("train.pt")
    train = pd.DataFrame(train["edge"])
    train.to_csv("edge.csv", index=False, header=False)
    zip_file("edge.csv")

    ### for compatibility with ogb pyglinkproppred dataset:
    with open("num-node-list.csv", "w") as f__:
        f__.write("{}".format(len(nodes["node idx"])))
    f__.close()
    with open("num-edge-list.csv", "w") as f__:
        f__.write("{}".format(len(train[0])))
    f__.close()
    zip_file("num-node-list.csv")
    zip_file("num-edge-list.csv")
    list_species = [9606 for i in range(len(nodes["node idx"]))]
    list_species = pd.DataFrame(list_species)
    list_species.to_csv("featdim2speciesid.csv", header=False, index=False)
    zip_file("featdim2speciesid.csv")
    make_ogb_dir()


##################################################################################################
# if  building files for performing tests over PPI dataset "huri", run:
main_func()
