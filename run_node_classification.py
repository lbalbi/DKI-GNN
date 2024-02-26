import pandas as pd
import numpy as np
import torch
import sklearn.metrics as metrics
import sklearn
import ogb
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

def make_dataset(path, name_="ogbn-proteins"):
    dataset = PygNodePropPredDataset(root=path, name=name_, transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)
    split_edge = dataset.get_idx_split()
    del dataset
    data.x = data.x.to(torch.float)
    print(data)
    return data, split_edge, data.x


import subprocess
def make_embeddings(embedder, annot_type="CC"):
    if embedder == "wang2vec":
        if annot_type == "ALL": command_to_execute = ["python3", "Embeddings/run_Wang2VecEmbeddings.py","--annotations_file_path","./ogbn_proteins_ALL_annotations.csv"]
        elif annot_type == "R2":command_to_execute = ["python3", "Embeddings/run_Wang2VecEmbeddings.py","--annotations_file_path","./ogbn_proteins_R2_annotations.csv"]
        else:command_to_execute = ["python3", "Embeddings/run_Wang2VecEmbeddings.py"]
    if embedder == "transe":
        if annot_type == "ALL": command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["TransE"]',"--annotations_file_path","./ogbn_proteins_ALL_annotations.csv"]
        elif annot_type == "R2":command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["TransE"]',"--annotations_file_path","./ogbn_proteins_R2_annotations.csv"]
        else:command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["TransE"]']
    if embedder == "complex":
        if annot_type == "ALL": command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["ComplEx"]',"--annotations_file_path","./ogbn_proteins_ALL_annotations.csv"]
        if annot_type == "R2": command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["ComplEx"]',"--annotations_file_path","./ogbn_proteins_R2_annotations.csv"]
        else: command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["ComplEx"]']
    if embedder == "distmult":
        if annot_type == "ALL": command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["distMult"]',"--annotations_file_path","./ogbn_proteins_ALL_annotations.csv"]
        elif annot_type == "R2": command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["distMult"]',"--annotations_file_path","./ogbn_proteins_R2_annotations.csv"]
        else: command_to_execute = ["python3", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["distMult"]']
    if embedder == "rdf2vec":
        if annot_type == "ALL": command_to_execute = ["python3", "Embeddings/run_RDF2VecEmbeddings.py","--annotations_file_path","./ogbn_proteins_ALL_annotations.csv"]
        elif annot_type == "R2": command_to_execute = ["python3", "Embeddings/run_RDF2VecEmbeddings.py","--annotations_file_path","./ogbn_proteins_R2_annotations.csv"]
        else: command_to_execute = ["python3", "Embeddings/run_RDF2VecEmbeddings.py"]

    subprocess.run(command_to_execute)

    print("{} embeddings done!".format(embedder)) # save embeddings in file

import ast
def get_embeddings(path_, annot_type="CC", embedder="", x_=None):
    if embedder == "wang2vec":
        embedding_file = "Embeddings/Embeddings_Wang2VECskip-gram_wl_50.txt"
    elif embedder == "transe":
        embedding_file = "Embeddings/Embeddings_TransE_50.txt"
    elif embedder == "complex":
        embedding_file = "Embeddings/Embeddings_ComplEx_50.txt"
    elif embedder == "distmult":
        embedding_file = "Embeddings/Embeddings_distMult_50.txt"
    elif embedder == "rdf2vec":
        embedding_file = "Embeddings/Embeddings_RDF2VEC_skip-gram_wl_50.txt"

    from pathlib import Path
    if embedder != "" and Path(path_ + "/embedding_{}.pt".format(embedder)).exists() == False:
        print(path_ + "/embedding_{}.pt".format(embedder))
        print("processing {} embedings into torch tensor format".format(embedder))
        with open('{}'.format(path_ + "/" + embedding_file), 'r') as f:
            mylist = ast.literal_eval(f.read())
        f.close()

        dict1,lista = {},[]
        all_idxs = pd.read_csv(path_ + "/" + annot_type + "ogbn_proteins_nodeidx2proteinid.csv", index_col=False)
        for i in range(len(all_idxs["node idx"])):
            dict1[all_idxs["node idx"][i]] = all_idxs["protein id"][i]
        for i in range(len(all_idxs["node idx"])):
            if dict1.get(i) != None:
                if mylist.get(dict1[i]) != None:
                    lista.append(mylist[dict1[i]])
        del mylist, dict1, all_idxs
        embedding = pd.DataFrame(lista)
        embedding = torch.from_numpy(embedding.values.astype(np.float32)).float()
        torch.save(embedding ,path_ + "/embedding_{}.pt".format(embedder))
    else: print("loading embeddings file ...")
    
    if  embedder != "" and x_ != None:
        x = torch.cat([x_, torch.load(path_ + "/embedding_{}.pt".format(embedder))], dim=-1)
    elif embedder != "" and x_ == None:
        x = torch.load(path_ + "/embedding_{}.pt".format(embedder))
    elif embedder == "" and x_ != None:
        x = x_
    try:    
        print(x)
        print(len(x))
        print(len(x[0]))
    except:pass
    return x.to("cuda")

def run_neural(dl_model, embeddings = None, data_path="./dataset"):
    if dl_model == "GCN":
        command_to_execute = ["python3", "Node_Classification/gnn_aaai.py", "--K1", "{}".format(embeddings), "--runs", "5"]
    elif dl_model == "GraphSAGE":
        command_to_execute = ["python3", "Node_Classification/gnn_aaai.py", "--use_sage", "--K1", "{}".format(embeddings), "--runs", "5"]
    elif dl_model == "NGNN_GCN":
        command_to_execute = ["python3", "Node_Classification/gnn_aaai.py", "--ngnn_gcn", "--K1", "{}".format(embeddings), "--runs", "5"]
    elif dl_model == "NGNN_GraphSAGE":
        command_to_execute = ["python3", "Node_Classification/gnn_aaai.py", "--ngnn_sage", "--K1", "{}".format(embeddings), "--runs", "5"]

    elif dl_model == "MWE_DGCN":
        command_to_execute = ["python3", "GAT/main_proteins_full_dgl.py", "--model", "MWE-DGCN", "--K1", "{}".format(embeddings)]
    elif dl_model == "GAT":
        command_to_execute = ["python3", "GAT/gat.py", "--K1", "{}".format(embeddings)]
    elif dl_model == "AGDN":
        command_to_execute = ["python3", "AGDN/main.py", "--model", "agdn", "--K1", "{}".format(embeddings),
                              "--sample-type", "random_cluster", "--train-partition-num", "6", "--eval-partition-num", "2", "--eval-times", "1",
                              "--lr", "0.01", "--advanced-optimizer", "--n-epochs", "2000", "--n-heads", "5", "--n-layers", "5",
                              "--weight-style", "HC", "--dropout", "0.0", "--n-hidden", "128", "--input-drop", "0.1", "--attn-drop", "0.",
                              "--hop-attn-drop", "0.", "--edge-drop", "0.1", "--norm", "none", "--K", "2"]
    elif dl_model == "DeeperGCN":
        command_to_execute = ["python3", "deep_gcns_torch-master/examples/ogb/ogbn_proteins/main.py", 
                              "--use_gpu","--cluster_number", "10", "--valid_cluster_number", "5", "--aggr", "add", "--block", "plain", 
                              "--conv", "gen", "--gcn_aggr", "max", "--num_layers", "3", "--mlp_layers", "2", "--norm", "layer", "--hidden_channels", "64",
                              "--epochs", "200", "--lr", "0.001", "--dropout", "0.0", "--num_evals", "1", "--nf_path", "{}".format(embeddings)]
    subprocess.run(command_to_execute)
    print("{} model was run!".format(dl_model))

import argparse

def main():
    parser = argparse.ArgumentParser(description='DKI pipeline')
    parser.add_argument('--data_path', type=str, action="store", default="./dataset")
    parser.add_argument('--dataset', type=str, action="store", default="ogbn-proteins")
    parser.add_argument('--model', type=str, action="store", default="GCN")
    parser.add_argument('--compute_embedding', action="store_true")
    parser.add_argument('--use_ppi', action="store_true")
    parser.add_argument('--randomize', action="store_true")
    parser.add_argument('--embedding', type=str, action="store", default="rdf2vec")
    parser.add_argument('--annot_type', type=str, action="store", default="CC", choices=["ALL","R2","CC"])
    parser.add_argument('--combine', action="store_true")
    args = parser.parse_args()
    print(args)

    data, split_edge, x = make_dataset(path = args.data_path, name_=args.dataset)

    if args.compute_embedding:
        make_embeddings(embedder = args.embedding)

    from pathlib import Path
    if args.randomize == True and Path(args.data_path + "/embedding_random.pt").exists() == False:
        print("randomizing features : ")
        x = torch.Tensor(np.random.rand(len(x),50))
        torch.save(x, args.data_path + "/embedding_random.pt")
        print(x)
        print(len(x), len(x[0]))
        x_ = get_embeddings(path_ = args.data_path, embedder = args.embedding, x_ = x)
    elif args.randomize == True and Path(args.data_path + "/embedding_random.pt").exists() == True:
        x = torch.load(args.data_path + "/embedding_random.pt")
        print(x)
        print(len(x), len(x[0]))
        x_ = get_embeddings(path_ = args.data_path, embedder = args.embedding, x_ = x)
    elif args.randomize == False and args.embedding != "": x_ = get_embeddings(path_ = args.data_path, embedder = args.embedding)
    torch.save(x_, "embedding_{}_{}.pt".format(args.embedding,args.model))
    if args.use_ppi == True and args.combined == True: run_neural(embeddings = "embedding_{}_{}.pt".format(args.embedding,args.model), dl_model = args.model, data_path=args.data_path, combined = True)
    else: run_neural(embeddings = "embedding_{}_{}.pt".format(args.embedding,args.model), dl_model = args.model, data_path=args.data_path)

if __name__ == "__main__":
    main()


#################################
# python3 run_node_classification.py --model GCN --compute_embedding --embedding rdf2vec
# python3 run_node_classification.py --model GCN --compute_embedding --use_ppi --embedding wang2vec
# python3 run_node_classification.py --model NGNN_GCN --embedding --use_ppi