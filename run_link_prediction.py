import pandas as pd
import numpy as np
import torch
import sklearn.metrics as metrics
import sklearn
import ogb
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

def make_dataset(path, name_="ogbl-ppa"):
    if name_ == "ogbl-ppa":
        dataset = PygLinkPropPredDataset(root=path, name=name_, transform=T.ToSparseTensor())
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        del dataset
        x = data.x.to(torch.float)

    return data, split_edge, data.x

def _eval_hits(y_pred_pos, y_pred_neg, K):

    if len(y_pred_neg) < K:
        return 1.
    try:
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)
    except:
        kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
        hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)
    return hitsK
    
def predictions(predicted_labels, list_labels):
    try: 
        list_labels = [float(i) for i in list_labels.str.strip("tensor()").values]
    except:
        list_labels = list_labels.numpy()

    predicted_labels = predicted_labels[:, 1]
    predicted_labels_ = sklearn.preprocessing.binarize(predicted_labels.reshape(-1, 1), threshold=0.5, copy=True)
    print(predicted_labels_)
    waf = metrics.f1_score(list_labels, predicted_labels_, average='binary')
    precision = metrics.precision_score(list_labels, predicted_labels_, average = "binary")
    recall = metrics.recall_score(list_labels, predicted_labels_, average = "binary")
    accuracy = metrics.accuracy_score(list_labels, predicted_labels_)
    hits_50 = _eval_hits(predicted_labels[:len(predicted_labels)//2], predicted_labels[len(predicted_labels)//2:],50)
    hits_100 = _eval_hits(predicted_labels[:len(predicted_labels)//2], predicted_labels[len(predicted_labels)//2:],100)
    return hits_50, hits_100, waf, precision, recall, accuracy

def writePredictions(predictions, y, path_output):
    file_predictions = open(path_output, 'w')
    file_predictions.write('Predicted_output' + '\t' + 'Expected_Output' + '\n')
    for i in range(len(y)):
        file_predictions.write(str(predictions[i]) + '\t' + str(y[i]) + '\n')
    file_predictions.close()

import subprocess
def make_embeddings(embedder, dataset):
    if embedder == "wang2vec":
        if dataset == "ogbl-ppa": command_to_execute = ["python", "Embeddings/run_Wang2VecEmbeddings.py", "--annotations_file_path","./ogbl_ppa_annotations.csv","--correspondence_file_path","./ogbl_ppa_nodeidx2proteinid.csv"]
        else: command_to_execute = ["python", "Embeddings/run_Wang2VecEmbeddings.py", "--annotations_file_path", "./annotations.csv","--correspondence_file_path","./nodeidx2proteinid.csv"]
    if embedder == "transe":
        if dataset == "ogbl-ppa": command_to_execute = ["python", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["TransE"]', "--annotations_file_path","./ogbl_ppa_annotations.csv","--correspondence_file_path","./ogbl_ppa_nodeidx2proteinid.csv"]
        else: command_to_execute = ["python", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["TransE"]', "--annotations_file_path","./annotations.csv","--correspondence_file_path","./nodeidx2proteinid.csv"]
    if embedder == "complex":
        if dataset == "ogbl-ppa": command_to_execute = ["python", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["ComplEx"]', "--annotations_file_path","./ogbl_ppa_annotations.csv","--correspondence_file_path","./ogbl_ppa_nodeidx2proteinid.csv"]
        else: command_to_execute = ["python", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["ComplEx"]', "--annotations_file_path","./annotations.csv","--correspondence_file_path","./nodeidx2proteinid.csv"]
    if embedder == "distmult":
        if dataset == "ogbl-ppa": command_to_execute = ["python", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["distMult"]', "--annotations_file_path", "./ogbl_ppa_annotations.csv","--correspondence_file_path","./ogbl_ppa_nodeidx2proteinid.csv"]
        else: command_to_execute = ["python", "Embeddings/run_OpenKEembeddings.py", "--embeddings", '["distMult"]', "--annotations_file_path","./annotations.csv","--correspondence_file_path","./nodeidx2proteinid.csv"]
    if embedder == "rdf2vec":
        if dataset == "ogbl-ppa": command_to_execute = ["python", "Embeddings/run_RDF2VecEmbeddings.py", "--annotations_file_path","./ogbl_ppa_annotations.csv" ,"--correspondence_file_path","./ogbl_ppa_nodeidx2proteinid.csv"]
        else: command_to_execute = ["python", "Embeddings/run_RDF2VecEmbeddings.py", "--annotations_file_path","./annotations.csv","--correspondence_file_path","./nodeidx2proteinid.csv"]

    subprocess.run(command_to_execute)

    print("{} embeddings done!".format(embedder)) # save embeddings in file

import ast
def get_embeddings(path_, dataset, embedder="", partitions=None, x_=None):
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
        if dataset == "ogbl-ppa":
            all_idxs = pd.read_csv(path_ + "Link_Prediction/ogbl_ppa_nodeidx2proteinid.csv", index_col=False)
        else: all_idxs = pd.read_csv(path_ + "/nodeidx2proteinid.csv", index_col=False)
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

    if partitions != None:
        pos_train_preds, pos_valid_preds, pos_test_preds = [],[],[]
        train_edge = partitions['train']['edge']
        neg_train_preds = torch.randint(0, x.size(0), train_edge.size(), dtype=torch.long)
        train_edge_ = torch.cat([train_edge, neg_train_preds], dim=0)
        valid_edge = torch.cat([partitions['valid']['edge'], partitions['valid']['edge_neg']], dim=0)
        test_edge = torch.cat([partitions['test']['edge'], partitions['test']['edge_neg']], dim=0)
        for h in range(len(train_edge_)):
            pos_train_preds.append(x[train_edge_[h][0]] * x[train_edge_[h][1]])
        train_pred = torch.stack((pos_train_preds))

        del train_edge,train_edge_, pos_train_preds
        for h in range(len(valid_edge)):
            pos_valid_preds.append(x[valid_edge[h][0]] * x[valid_edge[h][1]])
        valid_pred = torch.stack((pos_valid_preds))
        del valid_edge, pos_valid_preds

        for h in range(len(test_edge)):
            pos_test_preds.append(x[test_edge[h][0]] * x[test_edge[h][1]])
        test_pred = torch.stack((pos_test_preds))
        del test_edge, pos_test_preds
 
        y_train = torch.cat([torch.ones(len(partitions['train']['edge'])),torch.zeros(len(partitions['train']['edge']))])
        y_valid = torch.cat([torch.ones(len(partitions['valid']['edge'])),torch.zeros(len(partitions['valid']['edge_neg']))]) 
        y_test = torch.cat([torch.ones(len(partitions['test']['edge'])),torch.zeros(len(partitions['test']['edge_neg']))])
        return train_pred, valid_pred, test_pred, y_train, y_valid, y_test
    return x.to("cuda")

def run_ML(X_train, y_train, X_valid, y_valid, X_test, y_test, ML_model, path_output_predictions, embedder, cv=False):

    if ML_model == "MLP":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(256,), activation='relu',learning_rate_init=0.01, solver='adam',)
        #clf = GridSearchCV(model, {'max_depth': [2,4,6, None],'n_estimators': [50,100,200]})
        model.fit(X_train, y_train)
        print()
        print("finished training! ")
        predictions_test = model.predict_proba(X_test)
        predictions_valid = model.predict_proba(X_valid)   
        writePredictions(predictions_valid, y_valid, path_output_predictions + '{}_{}__ValidSet_v3'.format(ML_model,embedder))
        writePredictions(predictions_test, y_test, path_output_predictions + '{}_{}__TestSet_v3'.format(ML_model,embedder))
        hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v = predictions(predictions_valid, y_valid)
        hits_50_t, hits_100_t, waf_t, precision_t, recall_t, accuracy_t = predictions(predictions_test, y_test)
    
        #params = list(product(parameters[''], parameters['']))
        #list_res = []
        # for p in params:
        #     model = MLPClassifier()
        #     model.fit(X_train, y_train)
        #     predictions_valid = model.predict(X_valid)
        #     hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v = predictions(predictions_valid, y_valid)
        #     list_res.append([p[0], p[1], hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v])
        # print()
        # k_ = 0
        # for k in list_res:
        #     if k[1] > k_:
        #         k_ = k[1]
        #         best_depth, best_estimators, hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v = k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]
        # print("Best results and parameters: ", best_depth, best_estimators, hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v)
        # X_train_ = torch.cat([X_train, X_valid], dim=0)
        # y_train_ = torch.cat([y_train, y_valid], dim=0)
        # print(X_train_)
        # model = model = MLPClassifier()
        # model.fit(X_train_, y_train_)
        # predictions_test = model.predict(X_test)
        # predictions_train = model.predict(X_train_)
        # writePredictions(predictions_train, y_train_, path_output_predictions + '{}_{}__TrainSet_v2'.format(ML_model,embedder))
        # writePredictions(predictions_test, y_test, path_output_predictions + '{}_{}__TestSet_v2'.format(ML_model,embedder))
        # hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v = predictions(predictions_train, y_train_)
        # hits_50_t,hits_100_t, waf, precision, recall, accuracy = predictions(predictions_test, y_test)

        return hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v, hits_50_t,hits_100_t, waf_t, precision_t, recall_t, accuracy_t
    


    elif ML_model == "RF":
        from sklearn.ensemble import RandomForestClassifier
        from itertools import product

        if cv == False:
            #from sklearn.model_selection import GridSearchCV
            print("Model selected:  Random Forest without CV GRID SEARCH")
            model = RandomForestClassifier()
            #clf = GridSearchCV(model, {'max_depth': [2,4,6, None],'n_estimators': [50,100,200]})
            model.fit(X_train, y_train)
            print()
            print("finished training! ")
            predictions_test = model.predict_proba(X_test)
            predictions_valid = model.predict_proba(X_valid)            
            writePredictions(predictions_valid, y_valid, path_output_predictions + '{}_{}__ValidSet_v3'.format(ML_model,embedder))
            writePredictions(predictions_test, y_test, path_output_predictions + '{}_{}__TestSet_v3'.format(ML_model,embedder))
            hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v = predictions(predictions_valid, y_valid)
            hits_50_t, hits_100_t, waf_t, precision_t, recall_t, accuracy_t = predictions(predictions_test, y_test)
            return hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v, hits_50_t,hits_100_t, waf_t, precision_t, recall_t, accuracy_t


        if cv == True:
            print("Model selected:  Random Forest")
            parameters = {'max_depth':[2,4,6, None], 'n_estimators':[50,100,200]}
            params = list(product(parameters['max_depth'], parameters['n_estimators']))
            list_res = []
            for p in params:
                model = RandomForestClassifier(max_depth=p[0], n_estimators=p[1])
                model.fit(X_train, y_train)
                predictions_valid = model.predict(X_valid)
                hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v = predictions(predictions_valid, y_valid)
                list_res.append([p[0], p[1], hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v])
            print()
            k_ = 0
            for k in list_res:
                if k[1] > k_:
                    k_ = k[1]
                    best_depth, best_estimators, hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v = k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]
            print("Best results and parameters: ", best_depth, best_estimators, hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v)

            X_train_ = torch.cat([X_train, X_valid], dim=0)
            y_train_ = torch.cat([y_train, y_valid], dim=0)

            model = RandomForestClassifier(max_depth=best_depth, n_estimators=best_estimators)
            model.fit(X_train_, y_train_)
            predictions_test = model.predict(X_test)
            predictions_train = model.predict(X_train_)
            writePredictions(predictions_train, y_train_, path_output_predictions + '{}_{}__TrainSet_v2'.format(ML_model,embedder))
            writePredictions(predictions_test, y_test, path_output_predictions + '{}_{}__TestSet_v2'.format(ML_model,embedder))
            hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v = predictions(predictions_train, y_train_)
            hits_50_t,hits_100_t, waf, precision, recall, accuracy = predictions(predictions_test, y_test)
            return hits_50_v, hits_100_v, waf_v, precision_v, recall_v, accuracy_v, hits_50_t,hits_100_t, waf, precision, recall, accuracy


def run_neural(dl_model, embeddings = None, data_path="./dataset", combined=False):
    if dl_model == "GCN":
        if combined==True: command_to_execute = ["python", "gnn_lp.py", "--K", "{}".format(embeddings), "--runs", "5", "--combine"]
        else: command_to_execute = ["python3", "gnn_lp.py", "--K", "{}".format(embeddings), "--runs", "5", "--dataset","{}".format(data_path)]
    elif dl_model == "GraphSAGE":
        if combined==True: command_to_execute = ["python", "gnn_lp.py", "--use_sage", "--K", "{}".format(embeddings), "--runs", "5", "--combine"]
        else: command_to_execute = ["python3", "gnn_lp.py", "--use_sage", "--K", "{}".format(embeddings), "--runs", "5", "--dataset","{}".format(data_path)]
    elif dl_model == "NGNN_GCN":
        command_to_execute = ["python","/ngnn/main.py","--device","0","--root","{}".format(data_path),"--ngnn_type","input","--K1","{}".format(embeddings),"--epochs","80","--dropout","0.2","--num_layers","3","--lr","0.001","--batch_size","49152","--runs","10"]
    elif dl_model == "NGNN_GraphSAGE":
        command_to_execute = ["python","ngnn/main.py","--device","0","--root","{}".format(data_path),"--ngnn_type","input","--K1","{}".format(embeddings),"--use_sage","--epochs", "80","--dropout","0.2","--num_layers","3","--lr","0.001","--batch_size","49152","--runs","10"]
    elif dl_model == "AGDN":
        command_to_execute = ["python","/Adaptive-Graph-Diffusion-Networks-master/ogbl_no_sampling/src/main.py","--embK","{}".format(embeddings),"--epochs","40","--model","agdn","--eval-steps","1","--log-steps","1","--K","2","--hop-norm","--transition-matrix","gat","--negative-sampler","global","--eval-metric","hits","--lr","0.01","--n-layers","2","--n-hidden","64","--n-heads","1","--batch-size","65536","--dropout","0.","--attn-drop","0.2","--input-drop","0.","--diffusion-drop","0.","--use-emb","--no-node-feat","--loss-func","CE","--bn"]
    elif dl_model == "NGNN_SEAL":
        command_to_execute = ["python","ngnn-seal/main.py","--data_path","{}".format(data_path), "--K","{}".format(embeddings),"--ngnn_type","input","--hidden_channels","48","--epochs","50","--lr","0.00015","--batch_size","128","--num_workers","48","--train_percent","5","--val_percent","8","--use_feature","--dynamic_train","--dynamic_val","--dynamic_test","--runs","2"]
    subprocess.run(command_to_execute)
    print("{} model was run!".format(dl_model))

import argparse

def main():
    parser = argparse.ArgumentParser(description='DKI pipeline')
    parser.add_argument('--data_path', type=str, action="store", default="./dataset_R")
    parser.add_argument('--dataset', type=str, action="store", default="ogbl-ppa")
    parser.add_argument('--type', type=str, action="store", default="DL")
    parser.add_argument('--model', type=str, action="store", default="GCN")
    parser.add_argument('--compute_embedding', action="store_true")
    parser.add_argument('--use_ppi', action="store_true")
    parser.add_argument('--randomize', action="store_true")
    parser.add_argument('--embedding', type=str, action="store", default="opa2vec")
    parser.add_argument('--onlymetrics', action="store_true")
    parser.add_argument('--combine', action="store_true")
    args = parser.parse_args()
    print(args)

    data, split_edge, x = make_dataset(path = args.data_path, name_=args.dataset)

    if args.type == "DL":
        if args.compute_embedding:
            make_embeddings(embedder = args.embedding, dataset=args.dataset)

        from pathlib import Path
        if args.randomize == True and Path(args.data_path + "/embedding_random.pt").exists() == False:
            print("randomizing features : ")
            x = torch.Tensor(np.random.rand(len(x),50))
            torch.save(x, args.data_path + "/embedding_random.pt")
            x_ = get_embeddings(path_ = args.data_path, dataset = args.dataset, embedder = args.embedding, x_ = x)
        elif args.randomize == True and Path(args.data_path + "/embedding_random.pt").exists() == True:
            x = torch.load(args.data_path + "/embedding_random.pt")
            x_ = get_embeddings(path_ = args.data_path, dataset = args.dataset, embedder = args.embedding, x_ = x)
        elif args.randomize == False and args.embedding != "": x_ = get_embeddings(path_ = args.data_path, dataset = args.dataset, embedder = args.embedding)
        
        torch.save(x_, "embedding_{}_{}.pt".format(args.embedding,args.model))
        if args.use_ppi == True and args.combined == True: run_neural(embeddings = "embedding_{}_{}.pt".format(args.embedding,args.model), dl_model = args.model, data_path=args.data_path, combined = True)
        else: run_neural(embeddings = "embedding_{}_{}.pt".format(args.embedding,args.model), dl_model = args.model, data_path=args.data_path)

    elif args.type == "ML":
        if args.onlymetrics:
            predictions_valid = pd.read_csv("{}_{}_ValidSet_v2".format(args.model,args.embedding), sep="\t")["Predicted_output"]
            y_valid = pd.read_csv("{}_{}_ValidSet_v2".format(args.model,args.embedding), sep="\t")["Expected_Output"]
            predictions_test = pd.read_csv("{}_{}_TestSet_v2".format(args.model,args.embedding), sep="\t")["Predicted_output"]
            y_test = pd.read_csv("{}_{}_TestSet_v2".format(args.model,args.embedding), sep="\t")["Expected_Output"]

            hits_50_v, hits_100_v, waf_v, prec_v, rec_v, acc_v = predictions(predictions_valid, y_valid)
            hits_50_t, hits_100_t, waf_t, prec_t, rec_t, acc_t = predictions(predictions_test, y_test)
            print()
            print("Results for {} with embeddings from {} :".format(args.model,args.embedding))
            print("Validation: HITS@50 - {} , HITS@100 - {}  , WAF - {}, PRECISION - {}, RECALL - {}, ACC - {}, ".format(hits_50_v, hits_100_v, waf_v, prec_v, rec_v, acc_v))
            print("Testing: HITS@50 - {} , HITS@100 - {}  , WAF - {}, PRECISION - {}, RECALL - {}, ACC - {}, ".format(hits_50_t, hits_100_t, waf_t, prec_t, rec_t, acc_t))
            exit()

        if args.compute_embedding:
            make_embeddings(embedder = args.embedding, dataset=args.dataset)
        from pathlib import Path
        if args.use_ppi == True:
            x_train, x_valid, x_test, y_train, y_valid, y_test  = get_embeddings(path_ = args.data_path, dataset = args.dataset, embedder = args.embedding, partitions=split_edge, x_=x)
            args.embedding += "_ppi"
        elif args.randomize == True and Path(args.data_path + "/embedding_random.pt").exists() == False:
            print("randomizing features : ")
            x = torch.Tensor(np.random.rand(len(x),50))
            torch.save(x, args.data_path + "/embedding_random.pt")
            x_train, x_valid, x_test, y_train, y_valid, y_test = get_embeddings(path_ = args.data_path, dataset = args.dataset, embedder = args.embedding, partitions=split_edge, x_=x)
        elif args.randomize == True and Path(args.data_path + "/embedding_random.pt").exists() == True:
            x = torch.load(args.data_path + "/embedding_random.pt")
            x_train, x_valid, x_test, y_train, y_valid, y_test = get_embeddings(path_ = args.data_path, dataset = args.dataset, embedder = args.embedding, partitions=split_edge, x_=x)


        else: x_train, x_valid, x_test, y_train, y_valid, y_test = get_embeddings(path_ = args.data_path, dataset = args.dataset, embedder = args.embedding, partitions=split_edge)
        hits_50_v, hits_100_v, waf_v, prec_v, rec_v, acc_v, hits_50_t, hits_100_t, waf_t, prec_t, rec_t, acc_t = run_ML(x_train, y_train, x_valid, y_valid, x_test, y_test, ML_model = args.model, path_output_predictions="./", embedder=args.embedding)
        print()
        print("Results for {} with embeddings from {} :".format(args.model,args.embedding))
        print("Validation: HITS@50 - {} , HITS@100 - {} ,  WAF - {}, PRECISION - {}, RECALL - {}, ACC - {}, ".format(hits_50_v, hits_100_v, waf_v, prec_v, rec_v, acc_v))
        print()
        print("Testing: HITS@50 - {} , HITS@100 - {}  , WAF - {}, PRECISION - {}, RECALL - {}, ACC - {}, ".format(hits_50_t, hits_100_t, waf_t, prec_t, rec_t, acc_t))

if __name__ == "__main__":
    main()


#################################
# python3 run_link_prediction.py --type DL --model GCN --compute_embedding --embedding rdf2vec
# python3 run_link_prediction.py --type DL --model GCN --compute_embedding --use_ppi --embedding wang2vec
# python3 run_link_prediction.py --type ML --model RF --embedding distmult
# python3 run_link_prediction.py --type DL --model NGNN_GCN --embedding --use_ppi
