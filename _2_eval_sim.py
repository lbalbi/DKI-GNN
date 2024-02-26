import torch,pandas,numpy,itertools
from torchmetrics import *
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity


def rank_results(test_file,labels_file,output=None):
    if type(test_file) == dict :
        lista = []
        for (i,j) in test_file.keys():
            lista.append([i,j,test_file[(i,j)]])
        results = pandas.DataFrame(lista)
    else:
        results = pandas.read_csv("{}".format(test_file), header=None, index_col=False, sep="\t")
    results.set_axis(['Ent1', 'Ent2', 'sim'], axis=1, inplace=True)
    label = []
    if type(labels_file) == pandas.DataFrame:
        label = labels_file.values.tolist(); del labels_file
    else:
        labels = torch.load("{}".format(labels_file))
        for k in range(len(results['Ent1'])):
            if (results['Ent1'][k],results['Ent2'][k]) in labels["edge"]:
                label.append(1)
            elif (results['Ent1'][k],results['Ent2'][k]) in labels["edge_neg"]:
                label.append(0)
    results['label'] = pandas.DataFrame(label)
    results['rank'] = results['sim'].rank(method = "min")
    print(results)
    if output != None:
        results.to_csv(output,index=False)
    return _overall_metrics(results)


def _overall_metrics(results):
    accuracy = Accuracy("binary")
    accuracy = accuracy(torch.tensor(results['sim'].values), torch.tensor(results['label'].values))
    mrr = RetrievalMRR()
    mrr = mrr(torch.tensor(results['sim'].values), torch.tensor(results['label'].values), torch.tensor(numpy.asarray([i for i in range(len(results["Ent1"]))])))
    rrr = functional.retrieval_reciprocal_rank(torch.tensor(results['sim'].values), torch.tensor(results['label'].values))
    return results, [accuracy,mrr,rrr]



def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = numpy.trapz(auc_y, auc_x) / n_prots
    return auc

def compute_similarity(embeddings_file, edges):
    if type(embeddings_file) == numpy.ndarray:
        embeddings = embeddings_file.tolist() 
        del embeddings_file
    elif type(embeddings_file) == torch.Tensor:
        embeddings = embeddings_file.tolist()
    else:
        embeddings = torch.load(embeddings_file, map_location="cpu").tolist()
    dict_embeddings = {}
    for i in range(len(embeddings)):
        dict_embeddings[i] = embeddings[i]
    dict_res = {}
    
    if type(edges) == pandas.DataFrame:
        for index, row in edges.iterrows():
            #print(row[0],row[1])
            emb1 = numpy.array(dict_embeddings[row[0]])
            emb1 = emb1.reshape(1, len(emb1))
            emb2 = numpy.array(dict_embeddings[row[1]])
            emb2 = emb2.reshape(1, len(emb2))
            sim = cosine_similarity(emb1, emb2)[0][0]
            dict_res[(row[0],row[1])] = sim
    else:
        for e in list(edges):
            if e[0]% 100 == 0:
                print(e[0], e[1])
            e1,e2 = e[0],e[1]
            emb1 = numpy.array(dict_embeddings[e1])
            emb1 = emb1.reshape(1, len(emb1))
            emb2 = numpy.array(dict_embeddings[e2])
            emb2 = emb2.reshape(1, len(emb2))
            sim = cosine_similarity(emb1, emb2)[0][0]
            dict_res[(e1,e2)] = sim
    del edges,embeddings_file
    return dict_res


def compute_metrics_el_embeddings_v2(ents, results, dict_SS):
    pairs_ents, list_labels = results.loc[:,["Ent1","Ent2"]].values, results["label"].values

    print(pairs_ents)
    exit()

    top1, top10,top100,mean_rank = 0,0,0,0
    ranks = {}
    dic_index = {}
    for i in range(len(ents[0])):
        dic_index[ents[0][i]] = i

    labels = numpy.zeros((len(ents[0]), len(ents[0])), dtype=numpy.int32)
    sim = numpy.zeros((len(ents[0]), len(ents[0])), dtype=numpy.float32)
    for i1 in range(len(ents[0])):
        ent1 = ents[0][i1]
        for i2 in range(len(ents[0])):
            ent2 = ents[0][i2]
            if ent1 == ent2:
                sim[i1,i2] = 1
            else:
                sim[i1, i2] = dict_SS[(ent1,ent2)]

    for i in range(len(pairs_ents)):
        e1,e2 = pairs_ents[i]
        label = list_labels[i]
        pairs_ents.append((e1,e2))
        list_labels.append(label)

    for i in range(len(pairs_ents)):
        e1,e2 = pairs_ents[i]
        i1 = dic_index[e1]
        i2 = dic_index[e2]
        if list_labels[i]==1:
            labels[i1,i2] =1

            index = rankdata(-sim[e1, :], method='average')
            rank = index[e2]
            if rank <=1:
                top1 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

    n = len(pairs_ents)
    top1 /= n
    top10 /= n
    top100 /= n
    mean_rank /= n
    rank_auc = compute_rank_roc(ranks, len(ents[0]))
    roc_auc = compute_roc_auc(dict_SS, ents[0], pairs_ents, list_labels)
    return [top1, top10, top100, mean_rank, rank_auc, roc_auc] 


def ss_all_pairs_possible(entities,embeddings_file):
    x = itertools.combinations(entities[0], 2)
    del entities
    dict_res = compute_similarity(embeddings_file, x)
    return dict_res
    

    
    
        
