# -- DKI Domain Knowledge Injected GNN --
--------------------

The DKI approach has been extensively tested on two popular prediction tasks:
 - link prediction over "ogbl-ppa" (https://ogb.stanford.edu/) and "HuRI" (https://tdcommons.ai/multi_pred_tasks/ppi/) benchmark datasets;
 - node classification over "ogbn-proteins" dataset (https://ogb.stanford.edu/);




--------------------

## INSTRUCTIONS:

## 1st Step
### For model evaluation over the "ogbn-proteins" or "ogbl-ppa" tasks' datasets, the GO and GO annotations sets need to be downloaded and put in the main folder "DKI-GNN" prior to running any script. 
They can be downloaded at http://release.geneontology.org/2020-06-01/ontology/index.html and http://release.geneontology.org/2020-06-01/annotations/goa , respectively. 

### For first time performing evaluation over HURI dataset:
First run "huri_to_ogb_datatype.py" to load data and build pyg-readable link property prediction dataset similar to OGB's PPI dataset formats. Loading of OGB datasets "ogbl-ppa" and "ogbn-proteins" does not require this initial step.




## 2nd Step
### Run the python script starting with "run_" followed by specific task name. See parameters for script.
E.g. default embedding generation for rdf2vec KGE method and model evaluation for GCN-based link prediction over the HURI dataset can be done with 
command "python3 run_link_prediction.py --type DL --model GCN --compute_embedding --embedding rdf2vec"

For model testing with different parameters than the default, run the desired neural model's main script directly.


Ablation studies were performed with an MLP. For reproduction purposes these were included with the main code: For Node Classification, 
see the "mlp_ablation.py" script inside the corresponding folder. For Link Prediction, it can be run from the run_link_prediction.py with parameters "--type ML" and "--model MLP".

--------------------

## INSTALLATION REQUIREMENTS:
- OGB>=1.1.2
- Numpy>=1.16.0
- pandas>=0.24.0
- PyTorch>=1.8+cu101
- torch-geometric>=2.0.2+cu101
- tqdm==4.59.0
