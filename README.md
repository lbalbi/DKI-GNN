# -- DKI Domain Knowledge Injected GNN --

To perform testing of the DKI approach over one of the tasks avaliable (link prediction with the "ogbl-ppa" and "HuRI" benchmark datasets and node classification with "ogbn-proteins" dataset):


INSTRUCTIONS:
### 0 - For model evaluation over the "ogbn-proteins" or "ogbl-ppa" tasks' datasets, the GO and GO annotations sets need to be downloaded and put in the main folder "DKI-GNN" prior to running any script. 
They can be downloaded at http://release.geneontology.org/2020-06-01/ontology/index.html and http://release.geneontology.org/2020-06-01/annotations/goa , respectively. 

### 1 - For the first time performing an evaluation over HURI dataset first run "huri_to_ogb_datatype.py" to load data and build pyg-readable link property prediction dataset similar to OGB's PPI dataset formats. Loading of OGB datasets "ogbl-ppa" and "ogbn-proteins" does not require this initial step.


### 2 - Run the python script starting with "run_" followed by specific task name. See parameters for script.
E.g. default embedding generation for rdf2vec KGE method and model evaluation for GCN-based link prediction over the HURI dataset can be done with 
command "python3 run_link_prediction.py --type DL --model GCN --compute_embedding --embedding rdf2vec"

For model testing with different parameters than the default, run the desired neural model's main script directly.


Ablation studies were performed with an MLP. For reproduction purposes these were included with the main code: For Node Classification, 
see the "mlp_ablation.py" script inside the corresponding folder. For Link Prediction, it can be run from the run_link_prediction.py with parameters "--type ML" and "--model MLP".

## INSTALLATION REQUIREMENTS:
- OGB>=1.1.2
- Numpy>=1.16.0
- pandas>=0.24.0
- PyTorch>=1.8+cu101
- torch-geometric>=2.0.2+cu101
- tqdm==4.59.0
