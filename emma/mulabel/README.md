## Environment preparation

Activate the Python virtual environment
```bash
cd ~/projects/emma
source .venv/bin/activate
```
If missing you need to create it and install all dependencies via pip:
```bash
cd ~/projects/emma
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Corpus preparation

First, we have to extract the corpus from the raw json format:

```bash
./mulabel prep corpus_extract -c mulabel -l sl --public --postfix 2023_01,2023_02
```
The script expects source files in `data/mulabel/raw/src` and mapping files in `data/mulabel/raw/map` directory. 

Extract operation will produce the following CSV files in `tmp/mulabel` directory: 
```bash
lrp_mulabel_sl_p1_s0_2023_01.csv
lrp_mulabel_sl_p1_s0_2023_02.csv
mulabel_sl_p1_s0_2023_01.csv
mulabel_sl_p1_s0_2023_01_map_labels.csv
mulabel_sl_p1_s0_2023_02.csv
mulabel_sl_p1_s0_2023_02_map_labels.csv
```
LRP stands for label relevant passages, and mapping files are for the additional label data.  
The LRP files are larger because we produce passages of *1, 3, 5, 7, 9* sentence context sizes 
with respect to label occurrence in text.

Next, we need to merge the corpus files also removing the single occurrence label samples:
```bash
./mulabel prep corpus_merge -l sl --public --postfix 2023_01,2023_02
```
This produces the following files in `data/mulabel/raw` directory: 
```bash
lrp_mulabel_sl_p1_s0.csv
mulabel_sl_p1_s0.csv
mulabel_sl_p1_s0_map_labels.csv
```

The corpus contains duplicates or *almost* duplicated samples, so we need to find all duplicates.
We do dense retrieval search in the *Elasicsearch* index so we have to spin-up *Elasticsearch* node, 
create index templates and pump in the data. We achieve this via docker compose:

```bash
cd docker
docker compose rm -v  # cleanup previous containers and data
docker compose build  # this is done automatically if we use 'docker compose up', 
                      # but not if we change templates between the runs
docker compose up  # use -d to start it in the background
```

Now we have running *Elasicsearch* node accessible at `localhost:9266` and we pump in the data:
```bash
export CUDA_VISIBLE_DEVICES=0  # in our case we have to pin to specific Nvidia card cause we're "under-resourced" :)
./mulabel es drop -c mulabel -l sl --public  # delete previous index
./mulabel es init -c mulabel -l sl --public  # initialize new index with a template
./mulabel es pump -c mulabel -l sl --public  # pump in the data
```
If you want to pum in LRP data you just use the `lrp_` prefix to a collection name:
```bash
./mulabel es drop -c lrp_mulabel -l sl --public  # delete previous LRP index
./mulabel es init -c lrp_mulabel -l sl --public  # initialize new LRP index with a template
./mulabel es pump -c lrp_mulabel -l sl --public  # pump in the LRP data
```

During the data pump *BGE M3* model embeddings (dense vector text representation) is computed.  
Finally, we compute duplicate samples by iterating whole collection and searching for similar samples with a *0.99* threshold.
```bash
./mulabel es dedup -c mulabel -l sl --public
```
Dedup operation produces list of duplicates in a `data/mulabel/raw` directory:
```
mulabel_sl_p1_s0_duplicates.csv
```

Now, if we repeat steps, the scripts will mark the corpus's duplicates and do train/val/test split at 80/10/10:
```bash
./mulabel prep corpus_extract -c mulabel -l sl --public --postfix 2023_01,2023_02
./mulabel prep corpus_merge -c mulabel -l sl --public --postfix 2023_01,2023_02
./mulabel prep corpus_split -c mulabel -l sl --public
./mulabel prep corpus_split -c eurlex --label_col ml_label  # for splitting an alternative corpora

# pump in the data
export CUDA_VISIBLE_DEVICES=0  # in our case we have to pin to specific Nvidia card cause we're "under-resourced" :)
./mulabel es drop -c mulabel -l sl --public  --suffix train # delete the previous index
./mulabel es init -c mulabel -l sl --public  --suffix train # initialize a new index
./mulabel es pump -c mulabel -l sl --public  --suffix train # pump in the data

./mulabel es init -c eurlex --suffix train  # pump in the data for the alternative collection
./mulabel es pump -c eurlex --suffix train  
```

## Tests
When the corpus is prepared we can run zero shot classification tests. 
The script expects train/dev/test split files in `data/mulabel/split` directory and *Elasticsearch* indexed train collection. 

```
./mulabel es test_bge_m3 -c mulabel -l sl --public
```
