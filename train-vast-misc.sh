./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name bertmc --corpus eurlex
./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name xlmrb --corpus eurlex

./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name tobertmc --corpus eurlex
./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name toxlmrb --corpus eurlex

./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name bertmcplusrandom --corpus eurlex
./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name xlmrbplusrandom --corpus eurlex

apt install -y nvtop tree
git clone https://github.com/ivlcic/emma.git
cd emma
mkdir -p data/longdoc
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


./longdoc hf_train --batch 8 --epochs 20 --lr 5e-5 --model_name xlmrb --corpus eurlex --num_workers 4 --seed 2611

./longdoc hf_train --batch 8 --epochs 20 --lr 5e-5 --model_name bert --corpus eurlex --num_workers 4 --seed 2611

apt install -y tree nvtop vim wget unzip
git clone https://github.com/amazon-science/efficient-longdoc-classification.git
cd efficient-longdoc-classification/
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r src/requirements.txt

mkdir -p data/EURLEX57K
wget -O data/EURLEX57K/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip data/EURLEX57K/datasets.zip -d data/EURLEX57K
rm data/EURLEX57K/datasets.zip
rm -rf data/EURLEX57K/__MACOSX
mv data/EURLEX57K/dataset/* data/EURLEX57K
rm -rf data/EURLEX57K/dataset
wget -O data/EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json

python src/train.py --model_name bert --data eurlex --pairs --batch_size 8 --epochs 20 --lr 5e-05 --num_workers 4 --seed 2611