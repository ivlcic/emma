## Environment preparation
DEPRECATED

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

If you have problems with flash-attn plugin install wheel first: `pip install wheel`

In the root of a directory you have an entry-point scripts:
```
./longdoc  # for long documents embeddings tests
./mulabel  # DEPRECATED module
./newsmon  # Information retrieval based multilabel classification 
```

Each script invokes a module in `emma/[script-name]` directory based on a command. 
