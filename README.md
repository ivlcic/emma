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
```bash
./longdoc  # for long documents embeddings tests
./mulabel  # DEPRECATED module too much mess
./newsmon  # Information retrieval based multilabel classification 
```

So you need to
```bash
chmod +x newsmon
```

Each script invokes a module in `emma/[module-name]` directory based on a command.

For instance
`./newsmon fa test_rae -c newsmon -l sl --public`

This invokes `fa_test_rae` function from the `emma/newsmon/fa/__init__.py` script.

The data is usually loaded from `data/[module-name]` dir, results are put in `result/[module-name]` and,   
temporary space is in `tmp/[module-name]`

