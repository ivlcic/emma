This code is in large degree based on  [efficient-longdoc-classification](https://github.com/amazon-science/efficient-longdoc-classification)

```
@inproceedings{park-etal-2022-efficient,
    title = "Efficient Classification of Long Documents Using Transformers",
    author = "Park, Hyunji  and
      Vyas, Yogarshi  and
      Shah, Kashif",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.79",
    doi = "10.18653/v1/2022.acl-short.79",
    pages = "702--709",
}
```

I have refactored teh code to support other datasets and pre-trained models more easily, mainly to support multilingual models and research.

To get started just run
```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Prep the data:
```
./longdoc prep hyperpartisan eurlex 20news booksummaries
```

Run the training:
```
./longdoc train --batch 8 --epochs 20 --lr 3e-05 --model_name bert --corpus hyperpartisan
./longdoc train --batch 8 --epochs 20 --lr 3e-05 --model_name bertplusrandom --corpus hyperpartisan
./longdoc train --batch 8 --epochs 20 --lr 3e-05 --model_name bertplustextrank --corpus hyperpartisan
./longdoc train --batch 8 --epochs 20 --lr 3e-05 --model_name tobert --corpus hyperpartisan
```
