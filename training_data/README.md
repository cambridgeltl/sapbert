# SAP data

## Pretraining UMLS Training Data
Please download the [full release](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) of UMLS and follow `generate_pretraining_data.ipynb`.

## Finetuning Data
Similar to `generate_pretraining_data.ipynb`, you need to generate data in the format of 
```
label_id || entity_name_1 || entity_name_2
...
```
where `entity_name_1` and `entity_name_2` are synonym pairs sampled from a given labelled dataset. If one label is associated with multiple entity names in the dataset, you could traverse all the pairwise combinations.

## General Domain Parallel Data 
Parallel wikipedia title and word translation pairs used in [Learning Domain-Specialised Representations for Cross-Lingual
Biomedical Entity Linking](https://arxiv.org/pdf/2105.14398.pdf). Please download [here](https://www.dropbox.com/sh/gqwhrr4oj8ppu9x/AAC5RsNgC55R1XLSAbkw-kG6a?dl=0).

## Evalutation Data
See `evaluation/README.md`.
