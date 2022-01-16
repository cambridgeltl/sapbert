# SapBERT: Self-alignment pretraining for BERT

**\[news | 22 Aug 2021\]** SapBERT is integrated into NVIDIA's deep learning toolkit NeMo as its [entity linking module](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/entity_linking.html) (thank you NVIDIA!). You can play with it in this [google colab](https://colab.research.google.com/github/NVIDIA/NeMo/blob/v1.0.2/tutorials/nlp/Entity_Linking_Medical.ipynb).

--------

This repo holds code, data, and pretrained weights for **(1)** the **SapBERT** model presented in our NAACL 2021 paper: [*Self-Alignment Pretraining for Biomedical Entity Representations*](https://www.aclweb.org/anthology/2021.naacl-main.334.pdf); **(2)** the **cross-lingual SapBERT** and a cross-lingual biomedical entity linking benchmark (**XL-BEL**) proposed in our ACL 2021 paper: [*Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking*](https://arxiv.org/pdf/2105.14398.pdf).

![front-page-graph](/misc/sapbert_front_graphs_v6.png?raw=true)




## Huggingface Models

### English Models: [\[SapBERT\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) and [\[SapBERT-mean-token\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token)
Standard SapBERT as described in [\[Liu et al., NAACL 2021\]](https://www.aclweb.org/anthology/2021.naacl-main.334.pdf). Trained with UMLS 2020AA (English only), using `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` as the base model. For [\[SapBERT\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext), use `[CLS]` (before pooler) as the representation of the input; for [\[SapBERT-mean-token\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token), use mean-pooling across all tokens.

### Cross-Lingual Models: [\[SapBERT-XLMR\]](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR) and [\[SapBERT-XLMR-large\]](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large)
Cross-lingual SapBERT as described in [\[Liu et al., ACL 2021\]](https://arxiv.org/pdf/2105.14398.pdf). Trained with UMLS 2020AB (all languages), using `xlm-roberta-base`/`xlm-roberta-large` as the base model. Use `[CLS]` (before pooler) as the representation of the input.

## Environment
The code is tested with python 3.8, torch 1.7.0 and huggingface transformers 4.4.2. Please view `requirements.txt` for more details.

## Train SapBERT
Extract training data from UMLS as insrtructed in `training_data/generate_pretraining_data.ipynb` (we cannot directly release the training file due to licensing issues).

Run:
```bash
>> cd train/
>> ./pretrain.sh 0,1 
```
where `0,1` specifies the GPU devices. 

For finetuning on your customised dataset, generate data in the format of 
```
concept_id || entity_name_1 || entity_name_2
...
```
where `entity_name_1` and `entity_name_2` are synonym pairs (belonging to the same concept `concept_id`) sampled from a given labelled dataset. If one concept is associated with multiple entity names in the dataset, you could traverse all the pairwise combinations.

For cross-lingual SAP-tuning with general domain parallel data (muse, wiki titles, or both), the data can be found in `training_data/general_domain_parallel_data/`. An example script: `train/xling_train.sh`. 

## Evaluate SapBERT
For evaluation (both monlingual and cross-lingual), please view `evaluation/README.md` for details. `evaluation/xl_bel/` contains the XL-BEL benchmark proposed in [\[Liu et al., ACL 2021\]](https://arxiv.org/pdf/2105.14398.pdf).

## Citations
SapBERT: 
```bibtex
@inproceedings{liu2021self,
	title={Self-Alignment Pretraining for Biomedical Entity Representations},
	author={Liu, Fangyu and Shareghi, Ehsan and Meng, Zaiqiao and Basaldella, Marco and Collier, Nigel},
	booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
	pages={4228--4238},
	month = jun,
	year={2021}
}
```
Cross-lingual SapBERT and XL-BEL:
```bibtex
@inproceedings{liu2021learning,
	title={Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking},
	author={Liu, Fangyu and Vuli{\'c}, Ivan and Korhonen, Anna and Collier, Nigel},
	booktitle={Proceedings of ACL-IJCNLP 2021},
	pages = {565--574},
	month = aug,
	year={2021}
}
```

## Acknowledgement
Parts of the code are modified from [BioSyn](https://github.com/dmis-lab/BioSyn). We appreciate the authors for making BioSyn open-sourced.

## License
SapBERT is MIT licensed. See the [LICENSE](LICENSE) file for details.
