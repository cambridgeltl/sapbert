# SapBERT: Self-alignment pretraining for BERT


Thi§s repo holds code for the SapBERT model presented in our NAACL 2021 paper: [*Self-Alignment Pretraining for Biomedical Entity Representations*](https://www.aclweb.org/anthology/2021.naacl-main.334.pdf); and our ACL 2021 paper: [*Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking*](http://fangyuliu.me/media/pdfs/xlbel_acl2021_preprint.pdf).

![front-page-graph](/misc/sapbert_front_graphs_v6.png?raw=true)

## Huggingface Models

### English Models: [\[SapBERT\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) and [\[SapBERT-mean-token\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token)
Standard SapBERT as described in [\[Liu et al., NAACL 2021\]](https://www.aclweb.org/anthology/2021.naacl-main.334.pdf). Trained with UMLS 2020AA (English only), using `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` as the base model. For [\[SapBERT\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext), use `[CLS]` (before pooler) as the representation of the input; for [\[SapBERT-mean-token\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token), use mean-pooling across all tokens.

### Cross-Lingual Models: [\[SapBERT-XLMR\]](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR) and [\[SapBERT-XLMR-large\]](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large)
Cross-lingual SapBERT as described in [\[Liu et al., ACL 2021\]](http://fangyuliu.me/media/pdfs/xlbel_acl2021_preprint.pdf). Trained with UMLS 2020AB (all languages), using `xlm-roberta-base`/`xlm-roberta-large` as the base model. Use `[CLS]` (before pooler) as the representation of the input.

## Environment
The code is tested with python 3.8, torch 1.7.0 and huggingface transformers 4.4.2. Please view `requirements.txt` for more details.

## Train SapBERT
Extract training data from UMLS as insrtructed in `training_data/generate_pretraining_data.ipynb` (we cannot directly release the training file due to licensing issues).

Run:
```console
cd umls_pretraining
./pretrain.sh 0,1 
```
where `0,1` specifies the GPU devices. 

For cross-lingual Sap-tuning with general domain parallel data (muse, wiki titles, or both), the data can be found in `training_data/general_domain_parallel_data/`. Scripts are in `xl_bel/`. 

## Evaluate SapBERT
For monolingual evaluation (as in [\[Liu et al., NAACL 2021\]](https://www.aclweb.org/anthology/2021.naacl-main.334.pdf)), please view `evaluation/README.md` for details.
For cross-lingual evaluation (XL-BEL; as in [\[Liu et al., ACL 2021\]](http://fangyuliu.me/media/pdfs/xlbel_acl2021_preprint.pdf)), please view `xl_bel/README.md` for details.

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
	month = aug,
	year={2021}
}
```

## Acknowledgement
Parts of the code are modified from [BioSyn](https://github.com/dmis-lab/BioSyn). We appreciate the authors for making BioSyn open-sourced.

## License
SapBERT is MIT licensed. See the [LICENSE](LICENSE) file for details.
