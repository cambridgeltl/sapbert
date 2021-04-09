# SapBERT: Self-alignment pretraining for BERT


This repo holds code for reproducing the SapBERT model presented in our NAACL 2021 paper: *Self-Alignment Pretraining for Biomedical Entity Representations* [\[arxiv\]](https://arxiv.org/abs/2010.11784).


## Citation
```bibtex
@article{liu2020self,
	title={Self-alignment Pre-training for Biomedical Entity Representations},
	author={Liu, Fangyu and Shareghi, Ehsan and Meng, Zaiqiao and Basaldella, Marco and Collier, Nigel},
	journal={arXiv preprint arXiv:2010.11784},
	year={2020}
}
```

## Huggingface Models

### [\[SapBERT\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
Standard SapBERT as described in (Liu et al. 2020). Trained with UMLS 2020AA (English only), using `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` as the base model. Use `[CLS]` as the representation of the input.

### [\[SapBERT-XLMR\]](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR)
SapBERT trained with UMLS 2020AB, using `xlm-roberta-base` as the base model. Use `[CLS]` as the representation of the input.

### [\[SapBERT-mean-token\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token)
Same as the standard SapBERT but trained with mean-pooling instead of `[CLS]` representations.


## Environment
The code is tested with python 3.8, torch 1.7.0 and huggingface transformers 4.4.2. Please view `requirements.txt` for more details.

## Train SapBERT

Prepare training data as insrtructed in `data/generate_pretraining_data.ipynb`.

Run:
```console
cd umls_pretraining
./pretrain.sh 0,1 
```
where `0,1` specifies the GPU devices. 

## Evaluate SapBERT
Please view `evaluation/README.md` for details.

## Acknowledgement
Parts of the code are modified from [BioSyn](https://github.com/dmis-lab/BioSyn). We appreciate the authors for making BioSyn open-sourced.

## License
SapBERT is MIT licensed. See the [LICENSE](LICENSE) file for details.
