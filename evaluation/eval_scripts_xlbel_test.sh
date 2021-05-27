#MODEL_DIR="bert-base-multilingual-uncased" # can switch to your /path/to/model
#MODEL_DIR="xlm-roberta-base"
#MODEL_DIR="xlm-roberta-large"
#MODEL_DIR="cambridgeltl/SapBERT-from-PubMedBERT-fulltext" 
# note that this model is trained with UMLS 2020AB, performance will be slightly better than the model trained with UMLS 2020AA (as reported in the paper)
MODEL_DIR="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" 
#MODEL_DIR="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large"

DATA_DIR="./xl_bel/xlbel_v0.0/de_1k_test_query.txt"
DICT_PATH="./xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" # you need to download this from dropbox, see readme

CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \
	--output_dir tmp/ \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

