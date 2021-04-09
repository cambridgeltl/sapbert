MODEL_DIR="cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # can switch to your /path/to/model
#MODEL_DIR="GanjinZero/UMLSBert_ENG"
#MODEL_DIR="/home/newpapa/data/upload_model/UMLSBert_ENG"
#MODEL_DIR="/home/newpapa/repos/entity-linking-21/src/active_pretraining/tmp/al_sapbert/pubmedbert+sapbert_rand_10k_no_pos_no_miner/checkpoint_2"
MODEL_DIR="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" #
#MODEL_DIR="dmis-lab/biobert-v1.1"
MODEL_DIR="../umls_pretraining/tmp/mbert_or_xmlr+sapbert/pubmedbert+sapbert_2020aa_all_syn_11792953_mean_token_random_seed_33"

# ncbi-disease
DICT_PATH=data/ncbi-disease/test_dictionary.txt
DATA_DIR=data/ncbi-disease/processed_test

CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \
	--output_dir tmp/ \
	--use_cuda \
	--max_length 25 \
	--save_predictions \


# bc5cdr-disease
DICT_PATH=data/bc5cdr-disease/test_dictionary.txt
DATA_DIR=data/bc5cdr-disease/processed_test

CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \
	--output_dir tmp/ \
	--use_cuda \
	--max_length 25 \
	--save_predictions \

# bc5cdr-chemical
DICT_PATH=data/bc5cdr-chemical/test_dictionary.txt
DATA_DIR=data/bc5cdr-chemical/processed_test

CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \
	--output_dir tmp/ \
	--use_cuda \
	--max_length 25 \
	--save_predictions \
