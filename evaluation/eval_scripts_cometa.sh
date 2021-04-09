
MODEL_DIR="cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # can switch to your /path/to/model
MODEL_DIR="../umls_pretraining/tmp/mbert_or_xmlr+sapbert/pubmedbert+sapbert_2020aa_all_syn_11792953_mean_token_random_seed_33"

DICT_PATH=data/cometa/COMETA_id_sf_dictionary.txt

# stratified general
DATA_DIR=data/cometa/splits/stratified_general/test.csv
CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \
	--output_dir tmp/ \
	--use_cuda \
	--max_length 25 \
	--save_predictions \
	--cometa \
	--agg_mode "mean_pool"

# zeroshot general
DATA_DIR=data/cometa/splits/zeroshot_general/test.csv
CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \
	--output_dir tmp/ \
	--use_cuda \
	--max_length 25 \
	--save_predictions \
	--cometa \
	--agg_mode "mean_pool"
