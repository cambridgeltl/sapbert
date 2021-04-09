
MODEL_DIR="cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # can switch to your /path/to/model

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
