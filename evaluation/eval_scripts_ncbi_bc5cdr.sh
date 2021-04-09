MODEL_DIR="cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # can switch to your /path/to/model

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
