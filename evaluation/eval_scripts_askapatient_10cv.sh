
MODEL_DIR="cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # can switch to your /path/to/model
DICT_PATH=data/askapatient/AskAPatient.dict.txt

for VAR in 0 1 2 3 4 5 6 7 8 9
do 
	DATA_DIR=data/askapatient/AskAPatient.fold-$VAR.test.preprocessed_query.txt
	CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \
	--output_dir tmp/ \
	--use_cuda \
	--max_length 25 \
	--save_predictions \
	--custom_query_loader 
done
