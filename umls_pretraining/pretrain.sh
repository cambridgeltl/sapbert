CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
	--train_dir /home/newpapa/Downloads/umls_full_vocab_eng_8740644_uncased_tradename_added_no_dup_9712959_pairwise_pair_th50_11792953.txt \
	--output_dir tmp/mbert_or_xmlr+sapbert/pubmedbert+sapbert_2020aa_all_syn_11792953_mean_token_random_seed_33 \
	--use_cuda \
	--epoch 1 \
	--train_batch_size 256 \
	--learning_rate 2e-5 \
	--max_length 25 \
	--checkpoint_step 999999 \
	--parallel \
	--amp \
	--pairwise \
	--random_seed 33 \
	--loss ms_loss \
	--use_miner \
 --type_of_triplets "all" \
 --miner_margin 0.2 \
 --agg_mode "mean_pool"
 
#	--model_dir "emilyalsentzer/Bio_ClinicalBERT" \
#	--save_checkpoint_all \
#	--model_dir tmp/mbert_or_xmlr+sapbert/new_implmentation_self_retreive_full_umls_11792953_maxlen25_start_from_xlmr_random_seed_33 \
#	--model_dir "bert-base-multilingual-uncased" \
#	--model_dir "bert-base-uncased" \
#	--model_dir "allenai/scibert_scivocab_uncased" \
#	--model_dir "dmis-lab/biobert-v1.1" \
