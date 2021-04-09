import csv
import json
import numpy as np
import torch
import pdb
from tqdm import tqdm

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)

def predict_topk(model_wrapper, eval_dictionary, eval_queries, topk, cometa=False, agg_mode="cls"):

    encoder = model_wrapper.get_dense_encoder()
    tokenizer = model_wrapper.get_dense_tokenizer()
    
    # embed dictionary
    dict_names = [row[0] for row in eval_dictionary]
    
    print ("[start embedding dictionary...]")
    dict_dense_embeds = model_wrapper.embed_dense(names=dict_names, show_progress=True, agg_mode=agg_mode)
    
    mean_centering = False
    if mean_centering:
        tgt_space_mean_vec = dict_dense_embeds.mean(0)
        dict_dense_embeds -= tgt_space_mean_vec

    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        golden_cui = eval_query[1].replace("+","|")
        
        dict_mentions = []

        
        for mention in mentions:
            mention_dense_embeds = model_wrapper.embed_dense(names=[mention], agg_mode=agg_mode)
            
            if mean_centering:
                mention_dense_embeds -= tgt_space_mean_vec

            # get score matrix
            dense_score_matrix = model_wrapper.get_score_matrix(
                    query_embeds=mention_dense_embeds, 
                    dict_embeds=dict_dense_embeds,
            )
            score_matrix = dense_score_matrix

            candidate_idxs = model_wrapper.retrieve_candidate_cuda(
                    score_matrix = score_matrix, 
                    topk = topk,
                    batch_size=16,
                    show_progress=False
            )

            #print(candidate_idxs.shape)
            np_candidates = [eval_dictionary[ind] for ind in candidate_idxs[0].tolist()]#.squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                        'name':np_candidate[0],
                        'labelcui':np_candidate[1],
                        'label':check_label(np_candidate[1],golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result

def predict_topk_fast(model_wrapper, eval_dictionary, eval_queries, topk, agg_mode="cls"):
    """
    for MedMentions only
    """

    encoder = model_wrapper.get_dense_encoder()
    tokenizer = model_wrapper.get_dense_tokenizer()
    
    # embed dictionary
    dict_names = [row[0] for row in eval_dictionary]
    print ("[start embedding dictionary...]")
    dict_dense_embeds = model_wrapper.embed_dense(names=dict_names, show_progress=True, batch_size=4096, agg_mode=agg_mode)
    print ("dict_dense_embeds.shape:", dict_dense_embeds.shape)
    
    bs = 32
    candidate_idxs = None
    print ("[computing rankings...]")


    for i in tqdm(np.arange(0,len(eval_queries),bs), total=len(eval_queries)//bs+1):
        mentions = list(eval_queries[i:i+bs][:,0])
        
        mention_dense_embeds = model_wrapper.embed_dense(names=mentions, agg_mode=agg_mode)

        # get score matrix
        dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=mention_dense_embeds, 
            dict_embeds=dict_dense_embeds
        )
        score_matrix = dense_score_matrix
        candidate_idxs_batch = model_wrapper.retrieve_candidate_cuda(
            score_matrix = score_matrix, 
            topk = topk,
            batch_size=bs,
            show_progress=False
        )
        if candidate_idxs is None:
            candidate_idxs = candidate_idxs_batch
        else:
            candidate_idxs = np.vstack([candidate_idxs, candidate_idxs_batch])
            
    queries = []
    golden_cuis = list(eval_queries[:,1])
    mentions = list(eval_queries[:,0])
    print ("[writing json...]")
    for i in tqdm(range(len(eval_queries))):
        #print(candidate_idxs.shape)
        np_candidates = [eval_dictionary[ind] for ind in candidate_idxs[i].tolist()]#.squeeze()
        dict_candidates = []
        dict_mentions = []
        for np_candidate in np_candidates:
            dict_candidates.append({
                'name':np_candidate[0],
                'labelcui':np_candidate[1],
                'label':check_label(np_candidate[1],golden_cuis[i])
            })
        dict_mentions.append({
            'mention':mentions[i],
            'golden_cui':golden_cuis[i], # golden_cui can be composite cui
            'candidates':dict_candidates
        })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result

def evaluate(model_wrapper, eval_dictionary, eval_queries, topk, cometa=False, medmentions=False, agg_mode="cls"):
    
    if medmentions or cometa:
        result = predict_topk_fast(model_wrapper, eval_dictionary, eval_queries, topk, agg_mode=agg_mode)
    else:
        result = predict_topk(model_wrapper, eval_dictionary, eval_queries, topk, agg_mode=agg_mode)
    result = evaluate_topk_acc(result)
    
    return result



