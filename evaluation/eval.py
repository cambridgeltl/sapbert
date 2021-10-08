import argparse
import logging
import os
import json
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("../") 

from utils import (
    evaluate,
)

from src.data_loader import (
    DictionaryDataset,
    QueryDataset,
    QueryDataset_custom,
    QueryDataset_COMETA,
)
from src.model_wrapper import (
    Model_Wrapper
)
LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert evaluation')

    # Required
    parser.add_argument('--model_dir', required=True, help='Directory for model')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')

    # Run settings
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    parser.add_argument('--save_predictions', action="store_true", help="whether to save predictions")

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)
    
    # options for COMETA
    parser.add_argument('--cometa', action="store_true", \
            help="whether to load full sentence from COMETA (or just use the mention)")
    parser.add_argument('--medmentions', action="store_true")
    parser.add_argument('--custom_query_loader', action="store_true")
    #parser.add_argument('--cased', action="store_true")

    parser.add_argument('--agg_mode', type=str, default="cls", help="{cls|mean_pool|nospec}")

    args = parser.parse_args()
    return args
    
def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def load_dictionary(dictionary_path): 
    dictionary = DictionaryDataset(
        dictionary_path = dictionary_path
    )
    return dictionary.data

def load_queries(data_dir, filter_composite, filter_duplicate):
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    return dataset.data

def load_queries_COMETA(data_dir, load_full_sentence, filter_duplicate):
    dataset = QueryDataset_COMETA(
        data_dir=data_dir,
        load_full_sentence=load_full_sentence,
        filter_duplicate=filter_duplicate
    )
    return dataset.data
                
def main(args):
    init_logging()
    print(args)

    # load dictionary and data
    eval_dictionary = load_dictionary(dictionary_path=args.dictionary_path)
    print ("[reference dictionary loaded]")
    #if "chv-dev" in args.data_dir:
    if args.cometa:
        print ("[loading COMETA queries...]")
        eval_queries = load_queries_COMETA(       
            data_dir=args.data_dir,
            load_full_sentence=False,
            filter_duplicate=args.filter_duplicate
            )
        print ("[COMETA queries loaded]")
    elif args.custom_query_loader:
        print ("[loading custom queries...]")
        dataset = QueryDataset_custom(
                data_dir=args.data_dir,
                filter_duplicate=args.filter_duplicate
                )
        eval_queries = dataset.data
        print ("[custom queries loaded]")
    else:
        eval_queries = load_queries(
            data_dir=args.data_dir,
            filter_composite=args.filter_composite,
            filter_duplicate=args.filter_duplicate
            )

    model_wrapper = Model_Wrapper().load_model(
            path=args.model_dir,
            max_length=args.max_length,
            use_cuda=args.use_cuda,
    )
    
    print ("[start evaluating...]")
    result_evalset = evaluate(
            model_wrapper=model_wrapper,
            eval_dictionary=eval_dictionary,
            eval_queries=eval_queries,
            topk=10, # sort only the topk to save time
            cometa=args.cometa,
            medmentions=args.medmentions,
            agg_mode=args.agg_mode
            )
    
    LOGGER.info("acc@1={}".format(result_evalset['acc1']))
    LOGGER.info("acc@5={}".format(result_evalset['acc5']))
    #wandb.log({"acc1": result_evalset["acc1"], "acc5":  result_evalset["acc5"]})
    
    if args.save_predictions:
        output_file = os.path.join(args.output_dir,"predictions_eval.json")
        with open(output_file, 'w') as f:
            json.dump(result_evalset, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)
