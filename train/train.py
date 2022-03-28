#!/usr/bin/env python
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from pytorch_metric_learning import samplers
import logging
import time
import pdb
import os
import json
import random
from tqdm import tqdm

import sys
sys.path.append("../") 

import wandb
wandb.init(project="sapbert")

from src.data_loader import (
    DictionaryDataset,
    QueryDataset,
    QueryDataset_pretraining,
    MetricLearningDataset,
    MetricLearningDataset_pairwise,
)
from src.model_wrapper import (
    Model_Wrapper
)
from src.metric_learning import (
    Sap_Metric_Learning,
)

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert train')

    # Required
    parser.add_argument('--model_dir', 
                        help='Directory for pretrained model')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=240, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=3, type=int)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--amp', action="store_true", 
            help="automatic mixed precision training")
    parser.add_argument('--parallel', action="store_true") 
    #parser.add_argument('--cased', action="store_true") 
    parser.add_argument('--pairwise', action="store_true",
            help="if loading pairwise formatted datasets") 
    parser.add_argument('--random_seed',
                        help='epoch to train',
                        default=1996, type=int)
    parser.add_argument('--loss',
                        help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}",
                        default="ms_loss")
    parser.add_argument('--use_miner', action="store_true") 
    parser.add_argument('--miner_margin', default=0.2, type=float) 
    parser.add_argument('--type_of_triplets', default="all", type=str) 
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_all_tok}") 

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def load_queries_pretraining(data_dir, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset_pretraining(
        data_dir=data_dir,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def train(args, data_loader, model, scaler=None, model_wrapper=None, step_global=0):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()

        batch_x1, batch_x2, batch_y = data
        batch_x_cuda1, batch_x_cuda2 = {},{}
        for k,v in batch_x1.items():
            batch_x_cuda1[k] = v.cuda()
        for k,v in batch_x2.items():
            batch_x_cuda2[k] = v.cuda()

        batch_y_cuda = batch_y.cuda()
    
        if args.amp:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)  
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)  
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1
        #if (i+1) % 10 == 0:
        #LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1,train_loss / (train_steps+1e-9)))
        #LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1, loss.item()))

        # save model every K iterations
        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_iter_{}".format(str(step_global)))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global
    
def main(args):
    init_logging()
    #init_seed(args.seed)
    print(args)

    torch.manual_seed(args.random_seed)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    # load BERT tokenizer, dense_encoder
    model_wrapper = Model_Wrapper()
    encoder, tokenizer = model_wrapper.load_bert(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        #lowercase=not args.cased
    )
    
    # load SAP model
    model = Sap_Metric_Learning(
            encoder = encoder,
            learning_rate=args.learning_rate, 
            weight_decay=args.weight_decay,
            use_cuda=args.use_cuda,
            pairwise=args.pairwise,
            loss=args.loss,
            use_miner=args.use_miner,
            miner_margin=args.miner_margin,
            type_of_triplets=args.type_of_triplets,
            agg_mode=args.agg_mode,
    )

    if args.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)
        LOGGER.info("using nn.DataParallel")
    
    def collate_fn_batch_encoding(batch):
        query1, query2, query_id = zip(*batch)
        query_encodings1 = tokenizer.batch_encode_plus(
                list(query1), 
                max_length=args.max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")
        query_encodings2 = tokenizer.batch_encode_plus(
                list(query2), 
                max_length=args.max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")
        #query_encodings_cuda = {}
        #for k,v in query_encodings.items():
        #    query_encodings_cuda[k] = v.cuda()
        query_ids = torch.tensor(list(query_id))
        return  query_encodings1, query_encodings2, query_ids

    if args.pairwise:
        train_set = MetricLearningDataset_pairwise(
                path=args.train_dir,
                tokenizer = tokenizer
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=16,
            collate_fn=collate_fn_batch_encoding
        )
    else:
        train_set = MetricLearningDataset(
            path=args.train_dir,
            tokenizer = tokenizer
        )
        # using a sampler
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.train_batch_size,
            #shuffle=True,
            sampler=samplers.MPerClassSampler(train_set.query_ids,\
                2, length_before_new_iter=100000),
            num_workers=16, 
            )
    # mixed precision training 
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    step_global = 0
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))

        # train
        train_loss, step_global = train(args, data_loader=train_loader, model=model, scaler=scaler, model_wrapper=model_wrapper, step_global=step_global)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        
        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
        
        # save model last epoch
        if epoch == args.epoch:
            model_wrapper.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
