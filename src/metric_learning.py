import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
import random
from torch.cuda.amp import autocast
from pytorch_metric_learning import miners, losses, distances
LOGGER = logging.getLogger(__name__)


class Sap_Metric_Learning(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda, pairwise, 
            loss, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls"):

        LOGGER.info("Sap_Metric_Learning! learning_rate={} weight_decay={} use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
            learning_rate,weight_decay,use_cuda,loss,use_miner,miner_margin,type_of_triplets,agg_mode
        ))
        super(Sap_Metric_Learning, self).__init__()
        self.encoder = encoder
        self.pairwise = pairwise
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07) # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()


        print ("miner:", self.miner)
        print ("loss:", self.loss)
    
    @autocast() 
    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        
        last_hidden_state1 = self.encoder(**query_toks1, return_dict=True).last_hidden_state
        last_hidden_state2 = self.encoder(**query_toks2, return_dict=True).last_hidden_state
        if self.agg_mode=="cls":
            query_embed1 = last_hidden_state1[:,0]  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2[:,0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            query_embed1 = last_hidden_state1.mean(1)  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            query_embed1 = (last_hidden_state1 * query_toks1['attention_mask'].unsqueeze(-1)).sum(1) / query_toks1['attention_mask'].sum(-1).unsqueeze(-1)
            query_embed2 = (last_hidden_state2 * query_toks2['attention_mask'].unsqueeze(-1)).sum(1) / query_toks2['attention_mask'].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        
        labels = torch.cat([labels, labels], dim=0)
        
        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs) 
        else:
            return self.loss(query_embed, labels) 


    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table



