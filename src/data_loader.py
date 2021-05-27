import re
import os
import glob
import numpy as np
import random
import random
import pandas as pd
import json
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
LOGGER = logging.getLogger(__name__)


class QueryDataset_COMETA(Dataset):

    def __init__(self, data_dir, 
                load_full_sentence=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, load_full_sentence, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            load_full_sentence=load_full_sentence,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, load_full_sentence, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        data_table = pd.read_csv(data_dir, sep='\t', encoding='utf8')

        for row in data_table.iterrows():
            mention = row[1]["Term"]
            sentence = row[1]["Example"]
            
            #print (mention)
            #print (sentence)

            cui = row[1]["General SNOMED ID"] # TODO: allow general/specific options
            if load_full_sentence: 
                data.append((mention, sentence, cui))
            else:
                data.append((mention, cui))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        print ("query size:",len(data))
        
        # return np.array data
        data = np.array(data)
        
        return data

class QueryDataset_custom(Dataset):

    def __init__(self, data_dir, 
                load_full_sentence=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_duplicate={}".format(
            data_dir, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        with open(data_dir, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip("\n")
            if len(line.split("||")) == 2:
                _id, mention = line.split("||")
            elif len(line.split("||")) == 3: # in case using data with contexts
                _id, mention, context = line.split("||")
            else:
                raise NotImplementedError()
             
            data.append((mention, _id))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data, dtype=object)
        
        return data

class QueryDataset_pretraining(Dataset):

    def __init__(self, data_dir, 
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_duplicate={}".format(
            data_dir,filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        with open(data_dir, "r") as f:
            lines = f.readlines()

        for row in lines:
            row = row.rstrip("\n")
            snomed_id, mention = row.split("||")
            data.append((mention, snomed_id))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data

class QueryDataset(Dataset):

    def __init__(self, data_dir, 
                filter_composite=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, filter_composite, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_composite, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.txt"))
        file_types = ("*.concept", "*.txt")
        concept_files = []
        for ft in file_types:
            concept_files.extend(glob.glob(os.path.join(data_dir, ft)))

        for concept_file in tqdm(concept_files):
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            for concept in concepts:
                #print (concept)
                concept = concept.split("||")
                #if len(concept) !=5: continue
                mention = concept[3].strip().lower()
                cui = concept[4].strip()
                if cui.lower() =="cui-less": continue
                is_composite = (cui.replace("+","|").count("|") > 0)

                if filter_composite and is_composite:
                    continue
                else:
                    data.append((mention,cui))
        
        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data


class DictionaryDataset():
    """
    A class used to load dictionary data
    """
    def __init__(self, dictionary_path):
        """
        Parameters
        ----------
        dictionary_path : str
            The path of the dictionary
        draft : bool
            use only small subset
        """
        LOGGER.info("DictionaryDataset! dictionary_path={}".format(
            dictionary_path 
        ))
        self.data = self.load_data(dictionary_path)
        
    def load_data(self, dictionary_path):
        name_cui_map = {}
        data = []
        with open(dictionary_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "": continue
                cui, name = line.split("||")
                name = name.lower()
                if cui.lower() == "cui-less": continue
                data.append((name,cui))
        
        #LOGGER.info("concerting loaded dictionary data to numpy array...")
        #data = np.array(data)
        return data

class MetricLearningDataset_pairwise(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        for line in lines:
            line = line.rstrip("\n")
            query_id, name1, name2 = line.split("||")
            self.query_ids.append(query_id)
            self.query_names.append((name1, name2))
        self.tokenizer = tokenizer
        self.query_id_2_index_id = {k: v for v, k in enumerate(list(set(self.query_ids)))}
    
    def __getitem__(self, query_idx):

        query_name1 = self.query_names[query_idx][0]
        query_name2 = self.query_names[query_idx][1]
        query_id = self.query_ids[query_idx]
        query_id = int(self.query_id_2_index_id[query_id])

        return query_name1, query_name2, query_id


    def __len__(self):
        return len(self.query_names)



class MetricLearningDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        LOGGER.info("Initializing metric learning data set! ...")
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        cuis = []
        for line in lines:
            cui, _ = line.split("||")
            cuis.append(cui)

        self.cui2id = {k: v for v, k in enumerate(cuis)}
        for line in lines:
            line = line.rstrip("\n")
            cui, name = line.split("||")
            query_id = self.cui2id[cui]
            #if query_id.startswith("C"):
            #    query_id = query_id[1:]
            #query_id = int(query_id)
            self.query_ids.append(query_id)
            self.query_names.append(name)
        self.tokenizer = tokenizer
    
    def __getitem__(self, query_idx):

        query_name = self.query_names[query_idx]
        query_id = self.query_ids[query_idx]
        query_token = self.tokenizer.transform([query_name])[0]

        return torch.tensor(query_token), torch.tensor(query_id)

    def __len__(self):
        return len(self.query_names)
