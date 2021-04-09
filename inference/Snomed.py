import networkx as nx
import os

# Column indices of the interesting fields in SNOMED's
# Description and Relationship files
_SNOMED_TERM_FIELD_ID = 0  # term activity (deprecated terms will have active = 0)
_SNOMED_TERM_FIELD_ACTIVE = 2  # term activity (deprecated terms will have active = 0)
_SNOMED_DESC_FIELD_ACTIVE = 2  # definition activity (deprecated definition will have active = 0)
_SNOMED_DESC_FIELD_ID = 4  # term id
_SNOMED_DESC_FIELD_DEF = 7  # term description
_SNOMED_REL_FIELD_ACTIVE = 2  # relationship activity (deprecated relationship will have active = 0)
_SNOMED_REL_FIELD_ID = 7  # relationship id
_SNOMED_REL_FIELD_SOURCE = 4  # relationship source node
_SNOMED_REL_FIELD_TARGET = 5  # relationship target node

# Useful relationship IDs
_SNOMED_REL_IS_A = '116680003'


class Snomed:
    def __init__(self, snomed_path, release_id='20190731', taxonomy=True):
        self.snomed_path = snomed_path
        self.release_id = release_id
        self.definition_index = {}
        self.index_definition = {}
        self.graph = None
        self.taxonomy = taxonomy

    def load_snomed(self):

        # init graph
        self.graph = nx.DiGraph()
        
        # set of active nodes
        nodes = set()

        # load active nodes
        with open(os.path.join(
                self.snomed_path,
                f'Snapshot/Terminology/sct2_Concept_Snapshot_INT_{self.release_id}.txt'
        ),
                  mode='r',
                  encoding='utf8') as f:

            # skip header
            next(f)
            for line in f:
                concept = line.split('\t')
                if int(concept[_SNOMED_TERM_FIELD_ACTIVE]):
                    nodes.add(concept[_SNOMED_TERM_FIELD_ID])

        # load definitions
        with open(os.path.join(
                self.snomed_path,
                f'Snapshot/Terminology/sct2_Description_Snapshot-en_INT_{self.release_id}.txt'
        ),
                  mode='r',
                  encoding='utf8') as f:

            # skip header
            next(f)

            for line in f:
                definition = line.split('\t')
                cid = definition[_SNOMED_DESC_FIELD_ID]  # concept id
                cdesc = definition[
                    _SNOMED_DESC_FIELD_DEF]  # concept description

                # load only active defintions and only the first definition
                if cid in nodes and (int(definition[_SNOMED_DESC_FIELD_ACTIVE])
                                     ):
                    # add the first definition as an attribute
                    if (cid not in self.graph):
                        self.graph.add_node(cid, desc=cdesc)
                    # and put the first and the others in the indices
                    self.definition_index[cdesc] = cid
                    if cid not in self.index_definition:
                        self.index_definition[cid] = [cdesc]
                    else:
                        self.index_definition[cid].append(cdesc)
                # fi
            # for
        # with

        with open(os.path.join(
                self.snomed_path,
                f'Snapshot/Terminology/sct2_Relationship_Snapshot_INT_{self.release_id}.txt'
        ),
                  mode='r',
                  encoding='utf8') as f:

            # skip header
            next(f)

            for line in f:
                rel = line.split('\t')
                if int(rel[_SNOMED_REL_FIELD_ACTIVE]):
                    # load only IS-A relationships in taxonomy mode
                    if (self.taxonomy and (rel[_SNOMED_REL_FIELD_ID] != _SNOMED_REL_IS_A)):
                        continue
                    
                    self.graph.add_edge(
                        rel[_SNOMED_REL_FIELD_TARGET],rel[_SNOMED_REL_FIELD_SOURCE])
                # fi
            # for
        # with

    
    def __contains__(self, index):
        """
        Wrapper for `networkx.Graph.has_node()`
        """
        
        if type(index) != str:
            index = str(index)
            
            
        return index in self.graph
    
    
    def __getitem__(self, index):
        """
        Utility method to access nodes of SNOMED more easily.
        Allows using strings or ids as indices.
        
        Example:
        ```
        > snomed = Snomed('path/to/snomed')
        > snomed.load_snomed()
        > snomed['774007']
        {'desc': 'Head and neck'}
        > snomed[774007]
        {'desc': 'Head and neck'}
        ```
        """

        if type(index) != str:
            index = str(index)

        return self.graph.nodes[index]
    
    
    def predecessors(self, index):
        """
        Wrapper of networkx.digraph.predecessors()
        """
        if type(index) != str:
            index = str(index)

        return list(self.graph.predecessors(index))

    def successors(self, index):
        """
        Wrapper of networkx.digraph.successors()
        """
        if type(index) != str:
            index = str(index)

        return list(self.graph.successors(index))
    
    def distance(self, source, target):
        """
        Computes the distance between two nodes.
        """
        
        if type(source) != str:
            source = str(source)
            
        if type(target) != str:
            target = str(target)
            
        if nx.has_path(self.graph,source=source, target=target):        
            return nx.shortest_path_length(self.graph,source=source,target=target)
        else:
            return nx.shortest_path_length(self.graph,source=target,target=source)
        
    
    def is_ancestor(self, source, target):
        """
        Returns True if `source` is an ancestor of `target` in the SNOMED taxonomy.
        """
        
        if type(source) != str:
            source = str(source)
            
        if type(target) != str:
            target = str(target)
            
        return nx.has_path(self.graph,source=source, target=target)
    
        
    def safe_distance(self, source, target):
        """
        Computes the distance between two nodes. If there's not path between the source
        and target node, returns -1.
        """
        
        if type(source) != str:
            source = str(source)
            
        if type(target) != str:
            target = str(target)
            
        if nx.has_path(self.graph,source=source, target=target):        
            return nx.shortest_path_length(self.graph,source=source,target=target)
        elif nx.has_path(self.graph,source=target, target=source):  
            return nx.shortest_path_length(self.graph,source=target,target=source)
        else:
            return -1