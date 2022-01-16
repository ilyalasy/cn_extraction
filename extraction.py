from pathlib import Path
from typing import List, Literal
import spacy
import networkx as nx
import pandas as pd
import numpy as np 
import re

import json
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def extract_relation(r):
    s = r.replace('/r/','')
    s = re.sub('([A-Z]+)', r' \1', s).strip()
    return s

def extract_word(w):
    s = re.search('\/c\/en\/(.*?)(\/|$)', w).group(1)
    s = s.replace('_',' ')
    return s

def create_graph(df):
    g = nx.DiGraph()
    for i,row in df.iterrows():
        start = extract_word(row['from'])
        end = extract_word(row['to'])
        r = extract_relation(row['relation'])
        
        g.add_edge(start,end, relation=r,weight=row['weight'])
    return g

def filter_tokens(tokens,tokens_cache,return_lemmas=True):
    new_tokens = []
    for token in tokens:
        if not token.is_stop and not token.is_punct and token.text.lower() not in tokens_cache:
            if return_lemmas:
                new = token.lemma_
            else:
                if token.dep_ in ['compound','amod']:
                  new = token.text.lower()
                else:
                  new = token.lemma_.lower()
            new_tokens.append(new)
            tokens_cache[token.text.lower()] = 'cached'
    return new_tokens

def get_tokens(sent,tokens_cache):
    tokens = []
    prev_start = 0
    
    for chunk in sent.noun_chunks:
        if filtered:=filter_tokens(sent[prev_start:chunk.start], tokens_cache):
          tokens.append(filtered)
        if filtered:=filter_tokens(chunk,tokens_cache, return_lemmas=False):
          tokens.append(filtered)
        prev_start = chunk.end
    if filtered:=filter_tokens(sent[prev_start:],tokens_cache):
        tokens.append(filtered)
    return tokens

def preprocess(msg_str):
    global nlp, sent2wec
    doc = nlp(msg_str)
    vectors = sent2wec.encode([s.text for s in doc.sents])
    preprocessed = []
    tokens_cache = {}
    for sent,vec in zip(doc.sents,vectors):
        tokens = get_tokens(sent,tokens_cache)        
        preprocessed.append((tokens,vec) )
    return preprocessed

def get_relations(c):
    global conceptnet
    rels = []
    if c in conceptnet:
        for n, attrs in conceptnet[c].items():
            rels.append(f"{c} {attrs['relation']} {n}")
    return rels

def filter_relations(rels_vecs,sent_vec,limit=10):
    sims = cosine_similarity(rels_vecs,sent_vec)
    return np.argsort(sims,axis=0)[::-1][:limit].flatten()    

def flip_relation(rel):
    return rel.split(' ')[::-1]

def extract_from_msg(msg,limit=10):
    global sent2wec
    sentence_tokens = preprocess(msg)    
    all_rels = set()
    for sent,vec in sentence_tokens:
        for token in sent:
            rels = []
            if len(token) > 1:
                t = ' '.join(token)
                rels += get_relations(t)                
            for t in token:                                
                rels += get_relations(t)
            if rels:
                rels_vecs = sent2wec.encode(rels)
                vec = vec.reshape(1, -1)
                idxs = filter_relations(rels_vecs,vec,limit=limit)                
                rels = np.array(rels)     
                rels = rels[idxs]
                # rels = [r for r in rels if flip_relation(r) not in all_rels]
                all_rels.update(rels)
    return list(all_rels)

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    @property
    def total(self):
        return self._total
        
    @total.setter
    def total(self,value):
        self._total = value

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def process_convai_dialog(sample):
    if 'persona:' in sample:
        return sample

    x,y = sample.split('\t')
    msg = x.lstrip('0123456789 ')
    extracted = extract_from_msg(msg,limit=3)
    extracted = '|'.join(extracted)
    return f'{x}\t{y.strip()}\t{extracted}\n'

def process_bst_dialog(sample):        
    concepts = []
    for s_id,msg in sample['dialog']:
        extracted = extract_from_msg(msg,limit=3)
        concepts.append(extracted)
    return {'dialog':sample['dialog'], 'concepts':concepts,'personas':sample['personas']}
            
NEW_PATH = Path('with_concepts/')
func_map = {'bst':process_bst_dialog,'convai':process_convai_dialog}
def create_dataset(n_jobs:int, ds_paths:List[str], dataset:Literal['bst','convai']):
    process_dialog = func_map[dataset]
    with ProgressParallel(n_jobs=n_jobs) as parallel:
        for ds in ds_paths:
            ds = Path(ds)
            print(ds.name)
            with open(ds, 'r') as f:
                if ds.suffix == '.json':
                    data = json.loads(f.read())
                elif ds.suffix == '.txt':
                    data = f.readlines()    
                else:
                    raise NotImplementedError()      

            # data = data[:20]
            parallel.total = len(data)
            res = parallel(delayed(process_dialog)(sample) for sample in data)
            
            new_p = NEW_PATH / dataset 
            new_p.mkdir(parents=True,exist_ok=True)
            with open(new_p / ds.name, 'w') as f:
                if ds.suffix == '.json':
                    json.dump(res,f)
                elif ds.suffix == '.txt':
                    f.writelines(res)                    
                
            print('Processed dataset!')

sent2wec = None
nlp = None
conceptnet = None
def setup_models():
    global sent2wec,nlp,conceptnet
    print('Setting up all models...')
    sent2wec = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm",disable=['ner'])
    print('Models ready!')
    print('Loading conceptnet...')
    conceptnet = pd.read_csv('conceptnet_en_filtered.csv')
    conceptnet = create_graph(conceptnet)
    print('Conceptnet ready!')

if __name__ == '__main__':
    setup_models()
    bst_paths = [
        '/home/ilya/repos/ParlAI/data/blended_skill_talk/train.json',
        '/home/ilya/repos/ParlAI/data/blended_skill_talk/test.json',
        '/home/ilya/repos/ParlAI/data/blended_skill_talk/valid.json',
    ]
    
    convai_paths = [
        '/home/ilya/repos/ParlAI/data/ConvAI2/train_both_original_no_cands.txt',
        '/home/ilya/repos/ParlAI/data/ConvAI2/valid_both_original_no_cands.txt',
    ]


    create_dataset(1,convai_paths,dataset='convai')

# nohup python -u bst_extraction.py &