import json
import dask.dataframe as dd
from nltk.stem.snowball import SnowballStemmer

from extraction import extract_word
stemmer = SnowballStemmer("english")

def stem(concept):
    w = extract_word(concept)
    return stemmer.stem(w)
    
def filter(cn_path='conceptnet-assertions-5.7.0.csv'):
    kb = dd.read_csv(cn_path,sep='\t',names=['URI','relation','from','to','extra'])

    result = kb[kb['from'].str.startswith('/c/en/') & kb['to'].str.startswith('/c/en/')]
    print("Filtering only english")
    en_kb = result.compute()
    print("Filtering by weight")
    en_kb['weight'] = en_kb['extra'].apply(lambda x:json.loads(x)['weight'])
    en_kb = en_kb[en_kb['weight']>=1]

    print("Stemming")
    from_stems = en_kb['from'].apply(stem)
    to_stems = en_kb['to'].apply(stem)
    en_kb = en_kb[from_stems != to_stems]

    en_kb.to_csv('conceptnet_en_filtered.csv',index=False)

    relations = (en_kb.relation.value_counts(normalize=True).mul(100).round(1).astype(str) + '%').head(16)
    relations = relations[~relations.index.isin(['/r/Synonym','/r/Antonym','/r/DerivedFrom'])].index.tolist()
    print(spec_tokens = [f'__{r[3:]}__' for r in relations])

    en_kb[en_kb.relation.isin(relations)].to_csv('conceptnet_en_filtered_13.csv', index=False)