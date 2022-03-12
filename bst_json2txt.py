from pathlib import Path
import json

t = 'valid'

fname = Path(f'with_concepts/bst/{t}.json')
with open(fname,'r') as f:
    bst_new = json.load(f)
fname = Path(f'/home/ilya/repos/ParlAI/data/blended_skill_talk/{t}.txt')
with open(fname,'r') as f:
    bst = f.readlines()

ut_dict = {}
for episode in bst_new:
    for d,c in zip(episode['dialog'],episode['concepts']):
        text = d[1]
        ut_dict[text] = c
new_lines = []
for line in bst:
    for info in line.split('\t'):
        if info.startswith('free_message:'):            
            msg = info.replace('free_message:','')
            concepts = ut_dict[msg]
            concepts_idx = line.find('\tconcepts:')
            if concepts_idx != -1:
                line = line[:concepts_idx]
            new_line = line.strip() + '\tconcepts:' + '|'.join(concepts) + '\n'            
            new_lines.append(new_line)

with open(f'with_concepts/bst/{t}.txt', 'w') as f:
    f.writelines(new_lines)