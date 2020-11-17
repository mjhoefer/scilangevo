import csv
import numpy as np
import networkx as nx
from operator import itemgetter
import pandas as pd
from pathlib import Path
import json
import string

# for creating XML corpus for sketch engine
import xml.etree.cElementTree as ET




# functions taken from other file
def extract_lemma_text(json_file):
    data = json.loads(json_file.read_bytes())
    contents = ""

    # now we need to get the paper contents into something that can be parsed using TWEC.
    # This will largely be concatinating the tokens found in the json structure.
    data['full_text'] = ''
    data['lemmas'] = []
    for section in data['sections']:
        for subsection in section['subsections']:
            for sentence in subsection['sentences']:
                for token in sentence['tokens']:
                    if token['lemma'] not in string.punctuation:
                        contents += token['lemma'] + ' '

    return contents

def extract_regular_text(json_file):
    data = json.loads(json_file.read_bytes())
    contents = ""

    # now we need to get the paper contents into something that can be parsed using TWEC.
    # This will largely be concatinating the tokens found in the json structure.
    data['full_text'] = ''
    data['lemmas'] = []
    for section in data['sections']:
        for subsection in section['subsections']:
            for sentence in subsection['sentences']:
                for token in sentence['tokens']:
                    contents += token['word'] + ' '

    return contents

def write_out_contents(contents, path_to_write):
    with open(path_to_write, 'w', encoding='utf8') as outfile:
        outfile.write(contents)


def extract_lemma_text_and_write(json_file):
    content = extract_lemma_text(json_file)
    to_write = json_file.parent / (json_file.stem + '.words')
    write_out_contents(content, to_write)

def extract_regular_text_and_write(json_file, folder_dir = None, extension = '.words'):
    content = extract_regular_text(json_file)
    if not folder_dir:
        to_write = json_file.parent / (json_file.stem + extension)
    else:
        to_write = folder_dir / (json_file.stem + extension)
    write_out_contents(content, to_write)


def get_path_from_id(id, base_dir, extension='.words'):
    # need first char and the last two nums to get it all.
    first_char = id[0]
    two_nums = id[1:3]

    paper_dir = base_dir / first_char / (first_char + two_nums)

    # handle to particular file
    paper = paper_dir / (id + extension)

    return paper


### End functions taken from other file



#network_path = 'acl-arc.citation-graph.with-functions.tsv'

network_path = 'network.txt'
id_code_path = 'arc-paper-ids.tsv'

# read in the network
G = nx.read_edgelist(network_path, create_using=nx.DiGraph(), delimiter=' ', nodetype=str)

G.number_of_edges()
G.number_of_nodes()

in_degs = list(G.in_degree())

in_degs_s = sorted(in_degs,key=itemgetter(1), reverse=True)


# get the top 10 papers that are not 'external' to the corpus
num_papers = 0
max_papers = 10
papers_to_look_at = []
for i, (key, val) in enumerate(in_degs_s):
    if num_papers > max_papers:
        break
    if not key.startswith("External"):
        papers_to_look_at.append((key, val))
        num_papers += 1

# get all the text files!
p = Path('.')
p = p / '..' / 'project_code' / 'papers'
p = p / 'acl-arc-json' / 'json'

outdir = Path('.') / 'papers_for_tim'
root = ET.Element("xml")
year = 2000
for paper in papers_to_look_at:
    p_obj = get_path_from_id(paper[0], p, '.json')
    if p_obj.is_file():
        temp_content = extract_regular_text(p_obj)
        doc = ET.SubElement(root, "doc", year=str(year)).text = temp_content
        year += 1
    else:
        print("Missing a file ", str(p_obj))

tree = ET.ElementTree(root)
tree.write("top_10_corpus.xml")

exit()


df_p = pd.DataFrame(papers_to_look_at)
df_p.columns = ['paper_id', 'citations']

# read in the paper ids
df = pd.read_csv(id_code_path, sep='\t')

df.columns


# merge the dataframes

df_final = pd.merge(df_p, df, left_on='paper_id', how='inner', right_on = 'id')


df_final_nodupes = df_final.drop_duplicates(subset=['id'])

# this is a sorted list of our candidates
df_final_nodupes.to_csv('top_1000_papers.csv', header=True)


# next we need to create word embeddings!