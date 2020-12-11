from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec
from pathlib import Path
import json
import string
import networkx as nx
from operator import itemgetter

import time

import numpy

# for creating XML corpus for sketch engine
import xml.etree.cElementTree as ET

# for data analysis and viz
import pandas as pd
import altair as alt
import altair_viewer

# for time saving expensive network computations
import pickle

# for counting word occurances for embeddings
from collections import Counter

# for melting the dataframe by hand (ran out of memory with pandas)
import csv



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


# stop_words is a list of stop words
def extract_regular_text_limited_words(json_file, stop_words):
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
                    if token['word'] not in stop_words and len(token['word']) > 2:
                        contents += token['word'] + ' '

    return contents


def get_year(id, p):
    file = get_path_from_id(id, p, extension='.json')
    data = json.loads(file.read_bytes())
    return data['year']


# combines all the papers in the papers list into a single text file
# papers is a list of pathlib objects
# writes to outfile_name
# concats all the text files in papers into the outfile_name
# returns the outfile_name as a path object for convenience
def create_text_slice(papers, outfile_name):
    with open(outfile_name, 'w', encoding='utf-8') as out:
        for paper in papers:
            out.write(paper.read_text(encoding='utf-8') + '\n')
    return Path(outfile_name)


# returns a pathlib object from an id string
def get_path_from_id(id_in, base_dir, extension='.words'):
    # need first char and the last two nums to get it all.
    # print (id_in, '\n')
    if not isinstance(id_in, str):
        print("BREAK at: ", id_in)
    first_char = id_in[0]
    two_nums = id_in[1:3]

    paper_dir = base_dir / first_char / (first_char + two_nums)

    # handle to particular file
    paper = paper_dir / (id_in + extension)

    if paper.is_file():
        return paper
    else:
        return None


# not all papers in the citation network are actually in the corpus
# this function returns a list of every paper we have a .words file for
def get_list_of_papers(p):
    files_owned = []
    # walk dir and get files we need to parse
    for path_object in p.glob('**/*'):
        if path_object.is_file():
            # gather all json files
            if path_object.suffix == '.words':
                files_owned.append(path_object.stem)

    return files_owned


# turns a list of id strings into a list of Pathlib file objects
def get_paths_from_strings(ids, p):
    papers = []
    for id in ids:
        temp = get_path_from_id(id, p)
        if temp:
            papers.append(temp)
    return papers


# recursive function that returns a list of papers either upstream or downstream along the citation network
def get_connected_papers(G, node_id, curr_list=None, upstream=False):
    master = False
    if not curr_list:
        curr_list = set()
        master = True

    # get the list of connected papers
    if upstream:
        # sort of confusing, but the citation network "looks backward" so successors are technically upstream
        node_successors = list(G.successors(node_id))
    else:
        node_successors = list(G.predecessors(node_id))

    # and remove any elements that are already captured in the global list
    node_successors_real = [node for node in node_successors if node not in curr_list]

    # end condition - no new nodes to add to the global list
    if len(node_successors_real) == 0:
        return None
    else:
        for node in node_successors_real:
            curr_list.add(node)
            get_connected_papers(G, node, curr_list, upstream)

    if master:
        return curr_list



def get_stop_words(path_to_stop):
    to_ret = []
    with open(path_to_stop, 'r', encoding='utf-8') as infile:
        to_ret = infile.readlines()
    return [i.strip() for i in to_ret]


def create_xml_corpus(list_o_papers, p, outfile_path):
    root = ET.Element("xml")

    for paper in list_o_papers:
        p_obj = get_path_from_id(paper, p, '.json')
        if p_obj.is_file():
            temp_content = extract_regular_text(p_obj)
            curr_year = get_year(paper, p)
            doc = ET.SubElement(root, "doc", year=str(curr_year), author=str(paper),
                                title=paper[0]).text = "<s>" + temp_content + "</s>"
        else:
            print("Missing a file ", str(p_obj))

    tree = ET.ElementTree(root)
    tree.write(outfile_path)


def create_xml_corpus_reduced_words(list_o_papers, p, outfile_path, stop_word_path='STOPWORDS.txt'):
    root = ET.Element("xml")

    stops = get_stop_words(stop_word_path)

    for paper in list_o_papers:
        p_obj = get_path_from_id(paper, p, '.json')
        if p_obj.is_file():
            temp_content = extract_regular_text_limited_words(p_obj, stops)
            curr_year = get_year(paper, p)
            doc = ET.SubElement(root, "doc", year=str(curr_year), author=str(paper),
                                title=paper[0]).text = "<s>" + temp_content + "</s>"
        else:
            print("Missing a file ", str(p_obj))

    tree = ET.ElementTree(root)
    tree.write(outfile_path)


def cosine_similarity(word1, word2):
    to_ret = numpy.dot(word1, word2) / (
                numpy.linalg.norm(word1) * numpy.linalg.norm(word2))
    return to_ret


# model_list is a list of actual word2vec models
def get_intersection_vocab(model_list, limit_to_vocab_list=None):
    words = []

    for model in model_list:
        if len(words) == 0:
            words += model.wv.vocab.keys()
        words = [i for i in words if i in model.wv.vocab.keys()]

    if limit_to_vocab_list:
        return [i for i in words if i in limit_to_vocab_list]
    else:
        return words


# model_list is a ist of actual word2vec models
def get_union_vocab(model_list):
    words = []

    for model in model_list:
        words += model.wv.vocab.keys()
    return Counter(words)


def get_average_similarity(sim_dict):
    total = 0
    for key, value in sim_dict.items():
        total += value
    return total/len(sim_dict)


# gets all the similarity scores for two models
# if word is not in one of the models, it will just store None
# default word list is the intersection vocab set
# structure of the returned dict is key=word, value=similarity
def get_similarity_dict(model1, model2, words=None):
    if not words:
        meepers = [i for i in model1.wv.vocab.keys() if i in model2.wv.vocab.keys()]
    else:
        meepers=words
    similarities = {}

    for i, word in enumerate(meepers):
        if word not in model1.wv.vocab.keys() or word not in model2.wv.vocab.keys():
            similarities[meepers[i]] = None
        else:
            # get the similarity score for every word
            similarities[meepers[i]] = cosine_similarity(model1[meepers[i]], model2[meepers[i]])

    return similarities


# this function does all the pairwise comparisons between the dict of model_handles and the compass_handle
# basically getting a similarity score for each paper with the overall compass, across all the words
# in the word list
# if the particular word isn't in the paper, it will have None in the slot
# returns a list of dicts, easy for turning into a dataframe
def get_df_for_compass_comparisons(model_handles, compass_handle, word_list):
    my_dicts = []
    for key, model in model_handles.items():
        d = get_similarity_dict(model, c_handle, word_list)
        d['id'] = key
        d['citations'] = citation_counts[key]
        my_dicts.append(d)
    return my_dicts


## FUNCTION TO GENERATE FIRST RESULT IN PAPER - GEODESIC distance vs. EMBEDDING DISTANCE
## This function does the pairwise comparisons model to model
## it will return a list of dicts.
# # Containing Model 1, model 2, network distance (geodesic), and then similarity scores for the set of words
def get_pairwise_model_comparisons(model_handles, word_list, geodesic_dict, verbose=False):
    # the list o dicts
    myDicts = []

    # keep track of models we're done with
    completed_handles = []

    num_models = len(model_handles)
    num_perms = num_models ** 2
    one_percent = num_perms / 100
    curr_percent = 0
    if verbose:
        print ("need to run", num_perms, " model evaluations. One percent is ", one_percent)
    curr_count = 0

    # all permutations of model pairs
    for key1, model1 in model_handles.items():

        # for each model, create a pairwise model
        for key2, model2 in model_handles.items():
            if key2 not in completed_handles and not key2 == key1:
                if curr_count > (curr_percent * one_percent) + one_percent:
                    curr_percent += 1
                    if verbose:
                        print(curr_count, " pairs done. Percent complete: ", curr_percent)
                ## run the comparisons
                similarity_scores = get_similarity_dict(model1, model2, word_list)
                similarity_scores['model1'] = key1
                similarity_scores['model2'] = key2
                geo_dist = -1
                if key1 in geodesic_dict.keys():
                    if key2 in geodesic_dict[key1].keys():
                        geo_dist = geodesic_dict[key1][key2]
                if geo_dist == -1:
                    # check the reverse path
                    if key2 in geodesic_dict.keys():
                        if key1 in geodesic_dict[key2].keys():
                            geo_dist = geodesic_dict[key2][key1]
                similarity_scores['geodesic_dist'] = geo_dist
                myDicts.append(similarity_scores)
                curr_count += 1
        completed_handles.append(key1)
    return myDicts

# log files will be pickels of dicts with info
def read_log_file(log_path):
    return pickle.load(open(str(log_path), 'rb'))



## WINDOW DEFAULTS TO 5 ON WORD TO VEC!! INCREASE WINDOW SIZE TO GET MORE SEMANTIC RICHNESS
## OTHERWISE YOU MAY PICK UP ONLY SYNTACTIC SIMILARITIES
class Experiment:
    def __init__(self, prefix_in, size_in=50, siter_in=10, diter_in=10, workers_in=4, exp_dir='.',
                 paper_dir_in=Path('.') / '..' / 'project_code' / 'papers' / 'acl-arc-json' / 'json', window_in=5, g_in=None, p_deets_in=None):
        self.prefix = prefix_in
        self.size = size_in
        self.siter = siter_in
        self.diter = diter_in
        self.workers = workers_in
        self.embeddings = []
        self.slices = []
        self.compass_path = None  # path to the Compass word2vec model
        self.logfile = prefix_in + "__EXPERIMENT_LOG.log"
        self.aligner = None
        self.window = window_in
        self.paper_dir = paper_dir_in
        self.models = {}  # key is the paper id. Value is the model path
        self.graph = g_in

        self.training_deets = {}  # this dict holds info about the networked corpus training
        # path to log will have num upstream and downstream papers
        # id: {num_downstream: X, num_upstream: Y, log_path: path_to_log}

        self.paper_deets = p_deets_in

        # make the path if it does not exist yet
        if not Path(exp_dir).is_dir():
            Path(exp_dir).mkdir()
        self.experiment_dir = Path(exp_dir)

    def create_text_slice(self, list_of_paper_ids, txt_save_loc=None):
        paths = get_paths_from_strings(list_of_paper_ids, self.paper_dir)
        if not txt_save_loc:
            txt_save_loc = self.prefix + "_compass_text.txt"

        txt_save_loc = self.experiment_dir / txt_save_loc
        to_ret = create_text_slice(paths, str(txt_save_loc))
        return to_ret

    def train_compass(self, compass_text_path):
        self.aligner = TWEC(size=self.size, siter=self.siter, diter=self.diter, workers=self.workers, window=self.window)
        start2 = time.time()
        self.aligner.train_compass(compass_text_path, overwrite=False, filename=self.prefix + "__COMPASS.model")
        self.compass_path = str((Path('model') / (self.prefix + "__COMPASS.model")))
        end2 = time.time()
        elapsed2 = end2 - start2

        # write the training time to a file
        with open(self.logfile, 'w+', encoding='utf-8') as outfile:
            outfile.write(
                "compass training time: " + str(elapsed2) + " seconds")

    # highest level function - just falls a create text slice and a train compass function
    def train_compass_from_id_list(self, id_list):
        train_txt = self.create_text_slice(id_list)
        self.train_compass(str(train_txt))

    def train_single_paper_embeddings(self, ids):
        if not self.aligner:
            print ("Train compass first, use train_compass")
            return
        for id in ids:
            # check to see if the model already exists.
            # if it does, do not retrain
            if (Path('model') / (self.prefix + '_' + id + '.model')).is_file():
                self.models[id] = str((Path('model') / (self.prefix + '_' + id + '.model')))
            else:
                # get the paper
                c_path = get_path_from_id(id, self.paper_dir)
                slice_txt = self.create_text_slice([id], self.prefix + "_" + id + ".txt")

                # train a model for it!
                #new_slice = self.aligner.train_slice(str(slice_txt), save=True, saveName = str(self.experiment_dir / (self.prefix + '_' + id)))

                new_slice = self.aligner.train_slice(str(slice_txt), save=True,
                                                     saveName=(self.prefix + '_' + id))

                # what is new_slice?
                #add to model list
                self.models[id]= str((Path('model') / (self.prefix + '_' + id + '.model')))


    # this is for the nuanced experiment training embeddings on the entire upstream/downstream corpus
    def train_networked_corpus_embeddings(self, ids):
        if not self.aligner:
            print ("Train compass first, use train_compass")
            return

        if not self.graph:
            print ("Must supply a graph as argument in order to train on networked corpora")
            return


        count = 0
        # each paper will have three models trained. Upstream, self, and downstream.
        for id in ids:
            count += 1
            print ("starting on number ", count, " id: ", id)
            # check to see if the model already exists.
            # if it does, do not retrain

            log_path = (Path('network_training_logs') / (self.prefix + '_' + id + '.log'))
            if log_path.is_file():
                # log file complete. Let's read it and see what models we have
                deets = read_log_file(log_path)
                if deets['upstream_count'] > 0:
                    # load upstream model
                    if (Path('model') / (self.prefix + '_' + id + '_UPSTREAM.model')).is_file():
                        self.models[id + '_UPSTREAM'] = str((Path('model') / (self.prefix + '_' + id + '_UPSTREAM.model')))
                if deets['downstream_count'] > 0:
                    # load downstream model
                    if (Path('model') / (self.prefix + '_' + id + '_DOWNSTREAM.model')).is_file():
                        self.models[id + '_DOWNSTREAM'] = str(
                            (Path('model') / (self.prefix + '_' + id + '_DOWNSTREAM.model')))

                # load the paper-only model
                if (Path('model') / (self.prefix + '_' + id + '.model')).is_file():
                    self.models[id] = str((Path('model') / (self.prefix + '_' + id + '.model')))
            else:  # no log file, hasn't been trained yet
                deets = {}
                # get the paper
                c_path = get_path_from_id(id, self.paper_dir)
                slice_txt = self.create_text_slice([id], self.prefix + "_" + id + ".txt")

                # train a model for it!
                #new_slice = self.aligner.train_slice(str(slice_txt), save=True, saveName = str(self.experiment_dir / (self.prefix + '_' + id)))

                start_t = time.time()
                new_slice = self.aligner.train_slice(str(slice_txt), save=True,
                                                     saveName=(self.prefix + '_' + id))
                end_t = time.time()
                deets['self_training_time'] = end_t - start_t
                # what is new_slice?
                #add to model list
                self.models[id]= str((Path('model') / (self.prefix + '_' + id + '.model')))

                ### TRAINING UPSTREAM MODELS
                # now get upstream and train
                upstream_paper_ids = get_connected_papers(self.graph, id, upstream=True)

                # limit to only papers in the subset corpus
                upstream_paper_ids = [paper for paper in upstream_paper_ids if paper in ids]

                if len(upstream_paper_ids) > 0:
                    deets['upstream_count'] = len(upstream_paper_ids)
                    upstream_slice = self.create_text_slice(upstream_paper_ids, self.prefix + "_" + id + "_UPSTREAMS.txt")

                    start_t = time.time()
                    new_slice = self.aligner.train_slice(str(upstream_slice), save=True,
                                                         saveName=(self.prefix + '_' + id + "_UPSTREAM"))
                    end_t = time.time()
                    deets['upstream_training_time'] = end_t - start_t
                    # add to model list
                    self.models[id+'_UPSTREAM'] = str((Path('model') / (self.prefix + '_' + id + '_UPSTREAM.model')))
                else:
                    deets['upstream_count'] = 0
                    print("no upstream papers for ", id)


                ### TRAINING DOWNSTREAM MODELS
                # now get downstream and train
                downstream_paper_ids = get_connected_papers(self.graph, id, upstream=False)
                # limit to only papers in the subset corpus
                downstream_paper_ids = [paper for paper in downstream_paper_ids if paper in ids]

                if len(downstream_paper_ids) > 0:
                    deets['downstream_count'] = len(downstream_paper_ids)
                    downstream_slice = self.create_text_slice(downstream_paper_ids, self.prefix + "_" + id + "_DOWNSTREAMS.txt")

                    start_t = time.time()
                    new_slice = self.aligner.train_slice(str(downstream_slice), save=True,
                                                         saveName=(self.prefix + '_' + id + "_DOWNSTREAM"))
                    end_t = time.time()
                    deets['downstream_training_time'] = end_t - start_t
                    # add to model list
                    self.models[id + '_DOWNSTREAM'] = str((Path('model') / (self.prefix + '_' + id + '_DOWNSTREAM.model')))
                else:
                    deets['downstream_count'] = 0
                    print("no downstream papers for ", id)

                # write out the log file for future use
                pickle.dump(deets, open(str(log_path), 'wb'))

            self.training_deets[id] = deets

    def cosine_similarity(self, word1, word2):
        cosine_similarity = numpy.dot(word1, word2) / (
                    numpy.linalg.norm(word1) * numpy.linalg.norm(word2))
        return cosine_similarity

    def get_model_handle(self, model_path):
        return Word2Vec.load(model_path)

    def get_wordvec_from_model(self, word, model_path):
        # this may eventually go somewhere else
        model1 = Word2Vec.load(model_path)
        return model1[word]

    def compare_word_to_compass(self, word, model_path):
        this_word = self.get_wordvec_from_model(word, model_path)
        compass_word = self.get_wordvec_from_model(word, self.compass_path)
        return self.cosine_similarity(this_word, compass_word)


    # EXPERIMENT 3 DATA FRAME GENERATION FUNCTION
    # get a list of dicts with the data for exp 3
    def get_network_comparison_dict(self, ids, title_words_nostop, nn_window = 5):
        all_data = []
        if not self.compass_path:
            print("train compass first")
            return

        if not self.training_deets:
            print("run the training function first - train_networked_corpus_embeddings")
            return

        if not self.paper_deets:
            print("provide paper deets (title, year, etc) before calling this function")
            return

        for id in ids:
            if self.training_deets[id]['upstream_count'] == 0 or self.training_deets[id]['downstream_count'] == 0:
                print ("no upstream or downstream for this paper")
                continue

            self_model = self.get_model_handle(self.models[id])
            up_model = self.get_model_handle(self.models[id+"_UPSTREAM"])
            down_model = self.get_model_handle(self.models[id+"_DOWNSTREAM"])

            # need the intersection vocab of all three models, and the title words
            valid_words = get_intersection_vocab([self_model, up_model, down_model], title_words_nostop)

            # for each word in the intersection:
            # is_title_word, citations, similarity with upstream, similarity with downstream, common words with upstream
            # common words with downstream, up/down similarity, num_papers_upstream, num_papers_downstream
            for word in valid_words:
                curr_dict = {}
                curr_dict['upstream_count'] = self.training_deets[id]['upstream_count']
                curr_dict['downstream_count'] = self.training_deets[id]['downstream_count']
                curr_dict['upstream_similarity'] = cosine_similarity(self_model[word], up_model[word])
                curr_dict['downstream_similarity'] = cosine_similarity(self_model[word], down_model[word])
                curr_dict['up_and_down_similairty'] = cosine_similarity(up_model[word], down_model[word])
                self_neighbors = self_model.wv.most_similar(positive=[word], topn=nn_window)
                down_neighbors = down_model.wv.most_similar(positive=[word], topn=nn_window)
                up_neighbors = up_model.wv.most_similar(positive=[word], topn=nn_window)
                curr_dict['neighborhood_size'] = nn_window
                curr_dict['upstream_shared_neighbors'] = sum([1 for i in up_neighbors if i in self_neighbors])
                curr_dict['downstream_shared_neighbors'] = sum([1 for i in down_neighbors if i in self_neighbors])
                curr_dict['up_and_down_shared_neighbors'] = sum([1 for i in up_neighbors if i in down_neighbors])

                # words that showed up in the paper and are downstream neighborhood
                curr_dict['downstream_influence'] = sum([1 for i in self_neighbors if i in down_neighbors and i not in up_neighbors])

                curr_dict['word'] = word

                curr_dict['is_title_word'] = word in self.paper_deets[id]['title_word_list']
                curr_dict['citations'] = self.paper_deets[id]['citations']
                all_data.append(curr_dict)
        return all_data

# path to stopwords
p_stop = Path('.') / 'STOPWORDS.txt'
stopwords = get_stop_words(p_stop)


## PATH TO PAPERS
p = Path('.') / '..' / 'project_code' / 'papers' / 'acl-arc-json' / 'json'


### Just exploring the network for now
# What are the different components?

network_path = 'network.txt'
# id_code_path = 'arc-paper-ids.tsv'
#
# read in the network
G = nx.read_edgelist(network_path, create_using=nx.DiGraph(), delimiter=' ', nodetype=str)
#

#### EXPERIMENT 2
#### SUBSAMPLING BASED ON HIGHLY CITED PAPERS

in_degs = list(G.in_degree())

in_degs_s = sorted(in_degs,key=itemgetter(1), reverse=True)

citation_counts = {i[0]: i[1] for i in in_degs_s}

# remove external
in_degs_ss = [i for i in in_degs_s if 'External' not in i[0]]

# ensure we have papers for all the nodes we want:
paper_list = get_list_of_papers(p)

# remove IDs we don't have papers for
in_degs_sss = [i for i in in_degs_ss if i[0] in paper_list]

# missing_papers = [i for i in in_degs_ss if i[0] not in paper_list]

# top paper
top_id = in_degs_sss[0][0]

# get network from top papers (upstream and downstream)
downstreams = get_connected_papers(G, top_id, upstream=False) # length of zero here means nobody cited it :(
upstreams = get_connected_papers(G, top_id, upstream=True)  # these are the papers the given paper cites
all_papers = downstreams.union(upstreams)
all_papers.add(top_id)

# ensure we have all the text for these papers
all_papers_wtext = [i for i in all_papers if i in paper_list]


# subset only to papers that have at least one citation
all_papers_wtxt_cite_5 = [i for i in all_papers_wtext if citation_counts[i] > 5]

# create the xml corpus for this group, send to Tim
#create_xml_corpus_reduced_words(all_papers_wtxt_cite_5, p, 'over5citations.xml')

all_papers_wtxt_cite_10 = [i for i in all_papers_wtext if citation_counts[i] > 20]
#create_xml_corpus_reduced_words(all_papers_wtxt_cite_10, p, 'over20citations.xml')

all_papers_wtxt_cite_25 = [i for i in all_papers_wtext if citation_counts[i] > 25]
#create_xml_corpus_reduced_words(all_papers_wtxt_cite_25, p, 'over25citations.xml')


subset = all_papers_wtxt_cite_25

### First step, gather the titles of the subset to make the title list
path_to_ids = Path('.') / '..' / 'project_code' / 'arc-paper-ids.tsv'

# will be a dict of dicts, where the key is the id of the paper, and the dict contains:
# year, title, authors are the keys of the inner dict
paper_deets = {}

with open(path_to_ids, 'r', encoding='utf-8') as infile:
    test = csv.DictReader(infile, delimiter='\t')
    for row in test:
        paper_deets[row['id']] = row
        paper_deets[row['id']]['title_word_list'] = row['title'].split()
        if row['id'] in citation_counts.keys():
            paper_deets[row['id']]['citations'] = citation_counts[row['id']]

# create list of words of the 100 paper titles in the subset, minus stopwords
title_words = []
for id in subset:
    title_words += paper_deets[id]['title'].split()

title_words_nostop = [word for word in title_words if word not in stopwords]

len(title_words_nostop)
# 615 words not stop words


### SET THIS FLAG TO ENABLE CHUNKS OF CODE
exp_1 = False
exp_2 = False
exp_3 = True

if exp_1:
    # we will do our first experiment on the papers in all_papers2
    # first, just the compass, and one for each paper in the corpus.
    # this will enable us to kick start our experiment coding infrastructure

    # TRAIN COMPASS on all_papers_wtext
    experiment1 = Experiment("exp12", exp_dir='exp2_dir')

    # train the compass
    experiment1.train_compass_from_id_list(subset)

    # train all the individual papers
    experiment1.train_single_paper_embeddings(subset)


    # get common words...
    all_models = {key: experiment1.get_model_handle(val) for key, val in experiment1.models.items()}
    shared_words = get_intersection_vocab(all_models.values())

    # get similarity statistics between all the paper embeddings and the compass embeddings
    words_to_check = ['number', 'result', 'mmi', 'unsmoothed', 'mwer', 'corpus', 'error', 'translation', 'smooth']

    c_handle = experiment1.get_model_handle(experiment1.compass_path)

    ### check to compare the network distance with the similarity scores...
    ### are papers close in the citation network close in their similarity scores for word embeddings?
    ## each data point will be a pair of papers
    ## and will include the similarity across some set of words, and a network geodesic distance
    ## let's get the geodesic distance first


    # pickle the shortest paths as this takes time
    if not Path('shortest_paths.p').is_file():
        G_undirected = G.to_undirected()

        all_paths = {}

        for mNode in subset:
            all_paths[mNode] = nx.shortest_path_length(G_undirected, source = mNode)

        pickle.dump(all_paths, open('shortest_paths.p', 'wb'))
        shortest_paths_dict = all_paths
    else:
        shortest_paths_dict = pickle.load(open('shortest_paths.p', 'rb'))

    # pickle the shortest paths as this takes time
    if not Path('shortest_paths_directed.p').is_file():
        all_paths = {}

        for mNode in subset:
            all_paths[mNode] = nx.shortest_path_length(G, source = mNode)

        pickle.dump(all_paths, open('shortest_paths_directed.p', 'wb'))
        shortest_paths_dict = all_paths
    else:
        shortest_paths_dict = pickle.load(open('shortest_paths_directed.p', 'rb'))


    # quick look at the vocab in the network of 100
    words_to_compare = get_union_vocab(all_models.values())


    if not Path('experiment1_geo_dist_directed_all_words.csv').is_file():
        # turn counter into a list. Just grab embeddings with a frequency of more than 5 for sanity.
        min_occur = 5
        new_list = [myWord for (myWord, myCount) in words_to_compare.items() if myCount >= min_occur]

        to_write = str(sum(words_to_compare.values())) + " tokens across a vocab size of " + str(len(words_to_compare)) + \
                    ", when you remove all words with less than " + str(min_occur) + " occurances, there are " + \
                    str(len(new_list))
        with open("embedding stats_experiment1.txt", 'w', encoding='utf8') as outfile:
            outfile.write(to_write)



        ### ACTUALLY RUN THE EXPERIMENT

        ## SET SOME TIMER
        start = time.time()
        exp2results = get_pairwise_model_comparisons(all_models, new_list, shortest_paths_dict,verbose=True)
        end = time.time()
        diff = end-start
        print ('it took ', diff, ' seconds to run the experiment')

        # convert into a csv to save it
        df = pd.DataFrame(exp2results)

        df.to_csv('experiment1_geo_dist_directed_all_words.csv', header=True,index=False)

    print("start graphing!")


    # treating every embedding comparison as its own data point... let's see what we find
    # this will be a massive set of data
    df = pd.read_csv('experiment1_geo_dist_directed_all_words.csv')

    # well, even simpler. Compare embeddings between those connected in the network, and those who aren't!

    # to do these tests, we need a dataset of similarity and distance... would be good to know the word too.. reshape time
    #stacked = df.melt(id_vars=['model1', 'model2', 'geodesic_dist'])


    if not Path('experiment1_geo_dist_directed_all_words.csv').is_file():
        ### AHH RUNNING OUT OF MEMORY! YIKES!!
        # going to reset--- read in the CSV with a dict reader and just write a new csv with the data points we want. cool?
        long_data = [] # list of dicts with soon to be melted data
        with open('experiment1_geo_dist_directed_all_words.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, val in row.items():
                    curr_dict = {}
                    if key == 'model1' or key == 'model2' or key == 'geodesic_dist' or val == '':
                        pass
                    else:
                        curr_dict['word'] = key
                        curr_dict['similarity'] = val
                        curr_dict['geo_dist'] = int(row['geodesic_dist'])
                        long_data.append(curr_dict)


        # now save the melted data to a dataframe
        melted_df = pd.DataFrame(long_data)
        melted_df.to_csv('experiment1_geo_dist_directed_all_words_melted.csv', header=True,index=False)

    melted_df = pd.read_csv('experiment1_geo_dist_directed_all_words_melted.csv')

    melted_df["similarity"] = pd.to_numeric(melted_df["similarity"])
    len(melted_df)


    # need to add attribute to the dataframe first
    melted_df['is_title'] = pd.Series(melted_df['word'].isin(title_words_nostop))

    melted_nostop_df = melted_df[~melted_df['word'].isin(stopwords)]

    len(melted_nostop_df) / len(melted_df)
    # 64% after removing stopwords

    disconnected_nostop_df = melted_nostop_df[melted_nostop_df['geo_dist'] == -1]
    connected_nostop_df = melted_nostop_df[melted_nostop_df['geo_dist'] != -1]


    from scipy.stats import pearsonr
    from scipy.stats import ttest_ind

    r, p = pearsonr(melted_df['geo_dist'], melted_df['similarity'])

    disconnected_df = melted_df[melted_df['geo_dist'] == -1]
    connected_df = melted_df[melted_df['geo_dist'] != -1]


    t, p_t = ttest_ind(connected_df['similarity'], disconnected_df['similarity'])

    # there is really no difference
    disconnected_df['similarity'].mean()
    # 0.6316924669779691

    connected_df['similarity'].mean()
    # 0.635573349578994

    # variance?
    disconnected_df['similarity'].var()
    connected_df['similarity'].var()


### SKIPPING GRAPHING FOR NOW. MAYBE I SHOULD THOUGH
### TODO - graph all the geodesic - similarity comparisons


    title_words_counter = Counter(title_words)

    title_words_set = set(title_words)

    # now let's do some analysis just with these words
    # use the same 1.7 million data frame
    title_df = melted_df[melted_df['word'].isin(title_words_set)]

    len(title_df) / len(melted_df)
    # 17.4 % of comparisons involve title words

    r, p = pearsonr(title_df['geo_dist'], title_df['similarity'])

    disconnected_title_df = title_df[title_df['geo_dist'] == -1]
    connected_title_df = title_df[title_df['geo_dist'] != -1]


    t, p_t = ttest_ind(connected_title_df['similarity'], disconnected_title_df['similarity'])

    # there is really no difference
    disconnected_title_df['similarity'].mean()
    # 0.6109308238724026

    connected_title_df['similarity'].mean()
    # 0.617403017484626

    # variance?
    disconnected_df['similarity'].var()
    connected_df['similarity'].var()



    # also filter to make sure there's enough frequency



    # now let's do some analysis just with these words - get rid of stopwords
    title_nostop_df = title_df[title_df['word'].isin(title_words_nostop)]

    len(title_nostop_df) / len(melted_df)
    # 17.4 % of comparisons involve title words
    # 10.5 % of comparisons involve title no stop

    r, p = pearsonr(title_nostop_df['geo_dist'], title_nostop_df['similarity'])

    # r = 0.005435610301003945
    # p = 0.020663807195867908

    disconnected_title_nostop_df = title_nostop_df[title_nostop_df['geo_dist'] == -1]
    connected_title_notstop_df = title_nostop_df[title_nostop_df['geo_dist'] != -1]


    t, p_t = ttest_ind(connected_title_notstop_df['similarity'], disconnected_title_nostop_df['similarity'])
    # t = 13.5349
    # p_t = 1.0204391544366794e-41


    # nearly a percent increase in similarity when connected
    disconnected_title_nostop_df['similarity'].mean()
    # 0.6109308238724026

    connected_title_notstop_df['similarity'].mean()
    # 0.617403017484626


    # variance?
    disconnected_title_nostop_df['similarity'].var()
    connected_title_notstop_df['similarity'].var()


    # let's try to plot this and see what we find using matplotlib
    import matplotlib.pyplot as plt
    import matplotlib

    # trying scatter
    plt.scatter(title_nostop_df['geo_dist'], title_nostop_df['similarity'], color='b', alpha=.005, marker=',')

    plt.xlabel("Geodesic Distance")
    plt.ylabel("Cosine Similarity")

    plt.title("Embedding Similarity vs. Geodesic (network) Distance")

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(20, 10)
    fig.savefig("Embedding similarity vs. geodesic distance.png", dpi=400)

    ## that's one chart, not super informative given how many data points there are.

    # now let's plot the averages and the confidence intervals for each average

    import seaborn as sns
    sns.set_theme(style="darkgrid")


    ax = sns.pointplot(x="geo_dist", y="similarity", data=connected_title_notstop_df)
    ax.set_title('Word Embedding Similarity vs. Geodesic Distance in Citation Network')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xlabel('Geodesic Distance')
    plt.savefig('test_seaborn_v1_title_words_only_nostop.png')


    ax = sns.pointplot(x="geo_dist", y="similarity", data=connected_df)
    ax.set_title('Word Embedding Similarity vs. Geodesic Distance in Citation Network')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xlabel('Geodesic Distance')
    plt.savefig('test_seaborn_all_words.png')


    ax = sns.pointplot(x="geo_dist", y="similarity", hue='is_title', data=connected_df)
    ax.set_title('Word Embedding Similarity vs. Geodesic Distance in Citation Network')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xlabel('Geodesic Distance')
    #ax.legend.set_title('Is Title Word')
    plt.legend(title="Is Title Word")
    plt.savefig('test_seaborn_all_vs_title.png')

    ### Plot the geodesic distance vs. cosine similarity for non-stopwords, title vs non-title word
    ax = sns.pointplot(x="geo_dist", y="similarity", hue='is_title', data=connected_nostop_df)
    ax.set_title('Word Embedding Similarity vs. Geodesic Distance in Citation Network')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xlabel('Geodesic Distance')
    #ax.legend.set_title('Is Title Word')
    plt.legend(title="Is Title Word")
    plt.savefig('test_seaborn_all_vs_title_no_stopwords.png')



    ax = sns.violinplot(x="geo_dist", y="similarity", data=connected_title_notstop_df)

    ax = sns.violinplot(x="geo_dist", y="similarity", data=connected_df)

    ax = sns.violinplot(x="geo_dist", y="similarity", hue='is_title', data=connected_df)

    plt.savefig('test_seaborn_v1.png')


#################################################
#### EXPERIMENT TWO #############################
#################################################
# first, let's figure out how to find nearest neighbors in the embedding model.

# start with compass


if exp_2:
    if not Path('experiment2_data_v2.p').is_file():
        # this will be a dict. Key is the paper in question
        # value is a dict of :
        # upstream_vocab - dict of words: upstream word list
        # uniquely_added_words - list of words
        # downstream_influence (number of times the uniquely_added_words show up downstream) - word: number unique hits
        experiment2_data = {}

        count = 0

        # for every paper in the subset
        for id, paper_model in all_models.items():
            count += 1
            print ('starting paper ', id, ", ",count, " out of ", len(subset))
            curr_dict = {}
            curr_dict['upstream_vocab'] = {}
            curr_dict['upstream_word_users'] = {}

            # get every upstream paper
            upstream_papers = get_connected_papers(G, id, upstream=True)
            for up_p in upstream_papers:
                # for every upstream paper... if it's in the subset...
                if up_p in subset:
                    # for every intersection word
                    for word in get_intersection_vocab([paper_model, all_models[up_p]]):
                        if word in title_words_nostop:
                            # only looking at title words
                            if word not in curr_dict['upstream_vocab'].keys():
                                curr_dict['upstream_vocab'][word] = Counter()  # init the empty list of upstream nn's for this word
                                curr_dict['upstream_word_users'][word] = 0
                            curr_nns = [nn for nn, sim in all_models[up_p].wv.most_similar(positive=[word], topn=5)]
                            curr_dict['upstream_vocab'][word].update(curr_nns)
                            curr_dict['upstream_word_users'][word] += 1
            experiment2_data[id] = curr_dict

        # create the set of newly added words to the list of nn's for each word in each paper
        count = 0
        for id, myDict in experiment2_data.items():
            count += 1
            print('starting paper ', id, ", ", count, " out of ", len(subset))
            experiment2_data[id]['uniquely_added_words'] = {}  # dict with key: word, val = list of unique nn's added by this p.
            for word in all_models[id].wv.vocab:
                if word in title_words_nostop:
                    if word in experiment2_data[id]['upstream_vocab'].keys():
                        experiment2_data[id]['uniquely_added_words'][word] = \
                            [nWord for nWord in all_models[id].wv.most_similar(positive=[word], topn=5) if nWord not in \
                             experiment2_data[id]['upstream_vocab'][word]]
                    else:
                        # title word is brand new! So the whole set becomes unique
                        # print (id, " first time seeing this word: ", word)
                        experiment2_data[id]['uniquely_added_words'][word] = \
                            [nWord for nWord, sim in all_models[id].wv.most_similar(positive=[word], topn=5)]


        # now look for downstream influence
        count = 0
        for id, paper_model in all_models.items():
            count += 1
            print ('starting paper ', id, ", ",count, " out of ", len(subset))
            curr_dict = {}
            count_dict = {}

            # get every downstream paper
            downstream_papers = get_connected_papers(G, id, upstream=False)
            for down_p in downstream_papers:
                # for every downstream paper... if it's in the subset...
                if down_p in subset:
                    # for every intersection word
                    for word in get_intersection_vocab([paper_model, all_models[down_p]]):
                        if word in title_words_nostop:
                            # only looking at title words
                            if word not in curr_dict.keys():
                                curr_dict[word] = Counter()  # init the empty list of upstream nn's for this word
                                count_dict[word] = 0
                            curr_nns = [nn for nn, sim in all_models[down_p].wv.most_similar(positive=[word], topn=5)]
                            curr_dict[word].update([w for w in curr_nns if w in \
                                                   experiment2_data[id]['uniquely_added_words'][word]])
                            count_dict[word] += 1
            experiment2_data[id]['downstream_influence'] = curr_dict
            experiment2_data[id]['downstream_word_users'] = count_dict

        # now pickle experiment 2 data for future analysis
        pickle.dump(experiment2_data, open("experiment2_data_v2.p", 'wb'))

    exp2_data = pickle.load(open("experiment2_data_v2.p", 'rb'))


    ## converting the complicated datastructure from above into a pandas dataframe
    if not Path('experiment2-word-influence-v3.csv').is_file():
        ## This will hold the list of dicts for the dataframe
        list_o_influences = []
        for id, data in exp2_data.items():

            for word in all_models[id].wv.vocab:
                if word in title_words_nostop:  # limit data to only title words....
                    curr_dict = {}
                    curr_dict['word'] = word
                    curr_dict['paper_id'] = id
                    curr_dict['citations'] = citation_counts[id]
                    curr_dict['is_title_word'] = word in paper_deets[id]['title_word_list']
                    if word in data['downstream_word_users'].keys():
                        curr_dict['num_downstream_users'] = data['downstream_word_users'][word]
                    else:
                        curr_dict['num_downstream_users'] = 0

                    if word in data['upstream_word_users'].keys():
                        curr_dict['num_upstream_users'] = data['upstream_word_users'][word]
                    else:
                        curr_dict['num_upstream_users'] = 0



                    if word in data['uniquely_added_words'].keys():
                        curr_dict['num_uniquely_added_words'] = len(data['uniquely_added_words'][word])
                    else:
                        curr_dict['num_uniquely_added_words'] = 0

                    if word in data['downstream_influence'].keys():
                        curr_dict['num_downstream_influences'] = sum(data['downstream_influence'][word].values())
                    else:
                        curr_dict['num_downstream_influences'] = 0

                    if word in data['upstream_vocab'].keys():
                        curr_dict['upstream_unique_words'] = data['upstream_vocab'][word].keys()
                        curr_dict['upstream_unique_word_count'] = len(data['upstream_vocab'][word])
                        curr_dict['percent_of_new'] = curr_dict['num_uniquely_added_words'] / curr_dict['upstream_unique_word_count']

                    list_o_influences.append(curr_dict)

        set_df = pd.DataFrame(list_o_influences)
        set_df.to_csv("experiment2-word-influence-v3.csv", header=True,index=False)

    set_df = pd.read_csv("experiment2-word-influence-v3.csv")



    import seaborn as sns
    ax = sns.histplot(data=set_df, x="num_downstream_influences", binwidth=1, hue="is_title_word", element="step")
    #ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)

    # data viz not really clear. Let's look for statistical changes
    set_title_df = set_df[set_df['is_title_word']]
    set_notitle_df = set_df[~set_df['is_title_word']]

    len(set_title_df)
    len(set_notitle_df)
    len(set_df)

    t, p_t = ttest_ind(set_title_df['num_downstream_influences'], set_notitle_df['num_downstream_influences'])
    t # 2.0769027829629643
    p_t  # 0.03784729859253939

    # there is really no difference
    set_title_df['num_downstream_influences'].mean()
    # 0.28475336322869954

    set_notitle_df['num_downstream_influences'].mean()
    # 0.12276372609500308

    set_title_df


## That's a great result.


######################################################################################################################
#### EXPERIMENT THREE ################################################################################################
######################################################################################################################

##  Now let's look at treating the papers like they are pivot points in the corpus.
## Where we have upstream and downstream papers each forming their own corpus.
## And we look at the similarities or number of mutual nearest neighbors before and after
## Hypothesis: papers with words in title will have more similarity with downstream corpus vs. upstream, mediated thru citations

# create the housing experiment object

if exp_3:

    if not Path('experiment3-networkedCorpus-v1.csv').is_file():
        # TRAIN COMPASS on all_papers_wtext
        experiment3 = Experiment("exp3", exp_dir='exp3_dir', g_in=G, p_deets_in=paper_deets)

        # train the compass
        experiment3.train_compass_from_id_list(subset)


        # train all the individual papers
        experiment3.train_networked_corpus_embeddings(subset)

        # now let's think about making a dataframe for this...
        # for each word in each paper:
        # is_title_word, citations, similarity with upstream, similarity with downstream, common words with upstream
        # common words with downstream, up/down similarity, num_papers_upstream, num_papers_downstream

        # get a list of dicts with the data above
        exp3data = experiment3.get_network_comparison_dict(subset, title_words_nostop)

        df_3 = pd.DataFrame(exp3data)

        df_3.to_csv("experiment3-networkedCorpus-v1.csv",header=True,index=False)

    df_n = pd.read_csv("experiment3-networkedCorpus-v1.csv")

    df_n['extra_similarity'] = df_n['downstream_similarity'] - df_n['up_and_down_similairty']

    # now onto the data analysis for experiment 3
    # eventually we could split the data generation from the data analysis
    # that will be left as an exercise for the future self
    df_n['downstream_shared_neighbors'].describe()

    import seaborn as sns

    sns.set_theme(style="darkgrid")

    sns.histplot(df_n, x="extra_similarity", hue='citations',element="step")

    ax = sns.pointplot(x="citations", y="downstream_similarity", hue='is_title_word', data=df_n) ## OKAY FOR PAPER
    ax = sns.pointplot(x="is_title_word", y="downstream_similarity", data=df_n)  ## GOOD ONE FOR PAPER!!!

    ax = sns.pointplot(x="is_title_word", y="extra_similarity", data=df_n)

    ax = sns.pointplot(x="is_title_word", y="up_and_down_similairty", data=df_n)

    sns.scatterplot(data=df_n, x="citations", y="downstream_similarity", hue="is_title_word")

    sns.scatterplot(data=df_n, x="citations", y="extra_similarity", hue="is_title_word")

    sns.relplot(x="downstream_count", y="downstream_similarity", kind="line", ci="sd", data=df_n)

    sns.scatterplot(data=df_n, x="citations", y="up_and_down_similairty", hue="is_title_word")

    sns.relplot(x="citations", y="extra_similarity", kind="line", ci="sd", data=df_n)

    ax = sns.pointplot(x="downstream_shared_neighbors", y="citations", data=df_n)
    ax.set_title('Word Embedding Similarity vs. Geodesic Distance in Citation Network')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xlabel('Geodesic Distance')
    plt.savefig('test_seaborn_v1p.png')


    df_title_words = df_n[df_n['is_title_word'] == True]
    df_non_title_words = df_n[df_n['is_title_word'] == False]

    from scipy.stats import pearsonr
    from scipy.stats import ttest_ind

    # Now trying some statistical tests
    t, p_t = ttest_ind(df_title_words['downstream_similarity'], df_non_title_words['downstream_similarity'])
    t  # 6.223668729598324
    p_t  # 5.291064610878137e-10

    # there is really no difference
    df_title_words['downstream_similarity'].mean()
    # 0.6371298224048715

    df_non_title_words['downstream_similarity'].mean()
    # 0.5653249790444562


    t, p_t = ttest_ind(df_title_words['up_and_down_similairty'], df_non_title_words['up_and_down_similairty'])
    t  # 5.770948951210787
    p_t  # 8.40265357707634e-09

    # there is really no difference
    df_title_words['up_and_down_similairty'].mean()
    # 0.6371298224048715

    df_non_title_words['up_and_down_similairty'].mean()
    # 0.5653249790444562



    ## Correlation checks
    r, p = pearsonr(df_n['citations'], df_n['up_and_down_similairty'])
    r  # -0.060540256961420735
    p  # 4.152068569050007e-05






exit()

## issue with the dataset - so many papers are part of small internal network... because we subsampled too heavily.
## oh well.. I could technically re-do everything with a larger corpus. I guess

















# time for data analysis on this new dataset
# first check the hypothesis. Plot the number of influences

# 2d histogram
sns.histplot(set_df, x="num_downstream_influences", y="citations")
ax.set_ylim(0, 100)



## need to generate dataframe for analysis based on this data.
## going for: paper_id, word, is_title_word, # downstream papers, # downstream influences,
## influences per downstream paper, # unique words, # upstream papers, percent unique contribution




# largely expecting that is_title_word means the average number of downstream influences is higher








# so how does this work...
# we need to come up with a temporal order



# going to randomly sample down to 5000 to see what happens

## too big, try reading from file

alt.data_transformers.enable('json')

# now let's see what's up. Let's do a double distribution plot of similarities.
# two dsns..
c1 = alt.Chart(melted_df).mark_area(
    opacity=0.5,
    interpolate='step'
).encode(
    alt.X('similarity:Q', bin=alt.Bin(maxbins=100)),
    alt.Y('count()', stack=None),
    alt.Color('geo_dist:N')
).properties(
    title='Overlapping Histograms from Tidy/Long Data'
)
altair_viewer.display(c1)



## PICK UP HERE


len(words_to_compare)

# See a distribution of counter values
source = pd.DataFrame({'word_freq': words_to_compare.values()})

base = alt.Chart(source)

bar = base.mark_bar().encode(
    x=alt.X('word_freq', bin=alt.Bin(step=2), axis=alt.Axis(title='Word Frequency')),
    y=alt.Y('count()', axis=alt.Axis(title='Count'))
    #    y='count()'
)

altair_viewer.display(bar)

bar.save('word_freq.json')
# save this image
bar.save('word_freq.png', scale_factor=5.0)
bar.save('word_freq.svg')

# rule = base.mark_rule(color='red').encode(
#     x='mean(word_freq)',
#     size=alt.value(5)
# )
#
# altair_viewer.display(bar + rule)

# load from csv
#['number', 'result', 'mmi', 'unsmoothed', 'mwer', 'corpus', 'error', 'translation', 'smooth']
df = pd.read_csv('experiment1_geo_dist_directed.csv')


# some basic plots





to_plot = df[df['geodesic_dist'] != -1]


line = alt.Chart(to_plot).mark_line().encode(
    x='geodesic_dist',
    y='mean(number)'
)

band = alt.Chart(to_plot).mark_errorband(extent='ci').encode(
    x='geodesic_dist',
    y=alt.Y('number', title='Similarity in "Error" Embedding'),
)

altair_viewer.display(band + line)





# plotting distributions to see what's up there
dist_chart_1 = alt.Chart(df).mark_bar().encode(
    alt.X('corpus', bin=True),
    y='count()',
)
altair_viewer.display(dist_chart_1)




chart1 = alt.Chart(to_plot).mark_point().encode(
    alt.X('geodesic_dist',
          scale=alt.Scale(domain=(0, 10))
    ),
    y='error'
).configure_mark(
    opacity=0.1
)

errorbars = chart1.mark_errorbar().encode(
    x="x",
    y="ymin:Q",
    y2="ymax:Q"
)

altair_viewer.display(chart1 + errorbars)

#OLS model to see if we find anything

import statsmodels.api as sm

y = df['geodesic_dist']
X = df['error']
X = sm.add_constant(X)
model11 = sm.OLS(y, X).fit()
model11.summary()



alt.Chart(source).mark_bar().encode(
    alt.X("IMDB_Rating:Q", bin=True),
    y='count()',
)

chart2 = alt.Chart(df).mark_point().encode(
    alt.X('corpus',
          scale=alt.Scale(domain=(0, 1))
    ),
    y='citations'
)


altair_viewer.display(chart1)
altair_viewer.display(chart2)



exit()






#OLS model to see if we find anything

import statsmodels.api as sm

y = df['citations']
X = df[shared_words]
X = sm.add_constant(X)
model11 = sm.OLS(y, X).fit()
model11.summary()



#### THIS IS JUST SOME BASIC COMPARISONS TO THE COMPASS... NOT VERY MEANINGFUL I GUESS


my_dicts = experiment1.get_df_for_compass_comparisons(all_models, c_handle, words_to_check)

my_dicts2 = experiment1.get_df_for_compass_comparisons(all_models, c_handle, shared_words)

# now do some analysis in pd

df = pd.DataFrame(my_dicts2)

# where is unsmoothed
df2 = pd.DataFrame(my_dicts)


chart1 = alt.Chart(df2).mark_point().encode(
    alt.X('error',
          scale=alt.Scale(domain=(0, 1))
    ),
    y='citations'
)

chart2 = alt.Chart(df).mark_point().encode(
    alt.X('corpus',
          scale=alt.Scale(domain=(0, 1))
    ),
    y='citations'
)


altair_viewer.display(chart1)
altair_viewer.display(chart2)






exit()

#test the cosine similarity

test_model = experiment1.get_model_handle(experiment1.models[subset[0]])
meep = test_model.wv.vocab
mine = test_model['corpus']

c_handle = experiment1.get_model_handle(experiment1.compass_path)

sim_dict_all = experiment1.get_similarity_dict(test_model, c_handle)

meepers2 = get_intersection_vocab([test_model, c_handle])

all_model_handles = [experiment1.get_model_handle(i) for i in experiment1.models.values()]

meepersAll = get_intersection_vocab(all_model_handles)

similarities = {}
for id, m in experiment1.models.items():
    curr_model = experiment1.get_model_handle(m)
    sim_dict_intersection = experiment1.get_similarity_dict(curr_model, c_handle, words=meepersAll)
    similarities[id] = get_average_similarity(sim_dict_intersection)


# combine with citation counts and make a list of dicts for turning into a pandas dataframe
for_df = []
for myId, sim in similarities.items():
    # make a list of dicts to convert to pandas dataframe
    new_item = {}
    new_item['id'] = myId
    new_item['avg_similarity_w_compass'] = sim
    new_item['citations'] = citation_counts[myId]
    for_df.append(new_item)


# covert to pandas dataframe
df = pd.DataFrame(for_df)

df.to_csv("experiment1data.csv", header=True,index=False)


df = pd.read_csv("experiment1data.csv")

chart = alt.Chart(df).mark_point().encode(
    alt.X('avg_similarity_w_compass',
          scale=alt.Scale(domain=(.96, .99))
    ),
    y='citations'
)

altair_viewer.display(chart)

# just start by making a dataframe with each paper and the cosine similarity with each word (compared to compass)
# are the word embeddings from the paper in question higher in similarity to the papers that came before it? Or after it?

# I can do model.wv.vocab['word']['count'] and that should give me the count (?)


# create csv file of similarities, to be read into a dataframe for viz.


test_compare = experiment1.compare_word_to_compass('corpus', experiment1.models[subset[0]])

# so I can quickly get a distribution of comparisons between two years.
# first, find the intersection of the vocab
# then do pairwise cosine similarity, store in a dict with word: similarity.


print("PAUSE")

# TRAIN MODEL on every single individual paper


# remove external
# all_internal = [i for i in list(all_papers) if 'External' not in i]

# UNCOMMENT TO CREATE CORPUS
#create_xml_corpus(all_papers2, p, 'largest_id_corpus_sketch_engine')

exit()
#### EXPERIMENT 1 ########
#### COMPARE VECTOR ALIGNMENT BETWEEN EACH PAPER AND THE LARGEST WEAKLY CONNECTED COMPONENT #####


### FIRST STEP: GET THE LARGEST COMPONENT OF THE NETWORK (~98.3% of the corpus)
largest = max(nx.weakly_connected_components(G), key=len)

### NEXT, COMBINE ALL THE PAPERS IN THE COMPONENT INTO A SINGLE TEXT FILE
### COMMENTED BECUASE IT TAKES SOME TIME AND WE ALREADY RAN IT
#start1 = time.time()
#create_text_slice(get_paths_from_strings(largest, p), 'LARGEST_WEAK_COMPONENT.txt')
#end1 = time.time()
#elapsed1 = end1 - start1
#print (elapsed1)

### TRAIN THE COMPASS WORD EMBEDDINGS FOR THIS COMPONENT #####
### TAKES A WHILE SO WE WILL TIME IT ####
aligner = TWEC(size=50, siter=10, diter=10, workers=4)
start2 = time.time()
aligner.train_compass('LARGEST_WEAK_COMPONENT.txt', overwrite=False, filename="LARGEST_WEAK_COMPONENT.model")
end2 = time.time()
elapsed2 = end2 - start2
print (elapsed2)


# write the training time to a file
with open("experiment_1_train_compass_time.log", 'w', encoding='utf-8') as outfile:
    outfile.write("time to train compass on largest weakly connected component was \n" + str(elapsed2) + " seconds")


id = 'J92-1001'
fileP = get_path_from_id(id, p)
slice_one = aligner.train_slice(str(fileP), save=True)

### NOW GO THROUGH AND TRAIN MODELS FOR EVERY INDIVIDUAL PAPER
### THIS WILL TAKE QUITE A WHILE BECAUSE THERE'S 80k+ papers

### UHHH I NEED 86 gb of free space to actually be able to save all these word embedding models
## I might have to run this on the free PC I got... that should have space... lol

for paper in largest:
    fileP = get_path_from_id(paper, p)
    slice_one = aligner.train_slice(str(fileP), save=True, saveName='EXP1-' + paper)

### END EXPERIMENT !








downstreams = get_connected_papers(G, id, upstream=False) # length of zero here means nobody cited it :(
upstreams = get_connected_papers(G, id, upstream=True)  # these are the papers the given paper cites
all = downstreams.union(upstreams)
all.add(id)

# f_all = create_text_slice(get_paths_from_strings(all, p), 'J92-1001ALL.txt')
# f_up = create_text_slice(get_paths_from_strings(upstreams, p), 'J92-1001up.txt')
# f_down = create_text_slice(get_paths_from_strings(downstreams, p), 'J92-1001down.txt')
# f_self = create_text_slice(get_paths_from_strings([id], p), 'J92-1001self.txt')
#
#
# # decleare a TWEC object, siter is the number of iterations of the compass, diter is the number of iterations of each slice
# aligner = TWEC(size=50, siter=10, diter=10, workers=4)
#
# start = time.time()
#
# # train the compass (we are now learning the matrix that will be used to freeze the specific CBOW slices)
# # this may need some path modification to enable experimentation
# aligner.train_compass(str(f_down), overwrite=True)
#
# end =  time.time()
#
# elapsed = end-start
# print (elapsed)

#### TRAINING COMPASS FOR LARGEST COMPONENT ###

largest = max(nx.strongly_connected_components(G), key=len)



#
# upstreams = get_connected_papers(G, 'J92-1001', upstream=True)


### Testing out running the embeddings on the whole corpus to get an idea of time
p = Path('.')
p = p / '..' / 'project_code' / 'papers'
p = p / 'acl-arc-json' / 'json'


all_nodes = list(G.nodes)

papers = []
for node in all_nodes:
    temp = get_path_from_id(node, p)
    if temp:
        papers.append(temp)


create_text_slice(papers, 'every_paper_text.txt')

# decleare a TWEC object, siter is the number of iterations of the compass, diter is the number of iterations of each slice
aligner = TWEC(size=50, siter=10, diter=10, workers=4)

start = time.time()

# train the compass (we are now learning the matrix that will be used to freeze the specific CBOW slices)
# this may need some path modification to enable experimentation
aligner.train_compass('every_paper_text.txt', overwrite=True)

end =  time.time()

elapsed = start-end
print (elapsed)
# now see how long it takes to train the compass on that...


#len(list(G.successors('External_89521')))

exit()


G.number_of_edges()
G.number_of_nodes()

in_degs = list(G.in_degree())

in_degs_s = sorted(in_degs,key=itemgetter(1), reverse=True)

t = G.nodes['P13-1037']

testing =sorted(G.successors('P13-1037'))

all_nodes = list(G.nodes)

papers = []
for node in all_nodes:
    papers.append(get_path_from_id(node, ))

test = nx.weakly_connected_components(G)

nx.is_strongly_connected(G)

test1 = sorted(test)
y = [
    len(c)
    for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)
]
largest = max(nx.strongly_connected_components(G), key=len)
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








exit()

# example below



p = Path('.')
p = p / '..' / 'project_code' / 'papers'
p = p / 'acl-arc-json' / 'json'

experiment_dir = Path('.') / 'text_files'

test = get_path_from_id('J11-1005', p)
test1 = test
test2 = test

myList = [test, test1, test2]

slice1 = create_text_slice(myList, experiment_dir / 'testout.txt')


test = get_path_from_id('J11-1006', p)
test1 = test
test2 = test

myList = [test, test1, test2]

slice2= create_text_slice(myList, experiment_dir / 'testout2.txt')


slices = [slice1, slice2]

compass = create_text_slice(slices, experiment_dir / 'testCompass.txt')


# decleare a TWEC object, siter is the number of iterations of the compass, diter is the number of iterations of each slice
aligner = TWEC(size=50, siter=10, diter=10, workers=4)

# train the compass (we are now learning the matrix that will be used to freeze the specific CBOW slices)
# this may need some path modification to enable experimentation
aligner.train_compass(str(compass), overwrite=True)

# now you can train slices and they will be already aligned (we use two collections of arxiv papers of different years)
# these two objects are gensim objects
slice_one = aligner.train_slice(str(slice1), save=True)
slice_two = aligner.train_slice(str(slice2), save=True)

#once trained you can also load the trained and aligned embeddings
model1 = Word2Vec.load("model/testout.model")
model2 = Word2Vec.load("model/testout2.model")


model1['algorithm']
model2['algorithm']












### OLD JUNK



all_papers = set()

# recursive function that adds papers to the global variable.
def get_downstream_papers(G, node_id):
    global all_papers

    # get all the papers this one cites
    node_successors = list(G.successors(node_id))

    # and remove any elements that are already captured in the global list
    node_successors_real = [node for node in node_successors if node not in all_papers]

    # end condition - no new nodes to add to the global list
    if len(node_successors_real) == 0:
        return None
    else:
        for node in node_successors_real:
            all_papers.add(node)
            get_downstream_papers(G, node)




# SHORTEST PATH JUNK

    shortest_paths = nx.shortest_path_length(G_undirected, source='P13-1037')

    shortest_paths['P13-1037']

    # go through shortest path generator and only keep distances between internal papers (not starting with external)
    dict_o_lengths = {}  # will hold everything
    i = 0
    for (myKey, myDict) in shortest_paths:
        print(i)
        i += 1
        curr_dict = {}  # holds the key pairs for a single paper
        if not str.startswith(myKey, 'External'):
            for key1, dist in myDict.items():
                if not str.startswith(key1, 'External'):
                    curr_dict[key1] = dist
            dict_o_lengths[myKey] = curr_dict

    s2 = nx.shortest_path_length(G_undirected)
    s2dict = dict(s2)

    shortest_paths_dict = dict(shortest_paths)