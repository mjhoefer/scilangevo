from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec
from pathlib import Path
import json
import string
import networkx as nx
from operator import itemgetter

# combines all the papers in the papers list into a single text file
# papers is a list of pathlib objects
# writes to outfile_name
# concats all the text files in papers into the outfile_name
# returns the outfile_name as a path object for convenience
def create_text_slice(papers, outfile_name):
    with open(outfile_name, 'w', encoding='utf-8') as out:
        for paper in papers:
            out.write(paper.read_text(encoding='utf-8'))
    return Path(outfile_name)


# returns a pathlib object from an id string
def get_path_from_id(id, base_dir, extension='.words'):
    # need first char and the last two nums to get it all.
    first_char = id[0]
    two_nums = id[1:3]

    paper_dir = base_dir / first_char / (first_char + two_nums)

    # handle to particular file
    paper = paper_dir / (id + extension)

    if paper.is_file():
        return paper
    else:
        return None

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

### Just exploring the network for now
# What are the different components?

network_path = 'network.txt'
id_code_path = 'arc-paper-ids.tsv'

# read in the network
G = nx.read_edgelist(network_path, create_using=nx.DiGraph(), delimiter=' ', nodetype=str)

downstreams = get_connected_papers(G, 'P13-1037', upstream=False) # length of zero here means nobody cited it :(
upstreams = get_connected_papers(G, 'P13-1037', upstream=True)  # these are the papers the given paper cites

upstreams = get_connected_papers(G, 'J92-1001', upstream=True)


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
