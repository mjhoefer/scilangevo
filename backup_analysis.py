












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