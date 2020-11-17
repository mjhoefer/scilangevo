from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec

# decleare a TWEC object, siter is the number of iterations of the compass, diter is the number of iterations of each slice
aligner = TWEC(size=50, siter=10, diter=10, workers=4)

path = 'C:\\Users\\mjhoe\\classes\\compsem\\twec\\twec\\examples\\training'
path_c = path + "\\compass.txt"
path_1 = path + "\\arxiv_14.txt"
path_2 = path + "\\arxiv_9.txt"

# train the compass (we are now learning the matrix that will be used to freeze the specific CBOW slices)
aligner.train_compass(path_c, overwrite=True)

# now you can train slices and they will be already aligned (we use two collections of arxiv papers of different years)
# these two objects are gensim objects
slice_one = aligner.train_slice(path_1, save=False)
slice_two = aligner.train_slice(path_2, save=False)

#once trained you can also load the trained and aligned embeddings
model1 = Word2Vec.load("model/arxiv_14.model")
model2 = Word2Vec.load("model/arxiv_9.model")

model1['algorithm']



