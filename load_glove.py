import os
import numpy as np
from os.path import expanduser
import pickle

words = []
idx = 0
word2idx = {}
vectors = []
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

print(os.path.exists('~/Glove/glove.6B.300d.txt'))
print(os.getcwd())
print(expanduser("~"))
with open(expanduser("~")  + '/Glove/glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        # print(vect)
        vectors.append(vect)

glove = {w: vectors[word2idx[w]] for w in words}
# print(glove)
# print(glove['move'])
# print(glove['move'].shape)

file = open(expanduser("~")  + '/Glove/glove_dict_300.pkl', 'wb') 

pickle.dump(glove, file)

file.close()