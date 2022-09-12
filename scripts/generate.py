import numpy as np
import pandas as pd
from itertools import islice
from random import randint

# Generate slices of random data
def random_chunk(li, min_chunk=8000, max_chunk=15000):
    it = iter(li)
    while True:
        nxt = list(islice(it,randint(min_chunk,max_chunk)))
        if nxt:
            yield nxt
        else:
            break
    return nxt

train = pd.read_csv("../data/raw/creditcard.csv")
rand_indexes = np.arange(len(train))
np.random.shuffle(rand_indexes)
parts = list(random_chunk(rand_indexes))

for i in range(len(parts)):
    if i > 13:
        break
    train_random = train.iloc[parts[i]]
    train_random.to_csv('../data/processed/train_data_' + str(i) + '.csv',index=False)
    print(train_random.shape)