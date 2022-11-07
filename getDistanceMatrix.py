from matplotlib.pylab import ndarray
import pandas as pd
import numpy as np

# from ripser import ripser
# from persim import plot_diagrams

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy import sparse
import scipy.spatial as spatial

# from ripser import ripser
# from persim import plot_diagrams
# import tadasets

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
# import gudhi

import random

# import stablerank.srank as sr




dist = open('PoliticalData/mpdistrict.dat').readlines()
party = open('PoliticalData/mpparty.dat').readlines()
party = party[3:]
sex = open('PoliticalData/mpsex.dat').readlines()
sex = sex[2:]
votes = open('PoliticalData/votes.dat').readline()

dist = list(map(lambda x:x.strip(),dist))
party = list(map(lambda x:x.strip(),party))
sex = list(map(lambda x:x.strip(),sex))

df = pd.DataFrame()
df['District'] = dist
df['Party'] = party
df['Gender'] = sex


votesnp = np.array(list(votes.split(',')))
votenp = np.array_split(votesnp, 31)

dfvote = pd.DataFrame()
i = 0
for vote in votenp:
    df['Vote: ',str(i)] = vote.tolist()
    i = i+1
# print(df)

# print(votespd)

# arr = df.sort_values(by="Party", ascending=7) # Ordering by Party
# print(arr)
arr = df.to_numpy() # change from dataframe to array
arr = arr.astype(np.float64)
arrVote = arr[0:,3:] # dropping District, Party,Sex


dend = hierarchy.linkage(arrVote, 'ward') # Creating a Dendrogram
plt.figure()
dn = hierarchy.dendrogram(dend, labels=list(df['Gender']))
# plt.show()
# Future inmplementation: colorgrade every party

# Create distance-matrix (euclidean, manhattan, hamming)
distmat = spatial.distance_matrix(arrVote,arrVote)
distman = manhattan_distances(arrVote,arrVote)
print(distman)
que = (arrVote[:, None, :] != arrVote).sum(2)


# Sample from the distance matrix
samp = random.sample(range(len(distmat)),60)
distmat_red_Ham = np.zeros((len(samp),len(samp)))
k = 0
n = 0
for i in samp:
    for j in samp:
        distmat_red_Ham[k,n] = que[i,j]
        n = n + 1
    k = k + 1
    n = 0
np.savetxt('distHam', distmat_red_Ham, delimiter=' ')

distmat_red = np.zeros((len(samp),len(samp)))
k = 0
n = 0
for i in samp:
    for j in samp:
        distmat_red[k,n] = distman[i,j]
        n = n + 1
    k = k + 1
    n = 0
np.savetxt('distMan', distmat_red, delimiter=' ')

distmat_red = np.zeros((len(samp),len(samp)))
k = 0
n = 0
for i in samp:
    for j in samp:
        distmat_red[k,n] = distmat[i,j]
        n = n + 1
    k = k + 1
    n = 0
np.savetxt('distEuc', distmat_red, delimiter=' ')




# print(arrVote[0])
# print(spatial.distance.pdist([arrVote[0],arrVote[1]], "euclidean"))
# data_dis = [sr.Distance(spatial.distance.pdist(fig, "euclidean")) for fig in distmat]
data_dist = [str.Distance(fig) for fig in distmat]

# Converitng the distance objects into H0 stable ranks
clustering_methods = ["single", "complete", "average", "ward"]
data_h0sr = {}
train_h0sr = {}
for cm in clustering_methods:
    print(data_dist[0].get_h0sr(clustering_method=cm))
    data_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in data_dist]


# H0 homology
plt.figure(figsize=(10,7))
i = 0
for f in data_h0sr["single"]:
    if i <100:
        color = "red"
    else:
        color = "blue"
    f.plot(color=color, linewidth=0.5)
    i += 1




# # Random ripser plot
# diagrams = ripser(arr, thresh=5)['dgms']
# plot_diagrams(diagrams, show=True)



