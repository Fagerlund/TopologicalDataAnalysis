import pandas as pd
import numpy as np

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import scipy.spatial as spatial
import stablerank.srank as sr
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
print(df)

# print(votespd)

#arr = df.sort_values(by="Party", ascending=7) # Ordering by Party
arr = df.to_numpy() # change from dataframe to array



array_only_votes = arr[0:,3:] # dropping District, Party,Sex
dend = hierarchy.linkage(array_only_votes, 'ward')#,color=np.array(df["Party"].values)) # Creating a Dendrogram
plt.figure()
dist=dend

dn = hierarchy.dendrogram(dend,labels=list(df["Party"].values))
label_colors = {'0': 'r', '1': 'b', '2': 'g', '3': 'b','4': 'm', '5': 'c', '6': 'w', '7': 'y'}
ax = plt.gca()
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    lbl.set_color(label_colors[lbl.get_text()])

plt.show()

#for j in range(len()):
    

dn1 = hierarchy.dendrogram(dend)#,labels=list(df["Party"].values))

#indx=dn1["ivl"]
#for j in range(len(dn['leaves_color_list'])):
#    indx=int(dn["ivl"][j])
#    party=(df.iloc[indx]["Party"])
#    dn1['leaves_color_list'][j]="C"+party

data=dist
#%%
import gudhi as gd  
array_only_votes=array_only_votes.astype(np.float)
skeleton_protein0 = gd.RipsComplex(
    distance_matrix = array_only_votes, 
    max_edge_length = 0.8
) 

Rips_simplex_tree_protein0 = skeleton_protein0.create_simplex_tree(max_dimension = 2)
BarCodes_Rips0 = Rips_simplex_tree_protein0.persistence()
Rips_simplex_tree_protein0.persistence_intervals_in_dimension(0)

#
#rips_complex = gudhi.RipsComplex(points=array_only_votes, max_edge_length=9)
#simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
#diag = simplex_tree.persistence(min_persistence=0.4)##

#gudhi.plot_persistence_barcode(diag)#
#plt.show()


#%%

temp_df=pd.DataFrame({"party":dn["ivl"],"color":dn["leaves_color_list"]})

for i in(range(7)):
    colorstring="C"+str(i+1)
    temp_df2=temp_df.loc[temp_df['color'] == colorstring]
    print(dict(temp_df2["party"].value_counts()))
    
print(df["Party"].value_counts())

#%%
frac=0.9
fracparties=[]
#for party in(range(df['Party'].nunique())):
#    fracparties[party]=round(party*frac)


partiesdf=[]
for party in(range(df['Party'].nunique())):
    arr = df[df.Party == str(int(party))].to_numpy()
    array_only_votes = arr[0:,3:] # dropping District, Party,Sex
    partiesdf.append(array_only_votes)
    
    
#%%



# Converitng the data into distance objects
array_only_votes=array_only_votes.astype(np.float)
data_dist = [sr.Distance(spatial.distance.pdist(array_only_votes, "euclidean"))]
#train_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean")) for fig in train]
# Converitng the distance objects into H0 stable ranks
clustering_methods = ["single", "complete", "average", "ward"]
data_h0sr = {}
train_h0sr = {}
for cm in clustering_methods:
    data_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in data_dist]
    #train_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in train_dist]
    



#%%

#%% # this section we analyse the differnet blocks.
df.drop(df[df['Party'] == "0"].index, inplace = True)
#Right block formation If we want to analyse them combined
#df.drop(df[df['Party'] == "3"].index, inplace = True)
#df.drop(df[df['Party'] == "4"].index, inplace = True)
#df.drop(df[df['Party'] == "5"].index, inplace = True)
#Left block formation If we want to analyse them combined
#df.drop(df[df['Party'] == "1"].index, inplace = True)
#df.drop(df[df['Party'] == "2"].index, inplace = True)
#df.drop(df[df['Party'] == "6"].index, inplace = True)
#df.drop(df[df['Party'] == "7"].index, inplace = True)

Stables=[0,0,0,0,0,0,0]
colors=["blue","cyan","red","darkred","green","midnightblue","springgreen"]
plt.figure(figsize=(10,7))
for i in(range(1000)):
    partiesdf=[]
    for party in(df["Party"].unique()):
        arr = df[df.Party == str(int(party))]
        try:
            arr = arr.sample(n=10, random_state=i)
        except:
            arr=arr
        arr=arr.to_numpy()
        array_only_votes = arr[0:,3:] # dropping District, Party,Sex
        
        partiesdf.append(array_only_votes.astype(np.float))
    # Converitng the data into distance objects
    array_only_votes=array_only_votes.astype(np.float)
    data_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean"))for fig in partiesdf]
    #train_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean")) for fig in train]
    # Converitng the distance objects into H0 stable ranks
    clustering_methods = ["single", "complete", "average", "ward"]
    data_h0sr = {}
    train_h0sr = {}
    for cm in clustering_methods:
        data_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in data_dist]
        #train_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in train_dist]
    
    #array_only_votes = arr[0:,3:] # dropping District, Party,Sex
    
    

    j = 0
    for f in data_h0sr["ward"]:
        Stables[j]=Stables[j]+f
        #ax.plot(f)
        color=colors[j]
        #print(f)
        #f.plot(linewidth=1,label='party 1',color=color)
        j=j+1
    #plt.title("")
    

l=0
for val in Stables:
    color=colors[l]
    val.plot(linewidth=1,label='party 1',color=color)
    l=l+1
    plt.legend(['M', "FP","S","V","MP","KD","C"])
#%% # this section we analyse the differnet blocks.
#Right block formation
df['Party'] = np.where(df['Party'] == "2", "1", df['Party'])
df['Party'] = np.where(df['Party'] == "6", "1", df['Party'])
df['Party'] = np.where(df['Party'] == "7", "1", df['Party'])
#Left block formation
df['Party'] = np.where(df['Party'] == "4", "3", df['Party'])
df['Party'] = np.where(df['Party'] == "5", "3", df['Party'])
#Dropping the value with non-party member
df = df[df.Party != "0"]
plt.figure(figsize=(10,7))
for i in(range(100)):
    partiesdf=[]
    for party in(df["Party"].unique()):
        arr = df[df.Party == str(int(party))]
        try:
            arr = arr.sample(n=120, random_state=i)
        except:
            arr=arr
        arr=arr.to_numpy()
        array_only_votes = arr[0:,3:] # dropping District, Party,Sex
        
        partiesdf.append(array_only_votes.astype(np.float))
    # Converitng the data into distance objects
    array_only_votes=array_only_votes.astype(np.float)
    data_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean"))for fig in partiesdf]
    #train_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean")) for fig in train]
    # Converitng the distance objects into H0 stable ranks
    clustering_methods = ["single", "complete", "average", "ward"]
    data_h0sr = {}
    train_h0sr = {}
    for cm in clustering_methods:
        data_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in data_dist]
        #train_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in train_dist]
    
    #array_only_votes = arr[0:,3:] # dropping District, Party,Sex
    
    

    j = 0
    for f in data_h0sr["ward"]:
        #ax.plot(f)
        if j%2==0:
            color="Blue"
        else:
            color="Red"
        f.plot(linewidth=1,label='party 1',color=color)
        j=j+1
    plt.title("")
    plt.legend(["Right-leaning parties","Left-leaning parties"])
#%%

# Converitng the data into distance objects
array_only_votes=array_only_votes.astype(np.float)
data_dist = [sr.Distance(spatial.distance.pdist(array_only_votes, "euclidean"))]
#train_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean")) for fig in train]
# Converitng the distance objects into H0 stable ranks
clustering_methods = ["single", "complete", "average", "ward"]
data_h0sr = {}
train_h0sr = {}
for cm in clustering_methods:
    data_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in data_dist]
    #train_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in train_dist]

    
#%%
Iterations=[]
for i in(range(100)):
    partiesdf=[]
    for party in(range(df['Party'].nunique())):
        arr = df[df.Party == str(int(party))]
        try:
            arr = arr.sample(n=100, random_state=i)
        except:
            arr=arr
        arr=arr.to_numpy()
        array_only_votes = arr[0:,3:] # dropping District, Party,Sex
        
        partiesdf.append(array_only_votes.astype(np.float))
    data_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean"))for fig in partiesdf]
    Iterations.append(data_dist)

clustering_methods = ["single", "complete", "average", "ward"]
data_h0sr = {}
train_h0sr = {}
New_iter=[]
for val in Iterations:
    temp=[]
    for cm in clustering_methods:


        data_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in Iterations]
        
    
plt.figure(figsize=(10,7))
i = 0
for iteration in Iterations:
    for f in iteration["single"]:
        #ax.plot(f)
        f.plot(linewidth=1)
        
    plt.legend(["No political affiliation",'M', "FP","S","V","MP","KD","C"])


#%%

partiesdf=[]
for party in(range(df['Party'].nunique())):
    arr = df[df.Party == str(int(party))]
    try:
        arr = arr.sample(n=17, random_state=1)
    except:
        arr=arr
    arr=arr.to_numpy()
    array_only_votes = arr[0:,3:] # dropping District, Party,Sex
    
    partiesdf.append(array_only_votes.astype(np.float))
# Converitng the data into distance objects
array_only_votes=array_only_votes.astype(np.float)
data_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean"))for fig in partiesdf]
#train_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean")) for fig in train]
# Converitng the distance objects into H0 stable ranks
clustering_methods = ["single", "complete", "average", "ward"]
data_h0sr = {}
train_h0sr = {}
for cm in clustering_methods:
    data_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in data_dist]
    #train_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in train_dist]
    
#%%
plt.figure(figsize=(10,7))
i = 0
for f in data_h0sr["single"]:
    #ax.plot(f)
    f.plot(linewidth=1,label='party 1')
    
plt.legend(["No political affiliation",'M', "FP","S","V","MP","KD","C"])
    
#%%
    
plt.figure(figsize=(10,7))
i = 0
for f in data_h0sr["complete"]:
    if i <100:
        color = "red"
    else:
        color = "blue"
    f.plot(color=color, linewidth=0.5)
    i += 1

#%%    
    
plt.figure(figsize=(10,7))
i = 0
for f in data_h0sr["average"]:
    if i <100:
        color = "red"
    else:
        color = "blue"
    f.plot(color=color, linewidth=0.5)
    i += 1
#%%    
    
plt.figure(figsize=(10,7))
i = 0
for f in data_h0sr["ward"]:
    if i <100:
        color = "red"
    else:
        color = "blue"
    f.plot(color=color, linewidth=0.5)
    i += 1
    
    
    
    
    
    
#%%

import scipy.stats as st

def circle(c, r, s, error=0):
    t = np.random.uniform(high=2 * np.pi, size=s)
    y = np.sin(t) * r + c[1]
    x = np.cos(t) * r + c[0]
    sd = error * 0.635
    pdf = st.norm(loc=[0, 0], scale=(sd, sd))
    return pdf.rvs((s, 2)) + np.vstack([x, y]).transpose()



data = []
i = 0
while i < 100:
    c = circle([0,0], 1, 100, error=0.2)
    data.append(c)
    i += 1  
    
    
    
    
    
    
#%%

#Woijtchechs assignemeent
Hamming_distance=(array_only_votes[:, None, :] != array_only_votes).sum(2)

#Here we elect which person to choose,
dmin = 17
dmax = 22
Hamming_distance[Hamming_distance<dmin]=0
Hamming_distance[Hamming_distance>dmax]=0

Index_MP=4
import random
for i in(range(50)):# Iterations
    indexes=np.nonzero(Hamming_distance[Index_MP]) # SElect which MP to sample with.
    candidates=random.sample(list(indexes[0]), 50)
    candidates.append(Index_MP)
    temp=array_only_votes[candidates]
    #hamming_dist_sample=(temp[:, None, :] != temp).sum(2)
    
# Converitng the data into distance objects
    #hamming_dist_sample=hamming_dist_sample.astype(np.float)
    data_dist = [sr.Distance(spatial.distance.pdist(temp, "hamming"))]#euclidean
    #train_dist = [sr.Distance(spatial.distance.pdist(fig, "euclidean")) for fig in train]
    # Converitng the distance objects into H0 stable ranks
    clustering_methods = ["single", "complete", "average", "ward"]
    data_h0sr = {}
    train_h0sr = {}
    for cm in clustering_methods:
        data_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in data_dist]
        #train_h0sr[cm] = [d.get_h0sr(clustering_method=cm) for d in train_dist]
    
    #array_only_votes = arr[0:,3:] # dropping District, Party,Sex
    
    
    

    for f in data_h0sr["ward"]:
        #ax.plot(f)

        color="Red"
        f.plot(linewidth=1,label='party 1',color=color)

    plt.title("")
    
    


