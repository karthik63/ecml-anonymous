#DeltaCon: proposed in A Principled Massive-Graph Similarity Function
#node index start from 1
from __future__ import division
import pandas as pd
import numpy as np
import random
import time

from scipy.sparse import dok_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse import diags
from scipy import sparse
from numpy.linalg import inv
from numpy import concatenate
from numpy import square
from numpy import array
from numpy import trace
from numpy import amax
from math import sqrt
from tqdm import tqdm
import networkx as nx


# def GenAdjacentMatrix(feat_Mat):
# 	'''
# 	Get adjacent matrix from file
# 	'''
# 	# data=pd.read_table(f, delimiter=' ', header=None)
# 	# i=data[0]
# 	# j=data[1]
# 	# size=max([max(i), max(j)])
# 	# adjacent=dok_matrix((size, size), dtype=np.int)
# 	# for k in range(len(i)):
# 	# 	adjacent[i[k]-1, j[k]-1]=1
# 	# 	adjacent[j[k]-1, i[k]-1]=1
# 	# for k in range(size):
# 	# 	adjacent[k, k]=0
# 	n_nodes = feat_Mat.shape[0]
# 	print(n_nodes)
# 	#FB
# 	# avgdeg=n_nodes
# 	#WN
# 	# avgdeg=4
# 	#yg
# 	# avgdeg=14
# 	avgdeg = 25
# 	# avg_deg = n_nodes * 0.001
#
# 	n_edges = n_nodes * 100
#
# 	a_square = np.sum(feat_Mat * feat_Mat, 1, keepdims=True)
# 	b_square = np.sum(feat_Mat * feat_Mat, 1, keepdims=True).T
# 	two_a_b = 2*feat_Mat @ (feat_Mat.transpose())
#
# 	mat = a_square + b_square - two_a_b
#
# 	vals = np.reshape(mat, (-1))
#
# 	print(np.max(vals))
# 	print(np.min(vals))
#
# 	vals_partitioned = np.partition(vals, n_edges)
# 	partition_value = vals_partitioned[n_edges]
# 	print(vals_partitioned[n_edges-1])
# 	print(vals_partitioned[n_edges])
# 	print(vals_partitioned[n_edges+1])
#
# 	condition = mat < partition_value
#
# 	mat = np.where(condition, np.ones_like(mat, dtype=np.int8), np.zeros_like(mat, dtype=np.int8)).astype(np.int8)
#
# 	print(np.sum(mat))
#
# 	# G = nx.from_numpy_array(mat)
# 	# edges=sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True)
# 	# edges=edges[:avgdeg*n_nodes]
# 	# H=nx.Graph()
# 	# H.add_edges_from(edges)
# 	# adjacent = nx.to_scipy_sparse_matrix(H, format='dok')
# 	sparse_mat = sparse.dok_matrix(mat)
# 	print('done')
# 	return mat, sparse_mat

def GenAdjacentMatrix(feat_Mat, avgdeg):
	'''
	Get adjacent matrix from file
	'''

	# feat_Mat = feat_Mat.astype(np.float16)

	n_nodes = feat_Mat.shape[0]
	print(n_nodes)
	#FB
	# avgdeg=n_nodes
	#WN
	# avgdeg=4
	#yg
	# avgdeg=14
	# avgdeg = 5
	# avg_deg = n_nodes * 0.001

	n_edges = int(n_nodes * avgdeg / 2)
	# avgdeg = int(n_nodes * 0.01)

	a_square = np.sum(feat_Mat * feat_Mat, 1, keepdims=True)
	b_square = np.sum(feat_Mat * feat_Mat, 1, keepdims=True).T
	two_a_b = 2*feat_Mat @ (feat_Mat.transpose())

	print(1)

	mat = a_square + b_square - two_a_b

	print(2)
	vals_partitioned = np.partition(mat, avgdeg, axis=1)
	partition_values = np.expand_dims(vals_partitioned[:, avgdeg], 1)

	print(3)
	condition = mat < partition_values

	mat = np.where(condition, np.ones_like(mat, dtype=np.int8), np.zeros_like(mat, dtype=np.int8)).astype(np.int8)

	print(4)

	print(np.sum(mat))

	# G = nx.from_numpy_array(mat)
	# edges=sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True)
	# edges=edges[:avgdeg*n_nodes]
	# H=nx.Graph()
	# H.add_edges_from(edges)
	# adjacent = nx.to_scipy_sparse_matrix(H, format='dok')
	sparse_mat = sparse.dok_matrix(mat)
	print('done')
	return mat, sparse_mat

def Partition(num, size):
	'''
	randomly divide size nodes into num groups
	'''
	partitions={}
	nodes=[x for x in range(1, size+1)]
	group_size=int(size/num)
	for i in range(num-1):
		partitions[i]=[]
		for j in range(group_size):
			node=random.choice(nodes)
			nodes.remove(node)
			partitions[i].append(node)

	#the last partition get the rest nodes
	partitions[num-1]=nodes[:]

	return partitions

def Partition2e(partitions, size):
	'''
	change partition into e vector
	size is the dimension n
	'''
	e={}
	for p in partitions:
		e[p]=[]
		for i in range(1, size+1):
			if i in partitions[p]:
				e[p].append(1.0)
			else:
				e[p].append(0.0)
	return e

def InverseMatrix(A, partitions):
	'''
	use Fast Belief Propagatioin
	CITATION: Danai Koutra, Tai-You Ke, U. Kang, Duen Horng Chau, Hsing-Kuo
	Kenneth Pao, Christos Faloutsos
	Unifying Guilt-by-Association Approaches
	return [I+a*D-c*A]^-1
	'''
	num=len(partitions)		#the number of partition

	I=identity(A.shape[0])          #identity matirx
	D=diags(sum(A).toarray(), [0])  #diagonal degree matrix

	c1=trace(D.toarray())+2
	c2=trace(square(D).toarray())-1
	h_h=sqrt((-c1+sqrt(c1*c1+4*c2))/(8*c2))

	a=4*h_h*h_h/(1-4*h_h*h_h)
	c=2*h_h/(1-4*h_h*h_h)
	
	#M=I-c*A+a*D
		#S=inv(M.toarray())

	M=c*A-a*D
	for i in tqdm(range(num)):
		inv=array([partitions[i][:]]).T
		mat=array([partitions[i][:]]).T
		power=1
		while amax(M.toarray())>10**(-9) and power<10:
			mat=M.dot(mat)
			inv+=mat
			power+=1
		if i==0:
			MatrixR=inv
		else:
			MatrixR=concatenate((MatrixR, array(inv)), axis=1)

	S=csc_matrix(MatrixR)
	return S

def Similarity(A1, A2, g):
	'''
	use deltacon to compute similarity
	CITATION: Danai Koutra, Joshua T. Vogelstein, Christos Faloutsos
	DELTACON: A Principled Massive-Graph Similarity Function g is the number of partition
	'''
	size=A1.shape[0]

	partitions=Partition(g, size)
	e=Partition2e(partitions, size)

	print('partition done, about to invert')
	S1=InverseMatrix(A1, e)
	print('first inverse done')
	S2=InverseMatrix(A2, e)
	print('second inverse done')

	d=0
	for i in range(size):
			for j in range(g):
					d+=(sqrt(S1[i,j])-sqrt(S2[i,j]))**2
	d=sqrt(d)
	sim=1/(1+d)
	print(sim)
	return sim

def DeltaCon(A1, A2, g):
	#compute average sim
	Iteration=10
	average=0.0
	for i in range(Iteration):
		print(i)
		average+=Similarity(A1, A2, g)
	average/=Iteration
	return average

if __name__ == '__main__':
	start=time.time()

	# F1 = np.asarray([[1,1,2,3,4],
 #                [2,6,7,8,9], 
 #                [3,6,7,8,9], 
 #                [4,6,7,8,9], 
 #                [5,6,7,8,9]  
 #              ])
	# F2 = np.asarray([[1,1,2,3,4], 
 #                [2,2,3,4,5], 
 #                [3,6,7,8,9], 
 #                [4,2,3,4,5], 
 #                [5,6,7,8,9]  
 #              ])
	#FB
	# path1 = '/scratch/scratch1/vk/Experiments/FB15k-237-OWE/new_paper/before_any_training_wordnet_short/all_embeddings_r.npy'
	path1 = '/scratch/scratch1/vk/Experiments/FB15k-237-OWE/new_paper/before_any_training_wordnet_long/all_embeddings_r.npy'
	# path1 = '/scratch/scratch1/vk/Experiments/FB15k-237-OWE/new_paper/before_any_training_roberta_long/all_embeddings_r.npy'
	path2 = '/scratch/scratch1/vk/entities_r.p'

	#WN
	# path1 = '/scratch/scratch1/vk/Experiments/wn18rr/new_paper/before_any_training_wordnet/all_embeddings_r.npy'
	# path2 = '/scratch/scratch1/vk/owe_embs_kgzl_style/wn18rr/entities_r.p'

	#YAGO
	# path1 = '/scratch/scratch1/vk/Experiments/yago310/new_paper/before_any_training_wordnet/all_embeddings_r.npy'
	# path2 = '/scratch/scratch1/vk/owe_embs_kgzl_style/yago310/entities_r.p'

	print(path1)
	print(path2)

	F2 = np.hstack((np.load(path2, allow_pickle=True), np.load(path2[:-3] + 'i.p', allow_pickle=True)))

	n_nodes = F2.shape[0]

	F1 = np.load(path1, allow_pickle=True)[:n_nodes]

	A1_large, A1=GenAdjacentMatrix(F1, 100)
	A2_large, A2=GenAdjacentMatrix(F2, 100)

	print('========================')
	mat_and = A1_large * A2_large
	print(np.sum(mat_and))
	print(np.sum(mat_and) / n_nodes)
	print('========================')

	sim=DeltaCon(A1, A2, 5)
	print('sim:', sim)
	end=time.time()
	print('time:',(end-start))