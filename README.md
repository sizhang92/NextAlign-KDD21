# NextAlign-KDD21
Code for the paper "Balancing Consistency and Disparity in Network Alignment"

## Overview
The package contains the following folders and files:
- dataset: including the datasets used in the paper. Datasets are of the same format. Example illustrations are as below.
	- phone-email_0.2.npz: Phone-Email networks with 20% training data
	- node2vec_context_pairs_phone-email_0.2.npz: the extracted within-network positive context pairs by node2vec
	- rwr_emb_phone-email_0.2.npz: random walk with restart w.r.t. the anchor nodes
- layers: 
	- RelGCN.py: the proposed RelGCN layer for graph alignment
- model:
	- model.py: overall architecture of NextAlign
	- negative_sampling.py: the proposed negative sampling method
- utils: 
	- node2vec.py: original code from nodevec paper
	- rwr_scoring.py: computes the random walk with restart w.r.t. the anchor nodes from scratch
	- test.py: evaluate the performance on testing node alignments
	- utils.py: auxiliary functions
- train.py: main code to run the whole algorithm

## To run

Simply run the following command. Change dataset names per demand.

python train.py --epochs=200 --dataset=phone-email

## Reference

Zhang, Si, et al. "Balancing Consistency and Disparity in Network Alignment." Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2021.