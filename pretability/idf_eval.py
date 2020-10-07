import sys
import json
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk

# ************************IDF Distribution Analysis***********************************
# to find the idf average across the sentence. We will use this number as an indicator
# to group tokens into 2 groups: high IDF (has IDF higher than average) and low IDF 
# (has IDF lower than the average).
def sentence_idf_average(sentence, idf_dict):
	sum_idf = 0
	total_found_token = 0
	for token in sentence:
		if token in idf_dict:
			sum_idf += idf_dict[token]
			total_found_token += 1
	if total_found_token == 0:
		return sum_idf
	return sum_idf/total_found_token

# Use idf average to split tokens existing in idf_dict into two groups: high_idf
# and low_idf. 
def split_into_high_low_idf(sentence, idf_dict):
	idf_average = sentence_idf_average(sentence, idf_dict)
	high_idf, low_idf = [], []
	if idf_average == 0:
		return high_idf, low_idf
	for token in sentence:
		if token in idf_dict:
			token_idf = idf_dict[token]
			if token_idf >= idf_average:
				high_idf.append((token, token_idf))
			else:
				low_idf.append((token, token_idf))
	return high_idf, low_idf

# to better see the distribution, we find the average of each group in a sentence.
# The two average will then be plotted.
def average_idf_in_group(idf_group):
	sum_idf = 0
	for token in idf_group:
		sum_idf += token[1]
	return sum_idf/len(idf_group)


# plot the idf 
def plot_idf_distribution(avg_idf_highs, avg_idf_lows):
  	pos = [x for x in avg_idf_highs]
  	neg = [x for x in avg_idf_lows]
  	plt.plot(pos, [0 for _ in range(len(pos))], '.r')
  	plt.plot(neg, [0 for _ in range(len(neg))], '.b')
  	plt.title('IDF distribution over sentences')
  	plt.show()

# Full IDF distribution analysis creation
def idf_distribution_analysis(sentences, idf_dict):
	avg_idf_highs, avg_idf_lows = [], [] 
	for sentence in sentences:
		high_idf, low_idf = split_into_high_low_idf(sentence, idf_dict)
		if len(high_idf) != 0:
			avg_idf_highs.append(average_idf_in_group(high_idf))
		if len(low_idf) != 0:
			avg_idf_lows.append(average_idf_in_group(low_idf))
	plot_idf_distribution(avg_idf_highs, avg_idf_lows)
# *********************END IDF DISTRIBUTION ANALYSIS******************************

# ***************************************F1 vs Threshold Analysis*****************************************
# extract the max_idf from the sentence. This IDF value will help us determine what words in 
# the sentence can be used as a predicted concepts.
def max_idf_in_sentence(sentence, idf_dict):
	max_idf = [] # 0
	for word in sentence:
		if word in idf_dict:
			word_idf = idf_dict[word]
			if True: #word_idf >= max_idf:
				max_idf.append(word_idf)
	# to find the third highest IDF
	if len(max_idf) == 0:
		return []
	return [sorted(max_idf)[-min(len(max_idf),3)]]

# using the threshold of 0.5, 0.6, 0.7 from the max value to determine what concepts should be in the prediction
def get_predicted_concepts(sentence, idf_dict, threshold):
	# initialize the prediction holder
	predicted_concepts = []
	max_idf = max_idf_in_sentence(sentence, idf_dict)
	if len(max_idf) == 0: # change this from max_idf == 0 to get the top.
		return predicted_concepts
	for word in sentence:
		if word in idf_dict:
			if idf_dict[word] >= max_idf[0]:
				predicted_concepts.append(word)
	return predicted_concepts

def plot_2d_pts(data_points):
  threholds = [x for (x,y) in data_points]
  f1_avgs = [y for (x,y) in data_points]
  plt.plot(threholds, f1_avgs, '.r')
  plt.title('F1 vs Threshold')
  plt.show()

# calculate f1 average score across all the senteces
def evaluate(gold_concepts, predicted_concepts):
	sum_f1 = 0
	for i in range(len(gold_concepts)):
		matches = len(set(gold_concepts[i]) & set(predicted_concepts[i]))
		precision = matches/len(predicted_concepts[i])
		recall = matches/len(gold_concepts[i])
		# prevent cases where precision and recall both equal zero!
		# F1
		if max(precision, recall) != 0:
			sum_f1 += 2*precision*recall/(precision+recall)
		# Recall
		# if recall != 0:
		# 	sum_f1 += recall
	return sum_f1/len(gold_concepts)

def f1_vs_threshold_analysis(train_data, idf_dict, threshold):
	gold_concepts = []
	predicted_concepts = []
	for gold, sentence in train_data:
		pred = get_predicted_concepts(sentence, idf_dict, threshold)
		# for now we ignore all the sentences that does not have
		# any words in the IDF dictionary
		if len(pred) != 0:
			gold_concepts.append(gold.split())
			predicted_concepts.append(pred)
	f1_avg = evaluate(gold_concepts, predicted_concepts)
	print(len(gold_concepts), len(predicted_concepts))
	return (threshold, f1_avg)

# *********************************************End F1 analysis**********************************************

def main():
	# initialize stemmer
	p = nltk.PorterStemmer()
	# usuage: python3 idf_eval.py idf_dict.json train.json
	idf_dict = json.load(open(sys.argv[1]))
	train_data = []
	for json_obj in open(sys.argv[2]):
		json_obj = json.loads(json_obj)["question"]
		train_data.append((json_obj["question_concept"],  json_obj["stem"].translate(str.maketrans('','',string.punctuation)).lower().split()))
	# print(idf_dict["book"], train_data[0][1])
	sentences = [sent[1] for sent in train_data]
	idf_distribution_analysis(sentences, idf_dict)
	# F1 vs Threshold analysis
	threholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
	data_points = []
	for t in threholds:
		data_points.append(f1_vs_threshold_analysis(train_data, idf_dict, t))
	plot_2d_pts(data_points)


if __name__ == '__main__':
	main()






