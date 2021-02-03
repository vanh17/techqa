
from transformers import RobertaModel, RobertaTokenizer
from bertviz import util
import numpy as np
import re, string
import math
# from Alignment_function import compute_alignment_score, compute_alignment_score_individual_just

stop_words=string.punctuation.split()+['</s>','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

def get_norm(vec1):
    return [v1 / float(sum(vec1)) for v1 in vec1]

def compute_entropy(list1):
    entropy = []
    mean = sum(list1)/float(len(list1))
    # for prob1 in list1:
    #     entropy.append(math.log(prob1))
    # return -1*(sum(entropy)/float(len(entropy)))

    for prob1 in list1:
        entropy.append(abs(mean-prob1))
    return sum(entropy)/float(len(entropy))

"""
## not required for now

IDF_file_name = "WikiPedia_IDF_vals.json"
with open(IDF_file_name) as json_file:
    IDF_vals_Wiki = json.load(json_file)

MultiRC_IDF_file_name = "MultiRC_IDF_vals.json"
with open(MultiRC_IDF_file_name) as json_file:
    IDF_vals_MultiRC = json.load(json_file)
"""

model_version = 'roberta-base' ## line 26-30 is where we are giving models path - these are the model for which we are evaluating the attention scores.
model = RobertaModel.from_pretrained(model_version, output_hidden_states=True, output_attentions=True)

# model = RobertaModel.from_pretrained("/xdisk/vikasy/QASC_Evidence_Retrieval/QASC_TusharKhot_regression_supervised_EVIDENCE_RETRIEVAL_individual_20/QASC_TusharKhot_regression_supervised_EVIDENCE_RETRIEVAL_individual_20_1_4Epochs_128_RoBERTa_base/checkpoint-6000/", output_hidden_states=True, output_attentions=True)
# model = RobertaModel.from_pretrained("/xdisk/vikasy/QASC_Evidence_Retrieval/QASC_TusharKhot_PARTIAL_regression_supervised_EVIDENCE_CHAIN_RETRIEVAL_10/QASC_TusharKhot_PARTIAL_regression_supervised_EVIDENCE_CHAIN_RETRIEVAL_10_1_4Epochs_128_RoBERTa_base/checkpoint-6000/", output_hidden_states=True, output_attentions=True)


tokenizer = RobertaTokenizer.from_pretrained(model_version)

# sentence_pair_file="/xdisk/vikasy/QASC_dataset/QASC_TusharKhot_regression_supervised_EVIDENCE_RETRIEVAL_individual_20/test.tsv"
sentence_pair_file="dev"
orig_file_input_lines = open(sentence_pair_file+".tsv", "r").readlines() ## reading the dev.tsv file as we are performing attention analysis on the validation set.

All_positive_sentence_pairs = []  ## contains only GOLD evidence text and QA pairs as we are first interested in knowing if it worked for correct cases or not

for line_ind, line1 in enumerate(orig_file_input_lines):
    sent_and_ids = line1.strip().split("\t")
    if sent_and_ids[-1] == "1": ## we converted the dataset in sts-b format which is for regression task in huggingface transformers library.
       All_positive_sentence_pairs.append([sent_and_ids[7], sent_and_ids[8]])  ## QA and evidence text

# print ("the total number of text pairs are: ", len(All_positive_sentence_pairs), All_positive_sentence_pairs[0][0],All_positive_sentence_pairs[0][1])

for sind, sent_pairs in enumerate(All_positive_sentence_pairs):
    if sind % 100 == 0:
        print ("we are at this sentence pair : ", sind)
    sentence_a = sent_pairs[0]  ## sentence_a is question
    sentence_b = sent_pairs[1]  ## sentence_b is candidate evidence sentence


    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    embs_attention = model(input_ids)
    attention = embs_attention[-1]
    embeddings = embs_attention[-2]
    input_id_list = input_ids[0].tolist()  # Doing this for getting the indexes of each word.
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) # Doing this for getting the tokenized words

    if sentence_b:
        sentence_b_start =  tokens.index('</s>') + 2  ## sent1 </s> <s> sent2
    else:
        sentence_b_start = None
    sentence_a_end = sentence_b_start - 3  ## There are two '</s>' between the two sentences
    sentence_a_toks = tokens[1:sentence_a_end + 1]
    sentence_a_toks = [w1.replace("Ġ", "") for w1 in sentence_a_toks]
    sentence_b_toks = tokens[sentence_b_start:-1]
    sentence_b_toks = [w1.replace("Ġ", "") for w1 in sentence_b_toks]

    """
        Henry lives in Tucson
        [CLS]  Hen ry live s in Tuc son
        []    []  [] []   [] [] []  []
        Sent1_attention_index_dict = {"Henry":[1,2], "lives":[3,4]....... }
    
        Henry lives in Arizona
        [SEP]  He n ry live s in  Ari  Zona
        [] [] [] []   [] [] []   []
        Sent2_attention_index_dict = {"Henry":[1,2,3], "lives":[3,4]....... }
    """
    word_list_sentence_a = sentence_a.split(" ")
    word_list_sentence_b = sentence_b.split(" ")
    i, j = 0, 0
    sentence_a_attention_index_dict = {}
    while i < len(word_list_sentence_a) and j < len(sentence_a_toks):
           if word_list_sentence_a[i] != sentence_a_toks[j]:
               united_tokens = sentence_a_toks[j]
               index_list = [j+1]
               while united_tokens != word_list_sentence_a[i]:
                   j += 1
                   index_list.append(j+1)
                   united_tokens = united_tokens + sentence_a_toks[j]
               if united_tokens not in sentence_a_attention_index_dict:
                   sentence_a_attention_index_dict[united_tokens] = index_list
               else:
                   t = 1
                   while (united_tokens + "_" + str(t)) in sentence_a_attention_index_dict:
                       t += 1
                   sentence_a_attention_index_dict[united_tokens+"_"+str(t)] = index_list
           else:
               if sentence_a_toks[j] not in sentence_a_attention_index_dict:
                   sentence_a_attention_index_dict[sentence_a_toks[j]] = [j+1]
               else:
                   t = 1
                   while (sentence_a_toks[j] + "_" + str(t)) in sentence_a_attention_index_dict:
                       t += 1
                   sentence_a_attention_index_dict[sentence_a_toks[j]+"_"+str(t)] = [j+1]
           i += 1
           j += 1
    i, j = 0, 0
    sentence_b_attention_index_dict = {}
    while i < len(word_list_sentence_b) and j < len(sentence_b_toks):
           if word_list_sentence_b[i] != sentence_b_toks[j]:
               united_tokens = sentence_b_toks[j]
               index_list = [j+1]
               while united_tokens != word_list_sentence_b[i]:
                   j += 1
                   index_list.append(j+1)
                   united_tokens = united_tokens + sentence_b_toks[j]
               if united_tokens not in sentence_b_attention_index_dict:
                   sentence_b_attention_index_dict[united_tokens] = index_list
               else:
                   t = 1
                   while (united_tokens + "_" + str(t)) in sentence_b_attention_index_dict:
                       t += 1
                   sentence_b_attention_index_dict[united_tokens+"_"+str(t)] = index_list
           else:
               if sentence_b_toks[j] not in sentence_b_attention_index_dict:
                   sentence_b_attention_index_dict[sentence_b_toks[j]] = [j+1]
               else:
                   t = 1
                   while (sentence_b_toks[j] + "_" + str(t)) in sentence_b_attention_index_dict:
                       t += 1
                   sentence_b_attention_index_dict[sentence_b_toks[j]+"_"+str(t)] = [j+1]
           i += 1
           j += 1
    # End creating sentence attention index dict.
    if sind == 0:
       print ("Sentence A: ",sentence_a)
       print ("Sentence B: ",sentence_b)
       print ("this will give the tokenized subtokens from RoBERTa: ",tokens)
       print ("Sentence A token : ",sentence_a_toks)
       print ("Sentence B token : ",sentence_b_toks)
       print("sentence a dict", sentence_a_attention_index_dict)
       print("sentence b dict", sentence_b_attention_index_dict)

    sentence_a_space_seperated_toks = sentence_a.strip().split() ## question original toks.
    sentence_b_space_seperated_toks = sentence_b.strip().split()

    punct_indices = []
    non_stop_word_indexes_question = []

    for ind1, w1 in enumerate(sentence_a_space_seperated_toks):
        if w1 in string.punctuation or len(w1) == 1:
            punct_indices.append(ind1)
        else:
            non_stop_word_indexes_question.append(1 + ind1) ## because the first word is the <s>



    """  ## I will continue to work on this and complete the IDF part. 
    
    ### Writing attention weights here
    squeezed_attention_weights = util.format_attention(attention).tolist()
    All_layer_indices = [i for i in range(len(squeezed_attention_weights))] ## there are 12 number of layers in base and 24 in large
    # All_layer_indices = [11]
    if sind == 0:
        All_layer_CSAV_scores = {i1: [] for i1 in All_layer_indices}
        All_layer_Ques_Just_attention_scores = {i1: [] for i1 in All_layer_indices}
        All_layer_CLS_attention_scores = {i1: [] for i1 in All_layer_indices}
        All_layer_CLS_attention_scores_from_Ques_J1_J2 = {i1: [] for i1 in All_layer_indices}
        All_layer_CLS_attention_scores_from_nonLinks_vs_LINKING = {i1: [] for i1 in All_layer_indices} ## this one is suggested by Mihai where we look at the average attention
        All_layer_Entropy_quesAttention_scores = {i1: [] for i1 in All_layer_indices}

        All_layer_Ques_Just_Alignment_scores = {i1: [] for i1 in All_layer_indices}
        All_layer_Ques_Just_Alignment_Coverage_scores = {i1: [] for i1 in All_layer_indices}  ## this doesnt work well coz model pays more attention to specific words in the justifications.
        All_layer_Just1_Just2_Alignment_scores = {i1: [] for i1 in All_layer_indices}  ## this is important

    for layer_number in All_layer_indices:
        align_score_coverage = compute_alignment_score_individual_just(non_stop_word_indexes_query, non_stop_word_indexes_justification_1, embeddings, embeddings, layer_number)
        All_layer_Ques_Just_Alignment_scores[layer_number].append(align_score_coverage[0])
        All_layer_Ques_Just_Alignment_Coverage_scores[layer_number].append(align_score_coverage[1])

        align_score_coverage = compute_alignment_score_individual_just(non_stop_word_indexes_query, non_stop_word_indexes_justification_2, embeddings_2, embeddings_2, layer_number)
        All_layer_Ques_Just_Alignment_scores[layer_number].append(align_score_coverage[0])
        All_layer_Ques_Just_Alignment_Coverage_scores[layer_number].append(align_score_coverage[1])

        All_layer_Just1_Just2_Alignment_scores[layer_number].append(compute_alignment_score_individual_just(non_stop_word_indexes_justification_1,non_stop_word_indexes_justification_2, embeddings, embeddings_2, layer_number)[0])

        all_attention_head_weights = np.array(squeezed_attention_weights[layer_number][0])  ### this should be the length of the sentence + 2
        for i in range(len(squeezed_attention_weights[layer_number]) - 1):  ## layer = layer_number, head = i
            all_attention_head_weights += np.array(squeezed_attention_weights[layer_number][i + 1])
        # print ("the layer-wise attention weights look like ", all_attention_head_weights)
        all_attention_head_weights = all_attention_head_weights.tolist()
        ####
        
    """








