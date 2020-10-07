import torch
from transformers import *
import sys
import json
import random
import numpy as np
import string
import nltk
from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')

# ************************************************* CALLING BERT/TRANSFORMERS ****************************************************************************
# ********************************************************************************************************************************************************
# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut

# MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
#           (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
#           (GPT2Model,       GPT2Tokenizer,       'gpt2'),
#           (CTRLModel,       CTRLTokenizer,       'ctrl'),
#           (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
#           (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
#           (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
#           (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
#           (RobertaModel,    RobertaTokenizer,    'roberta-base'),
#           (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
#          ]

# use BERT weights and BERT tokenizer for now.
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Encode text
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                     BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
for model_class in [BertModel]: #BERT_MODEL_CLASSES: comment this out for now.
    # Load pretrained model/tokenizer
    model = model_class.from_pretrained(pretrained_weights)

    # Models can return full list of hidden-states & attentions weights at each layer
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=True)
# ********************************************* END INITIALIZING BERT/TRANSFORMERS *********************************************************************

def get_bert_embeddings(sentence):
  global tokenizer, model 
  encoded_tokens = tokenizer.encode(sentence)
  input_ids = torch.tensor([encoded_tokens])
  all_hidden_states, all_attentions = model(input_ids)[-2:]
  # for now get the last hidden states. 
  return all_hidden_states[11][0], encoded_tokens


# There are two type of hard tokenization
# 1. let's can't won't don't does't isn't aren't wasn't weren't hasn't shouldn't mustn't didn't wouldn't
# 2. ##s 
# For 1 we don't have to worry about these because they will not likely to be concepts anyway.
def convert_embed_to_words(list_ids, list_embeds):
  global special_tokens, stopwords_list
  tokens = tokenizer.convert_ids_to_tokens(list_ids)
  word_embeddings = []
  # ignore the [CLS] and [SEP] tokens at the beigning and at the end of the list
  for i in range(1, len(tokens)-1):
    if "##" not in tokens[i]:
      if tokens[i] not in stopwords_list:
        word_embeddings.append((tokens[i], list_embeds[i]))
    else:
      print(word_embeddings[-1][0]+tokens[i][2:])
      word_embeddings[-1] = (word_embeddings[-1][0]+tokens[i][2:], (np.array(word_embeddings[-1][1])+ np.array(list_embeds[i]))/2)
  return [(word, np.linalg.norm(embed.detach().numpy())) for word, embed in word_embeddings]


# ***************************************F1 vs Threshold Analysis*****************************************
# extract the max_norm from the sentence. This NORM value will help us determine what words in 
# the sentence can be used as a predicted concepts.
# Edited 09/08, make this function to return min of NORM
def max_norm_in_sentence(processed_sentence, t):
  max_norm = [] # 0
  for word, embed in processed_sentence:
        max_norm.append(embed)
  # to find the third highest IDF
  if len(max_norm) == 0:
    return []
  # get the third largest norm
  # # EDIT 09/08/: remove the negative sign in front of min and add -1  to return the smallest number
  return [sorted(max_norm)[min(len(max_norm)-1,t)]]

# using the top n = 3 largest from the max value to determine what concepts should be in the prediction
def get_predicted_concepts(processed_sentence, t):
  # EDIT 09/08/: add global stopwords_list
  global stopwords_list
  # initialize the prediction holder
  predicted_concepts = []
  max_norm = max_norm_in_sentence(processed_sentence, t)
  if len(max_norm) == 0:
    return predicted_concepts
  for word, embed in processed_sentence:
      # EDIT 09/08/: change >= to <=
      if embed <= max_norm[0] and word not in stopwords_list:
        predicted_concepts.append(word)
  return predicted_concepts

# Added 09/8: to find the predicted words concepts based on the surrounding neighbors.
def get_predicted_window_concepts(processed_sentence):
  global stopwords_list
  # initialize the prediction holder
  predicted_concepts = []
  max_norm = max_norm_in_sentence(processed_sentence, 1)
  if len(max_norm) == 0:
    return predicted_concepts
  processed_sentence = [w for w in processed_sentence if w not in stopwords_list]
  for i in range(len(processed_sentence)):
      word, embed = processed_sentence[i]
      if embed == max_norm[0]:
        predicted_concepts.append(word)
        if i > 0:
          predicted_concepts.append(processed_sentence[i-1][0])
        if i < len(processed_sentence)-1:
          predicted_concepts.append(processed_sentence[i+1][0])
  return predicted_concepts

# calculate f1 average score across all the senteces
def evaluate(gold_concepts, predicted_concepts):
  sum_f1 = 0
  size = 0
  for i in range(len(gold_concepts)):
    if len(gold_concepts[i]) > 0:
      matches = len(set(gold_concepts[i]) & set(predicted_concepts[i]))
      precision = matches/len(predicted_concepts[i])
      recall = matches/len(gold_concepts[i])
      # prevent cases where precision and recall both equal zero!
      # F1
      if max(precision, recall) != 0:
        sum_f1 += 2*precision*recall/(precision+recall)
        size += 1
      # Recall
      # if recall != 0:
      #   sum_f1 += recall
  return sum_f1/size

def f1_vs_threshold_analysis(train_data, t):
  # # EDIT 09/08/: add global stopword here
  global stopwords_list
  gold_concepts = []
  predicted_concepts = []
  for gold, sentence in train_data:
    last_hidden_states, encoded_tokens = get_bert_embeddings(sentence)
    processed_sentence = convert_embed_to_words(encoded_tokens, last_hidden_states)
    pred = get_predicted_concepts(processed_sentence, t)
    # for now we ignore all the sentences that does not have
    # any words in the IDF dictionary
    if len(pred) != 0:
      # EDIT 09/08/: remove stopwords from gold targeted
      gold_concepts.append([w for w in gold.split() if w not in stopwords_list])
      predicted_concepts.append(pred)
  f1_avg = evaluate(gold_concepts, predicted_concepts)
  return (t, f1_avg)
# ***************************************************END F1 VS Threshold Analysis***************************************

def main():
  # initialize stemmer
  p = nltk.PorterStemmer()
  train_data = []
  for json_obj in open(sys.argv[1]):
    json_obj = json.loads(json_obj)["question"]
    # # EDIT 09/08/: remove all n't in the sentence to make sure we can remove stopwords easier
    train_data.append((json_obj["question_concept"],  json_obj["stem"].translate(str.maketrans('','',string.punctuation)).lower().replace("n't", "").split()))
    # F1 vs Threshold analysis
  threholds = [1, 2, 3, 4, 5]
  data_points = []
  for t in threholds:
    print("start threshold: ", t)
    data_points.append(f1_vs_threshold_analysis(train_data, t))
    print("results from threshold ", t, data_points[-1])
  print(data_points)

if __name__ == '__main__':
  main()

