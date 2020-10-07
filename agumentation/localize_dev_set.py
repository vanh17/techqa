import json
# for now we import sys for fast research 
import sys

def main():
	# usage: python3 get_localize_dev_set.py dev_Q_A.json new_predictions_with_new_span.json new_dev_Q_A.json
	# opening the json file
	dev = open(sys.argv[1], "r")
	localized_dev = open(sys.argv[2], "r")

	# return json object (list of dictionary)
	dev = json.load(dev) # this be a list of 300 sentences
	localized_dev = json.load(localized_dev) # this will be a dictionary of two attributes "threshold" and "predictions". The later is a list
	# check if the localized dev is in the right format of the predictions.json
	assert("predictions" in localized_dev)
	# check if the original dev set and the localized dev have the same number of questions.
	assert(len(dev) == len(localized_dev["predictions"]))
	# throw out the "threshold" attribute, only focus on the predictions.
	localized_dev = localized_dev["predictions"]

	# iterating each question for each question. Each question here is a dictionary
	for i in range(len(dev)):
		# get the question id
		qid = dev[i]["QUESTION_ID"]
		# get the list of docs from localized_dev
		localized_preds = localized_dev[qid]
		# create new list of docs containing only docs from the first step predictions
		new_doc_ids = [pred["doc_id"] for pred in localized_preds]
		# assign new_doc_ids to the replace the old doc_ids.
		dev[i]["DOC_IDS"] = new_doc_ids

	# dump agumented data to new json file
	with open(sys.argv[3], 'w') as outfile:
		formatted_json = json.dumps(dev, indent=2)
		outfile.write(formatted_json)

# call main for function to start:
if __name__ == '__main__':
	main()