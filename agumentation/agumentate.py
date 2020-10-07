import json
# for now we import sys for fast research 
import sys
import random

def main():
	# usage: python3 agumentate.py file.json positive_sub_sampling front_offset back_offset new.json
	# opening the json file
	file = open(sys.argv[1], "r")

	# dictionary keys for better create documents
	fields = ['QUESTION_ID', 'QUESTION_TITLE', 'QUESTION_TEXT', 'DOCUMENT', 'ANSWER', 'START_OFFSET', 'END_OFFSET', 'ANSWERABLE', 'DOC_IDS']

	# return json object (list of dictionary)
	data = json.load(file)

	# intial new data
	new_data = []

	for sample in data:
		new_data.append(sample)
		if sample["ANSWERABLE"] == "Y" and random.random() <= float(sys.argv[2]):	
			# create new sample (10, 10):
			new_sample = {}
			new_sample['QUESTION_ID'] = sample['QUESTION_ID']
			new_sample['QUESTION_TITLE'] = sample['QUESTION_TITLE']
			new_sample['QUESTION_TEXT'] = sample['QUESTION_TEXT']
			new_sample['DOCUMENT'] = sample['DOCUMENT']
			new_sample['ANSWER'] = sample['ANSWER']
			new_sample['START_OFFSET'] = int(sample['START_OFFSET']) + 10
			new_sample['END_OFFSET'] = int(sample['END_OFFSET']) + 10
			new_sample['ANSWERABLE'] = sample['ANSWERABLE']
			new_sample['DOC_IDS'] = sample['DOC_IDS']
			new_data.append(new_sample)
			# create new sample (-10, -10)
			new_sample = {}
			new_sample['QUESTION_ID'] = sample['QUESTION_ID']
			new_sample['QUESTION_TITLE'] = sample['QUESTION_TITLE']
			new_sample['QUESTION_TEXT'] = sample['QUESTION_TEXT']
			new_sample['DOCUMENT'] = sample['DOCUMENT']
			new_sample['ANSWER'] = sample['ANSWER']
			new_sample['START_OFFSET'] = int(sample['START_OFFSET']) -10
			new_sample['END_OFFSET'] = int(sample['END_OFFSET']) -10
			new_sample['ANSWERABLE'] = sample['ANSWERABLE']
			new_sample['DOC_IDS'] = sample['DOC_IDS']
			new_data.append(new_sample)
			# create new sample (-20, -20)
			new_sample = {}
			new_sample['QUESTION_ID'] = sample['QUESTION_ID']
			new_sample['QUESTION_TITLE'] = sample['QUESTION_TITLE']
			new_sample['QUESTION_TEXT'] = sample['QUESTION_TEXT']
			new_sample['DOCUMENT'] = sample['DOCUMENT']
			new_sample['ANSWER'] = sample['ANSWER']
			new_sample['START_OFFSET'] = int(sample['START_OFFSET']) -20
			new_sample['END_OFFSET'] = int(sample['END_OFFSET']) -20
			new_sample['ANSWERABLE'] = sample['ANSWERABLE']
			new_sample['DOC_IDS'] = sample['DOC_IDS']
			new_data.append(new_sample)
			# create new sample (20, 20)
			new_sample = {}
			new_sample['QUESTION_ID'] = sample['QUESTION_ID']
			new_sample['QUESTION_TITLE'] = sample['QUESTION_TITLE']
			new_sample['QUESTION_TEXT'] = sample['QUESTION_TEXT']
			new_sample['DOCUMENT'] = sample['DOCUMENT']
			new_sample['ANSWER'] = sample['ANSWER']
			new_sample['START_OFFSET'] = int(sample['START_OFFSET']) + 20
			new_sample['END_OFFSET'] = int(sample['END_OFFSET']) + 20
			new_sample['ANSWERABLE'] = sample['ANSWERABLE']
			new_sample['DOC_IDS'] = sample['DOC_IDS']
			new_data.append(new_sample)
			# create new sample (20, 0)
			new_sample = {}
			new_sample['QUESTION_ID'] = sample['QUESTION_ID']
			new_sample['QUESTION_TITLE'] = sample['QUESTION_TITLE']
			new_sample['QUESTION_TEXT'] = sample['QUESTION_TEXT']
			new_sample['DOCUMENT'] = sample['DOCUMENT']
			new_sample['ANSWER'] = sample['ANSWER']
			new_sample['START_OFFSET'] = int(sample['START_OFFSET']) + 20
			new_sample['END_OFFSET'] = int(sample['END_OFFSET'])
			new_sample['ANSWERABLE'] = sample['ANSWERABLE']
			new_sample['DOC_IDS'] = sample['DOC_IDS']
			new_data.append(new_sample)
	# close the file
	file.close()

	# shuffle the sample
	random.shuffle(new_data)

	# dump agumented data to new json file
	with open(sys.argv[-1], 'w') as outfile:
		json.dump(new_data, outfile)

# call main for function to start:
if __name__ == '__main__':
	main()