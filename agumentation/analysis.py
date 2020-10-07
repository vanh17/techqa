import json
# for now we import sys for fast research 
import sys

def main():
	# usage: python3 analysis.py new_dev_Q_A.json
	# opening the json file
	simplified = open(sys.argv[1], "r")

	# return json object (list of dictionary)
	simplified = json.load(simplified) # this is simplified span predictions.

	# iterating each documents. Each documents here is a dictionary
	no_docs = 0
	for q in simplified:
		# if it does not in the localized, we just ignore it because it is not a candidate.
		if no_docs < len(q["DOC_IDS"]):
			no_docs = len(q["DOC_IDS"])

	# dump agumented data to new json file
	print(no_docs)

# call main for function to start:
if __name__ == '__main__':
	main()