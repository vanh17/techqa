import json
# for now we import sys for fast research 
import sys

def main():
	# usage: python3 get_simplified_spans_for_predictions.py old_predictions.json new_doc_span.json original_span_predictions.json
	# opening the json file
	simplified = open(sys.argv[1], "r")
	localized_dev = open(sys.argv[2], "r")

	# return json object (list of dictionary)
	simplified = json.load(simplified) # this is simplified span predictions.
	localized_dev = json.load(localized_dev) # this will be a dictionary consiting documents that is partial of the new spans.

	# iterating each documents. Each documents here is a dictionary
	for q in simplified["predictions"].keys():
		# if it does not in the localized, we just ignore it because it is not a candidate.
		for i in range(len(simplified["predictions"][q])):
			if simplified["predictions"][q][i]["doc_id"] in localized_dev:
				simplified["predictions"][q][i]["start_offset"] += localized_dev[simplified["predictions"][q][i]["doc_id"]]["start_offset"]
				simplified["predictions"][q][i]["end_offset"] += localized_dev[simplified["predictions"][q][i]["doc_id"]]["start_offset"]

	# dump agumented data to new json file
	with open(sys.argv[3], 'w') as outfile:
		formatted_json = json.dumps(simplified, indent=2)
		outfile.write(formatted_json)

# call main for function to start:
if __name__ == '__main__':
	main()