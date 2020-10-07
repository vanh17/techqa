import json
# for now we import sys for fast research 
import sys

def main():
	# usage: python3 get_important_spans.py predictions_20.json prediction_with_new_span.json doc_new_spans.json
	# opening the json file
	file = open(sys.argv[1], "r")

	# return json object (list of dictionary)
	data = json.load(file)

	# create doc spans for all spans
	doc_spans = {}

	# iterating each predictions for each question.
	for question in data["predictions"].keys():
		# for each question, iterate through document to find smallest start_offset and largest end_offset
		# and store it inside this dictionary
		offset_spans = {}
		# create new prediction with altered spans for each question
		new_predictions = []
		for pred in data["predictions"][question]:
			if pred["doc_id"] not in offset_spans:
				offset_spans[pred["doc_id"]] = {}
				doc_spans[pred["doc_id"]] = {}
				offset_spans[pred["doc_id"]]["start_offset"] = pred["start_offset"]
				offset_spans[pred["doc_id"]]["end_offset"] = pred["end_offset"]
			else:
				if pred["start_offset"] < offset_spans[pred["doc_id"]]["start_offset"]:
					offset_spans[pred["doc_id"]]["start_offset"] = pred["start_offset"]
				if pred["end_offset"] > offset_spans[pred["doc_id"]]["end_offset"]:
					offset_spans[pred["doc_id"]]["end_offset"] = pred["end_offset"]
			if pred["doc_id"] not in doc_spans:
				doc_spans[pred["doc_id"]]["end_offset"] = pred["end_offset"]
				doc_spans[pred["doc_id"]]["start_offset"] = pred["start_offset"]
			else:
				if pred["start_offset"] < doc_spans[pred["doc_id"]]["start_offset"]:
					doc_spans[pred["doc_id"]]["start_offset"] = pred["start_offset"]
				if pred["end_offset"] > doc_spans[pred["doc_id"]]["end_offset"]:
					doc_spans[pred["doc_id"]]["end_offset"] = pred["end_offset"]
		# altering the start_offset and end_offset
		# for pred in data["predictions"][question]:
		# 	pred["start_offset"] = offset_spans[pred["doc_id"]]["start_offset"]
		# 	pred["end_offset"] = offset_spans[pred["doc_id"]]["end_offset"]
		# we want to keep one document for each question remove the duplicate
		for pred in data["predictions"][question]:
			if pred["doc_id"] in offset_spans:
				pred["start_offset"] = offset_spans[pred["doc_id"]]["start_offset"]
				pred["end_offset"] = offset_spans[pred["doc_id"]]["end_offset"]
				new_predictions.append(pred)
				offset_spans.pop(pred["doc_id"], None)
			else:
				continue
		data["predictions"][question] = new_predictions
	# close the file
	file.close()

	# dump agumented data to new json file
	with open(sys.argv[2], 'w') as outfile:
		formatted_json = json.dumps(data, indent=2)
		outfile.write(formatted_json)

	# dump doc spans to new json file
	with open(sys.argv[3], 'w') as outfile:
		formatted_json = json.dumps(doc_spans, indent=2)
		outfile.write(formatted_json)

# call main for function to start:
if __name__ == '__main__':
	main()