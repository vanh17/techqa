import json
# for now we import sys for fast research 
import sys

def main():
	# usage: python3 localize_doc_spans.py  old_doc_technotes.json new_doc_span.json new_technotes_Q_A.json
	# opening the json file
	dev = open(sys.argv[1], "r")
	localized_dev = open(sys.argv[2], "r")

	# return json object (list of dictionary)
	dev = json.load(dev) # this is the technotes dictionary from IBM (we will modify the text field of it using new_doc_span from predictions)
	localized_dev = json.load(localized_dev) # this will be a dictionary consiting documents that is partial of the new spans.

	# iterating each documents. Each documents here is a dictionary
	for doc in dev.keys():
		# if it does not in the localized, we just ignore it because it is not a candidate.
		if doc in localized_dev:
			start = localized_dev[doc]["start_offset"]
			end = localized_dev[doc]["end_offset"]
			# we might want to experience with this windows to see how does it works.
			dev[doc]["text"] = dev[doc]["text"][start:end] 

	# dump agumented data to new json file
	with open(sys.argv[3], 'w') as outfile:
		formatted_json = json.dumps(dev, indent=2)
		outfile.write(formatted_json)

# call main for function to start:
if __name__ == '__main__':
	main()