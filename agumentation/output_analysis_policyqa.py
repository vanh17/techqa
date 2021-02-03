import json
# for now we import sys for fast research 
import sys
import random

def main():
	# usage: python3 output_analysis_policyqa.py best_config_preds.json baseline_preds.json dev.json analysis.txt
	# opening the json file
	best_config = json.load(open(sys.argv[1], "r"))
	baseline = json.load(open(sys.argv[2], "r"))
	data = json.load(open(sys.argv[3], "r"))


	# intial holder of outputs that are not the same from best config and baseline.
	# This will hold information in the format of {question, original, best, baseline, context}
	differences = []

	for sample in data["data"]: # this is to traverse through the list of policies. 
	# each policy contains a policy title and the paragaphs.
		for p in sample['paragraphs']: # each p is a dict {qas, index, context, and summary}
			for q in p["qas"]: # now getting hold of each question for each p (paragaph) in paragraphS 
				# first we need to add any question from original p["qas"] to this
				q_id = q["id"]
				original = q["answers"][0]["text"]
				best_ans = best_config[q_id]
				base_ans = baseline[q_id]
				# we focus on the exact match on PolicyQA dataset because it shows the detailed performance improvement of the model.
				if best_ans != base_ans and best_ans == original:
					differences.append((q["question"], original, best_ans, base_ans, p["context"]))

	print(len(differences))

	# dump agumented data to new json file
	with open(sys.argv[4], 'w') as outfile:
		for turple in differences:
			outfile.write(turple[0] + "\t" + turple[1] + "\t" + turple[2] + "\t" + turple[3] + "\t" + turple[4] + "\n"+"\n")

# call main for function to start:
if __name__ == '__main__':
	main()