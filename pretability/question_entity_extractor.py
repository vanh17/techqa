import json
import sys
import random 

def main():
	# usuage: python3 question_entity_extractor.py original.json question_entity_only.json
	# load question, entity pairs one at time.
	train_data = []
	for json_obj in open(sys.argv[1]):
		json_obj = json.loads(json_obj)["question"]
		train_data.append(json_obj["stem"] + "\t" + json_obj["question_concept"])
	# dump agumented data to new json file
	train_data = random.choices(train_data, k=200)
	with open(sys.argv[2], 'w') as outfile:
		formatted_json = json.dumps(train_data, indent=2)
		outfile.write(formatted_json)


if __name__ == '__main__':
	main()