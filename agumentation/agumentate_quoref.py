import json
# for now we import sys for fast research 
import sys
import random

def main():
	# usage: python3 agumentate_quoref.py quoref_file.json positive_sub_sampling new.json
	# opening the json file
	file = open(sys.argv[1], "r")

	# return json object (list of dictionary)
	data = json.load(file)

	print(data.keys())
	print(len(data["data"]))
	print(data["data"][0].keys())
	print(len(data["data"][0]["paragraphs"]))
	print(len(data["data"][0]["paragraphs"][0]))
	print(data["data"][0]["paragraphs"][0].keys())	
	print(len(data["data"][0]["paragraphs"][0]["qas"]))	
	print(data["data"][0]["paragraphs"][0]["qas"][0].keys())
	print(len(data["data"][0]["paragraphs"][0]["qas"][0]["answers"]))
	c = 0
	for p in data["data"]:
		for s in p["paragraphs"]:
			for qas in s["qas"]:
				c += 1
	print(c)

	

	# intial new data dictionary
	new_data = {"version": "augmentation", 'data': []}

	for sample in data["data"]: # this is to traverse through the list of policies. 
	# each policy contains a policy title and the paragaphs.
		# create new policy to append to new_data dictionary
		new_policy = {'title': sample['title'], 'paragraphs': []}
		for p in sample['paragraphs']: # each p is a dict {qas, index, context, and summary}
			# create new paragraph here to append to new_policy
			new_p = {"qas": [], "context_id": p["context_id"], "context": p["context"]}
			for q in p["qas"]: # now getting hold of each question for each p (paragaph) in paragraphS 
				# first we need to add any question from original p["qas"] to this
				if len(q["answers"]) == 0:
					print("found original empty!!!!!!!!")
				new_p["qas"].append(q)
				# Then using random number here to make sure that we can get the 20%, 40%, 60%, 80% augmentation percentage
				if random.random() <= float(sys.argv[2]):	
					# create new question (5, 5):
					new_q = {"question": q["question"], "id": q["id"]+"1", "answers": []}
					# traverse through each answer for this question q, and move the window span (10, 10)
					for ans in q["answers"]:
						new_end = ans["answer_start"] + len(ans["text"]) + 5
						new_start = ans["answer_start"] + 5
						# create new answer for this questions
						new_ans = {"text": p["context"][new_start:new_end], "answer_start": new_start}
						if new_start < len(p["context"]):
							new_q["answers"].append(new_ans)
					if len(new_q["answers"]) != 0:
						new_p["qas"].append(new_q)

					# create new sample (-5, -5)
					new_q = {"question": q["question"],"id": q["id"]+"2", "answers": []}
					# traverse through each answer for this question q, and move the window span (-10, -10)
					for ans in q["answers"]:
						new_end = ans["answer_start"] + len(ans["text"]) - 5
						new_start = max(0, ans["answer_start"] - 5)
						# create new answer for this questions
						new_ans = {"text": p["context"][new_start:new_end], "answer_start": new_start}
						new_q["answers"].append(new_ans)
					if len(new_q["answers"]) != 0:
						new_p["qas"].append(new_q)
			
					# create new sample (-5, 0)
					new_q = {"question": q["question"], "id": q["id"]+"3", "answers": []}
					# traverse through each answer for this question q, and move the window span (-15, 0)
					for ans in q["answers"]:
						new_end = ans["answer_start"] + len(ans["text"])
						new_start = max(0, ans["answer_start"] - 5)
						# create new answer for this questions
						new_ans = {"text": p["context"][new_start:new_end], "answer_start": new_start}
						new_q["answers"].append(new_ans)
					if len(new_q["answers"]) != 0:
						new_p["qas"].append(new_q)
			
					# create new sample (0, 5)
					new_q = {"question": q["question"], "id": q["id"]+"4", "answers": []}
					# traverse through each answer for this question q, and move the window span (0, 15)
					for ans in q["answers"]:
						new_end = ans["answer_start"] + len(ans["text"]) + 5
						new_start = ans["answer_start"]
						# create new answer for this questions
						new_ans = {"text": p["context"][new_start:new_end], "answer_start": new_start}
						new_q["answers"].append(new_ans)
					if len(new_q["answers"]) != 0:
						new_p["qas"].append(new_q)
					# shuffle the question.
					random.shuffle(new_p["qas"])
			# append new paragraph to each new policy, going up one level
			new_policy["paragraphs"].append(new_p)
		# going up another level
		# append new policy to the new_data
		new_data['data'].append(new_policy)	
	
	# close the file
	file.close()

	c = 0
	for p in new_data["data"]:
		for s in p["paragraphs"]:
			for qas in s["qas"]:
				c += 1
				if len(qas["answers"]) == 0:
					print("empty answer!!!!!!")
	print(c)

	# dump agumented data to new json file
	with open(sys.argv[3], 'w') as outfile:
		json.dump(new_data, outfile)

# call main for function to start:
if __name__ == '__main__':
	main()