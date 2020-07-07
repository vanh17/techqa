import sys

def main():
	file = open(sys.argv[1], "r")
	capitalized_dict= {}
	file = file.readlines()
	for line in file:
		#  label sentence1 sentence2 id
		line = line.split("\t")
		label = line[0]
		line = line[2].lstrip().split(" ")
		if label == "1":
			if line[0][:5] not in capitalized_dict:
				capitalized_dict[line[0][:5]] = 1
			else:
				capitalized_dict[line[0][:5]] += 1
	sorted_dict = sorted(capitalized_dict.items(), key=lambda x: x[1], reverse=True)
	print(sorted_dict)
	for word in sorted_dict:
		print(word[0], word[1])

main()