import sys
def convert_to_text(string_list):
	string = ""
	for word in string_list:
		word = word[1: -1]
		if word == '*' or "[SEP]" in word:
			continue
		if "##" in word:
			string += word[2:]
		else:
			string += " " + word
	return string

def main():
	if len(sys.argv) != 3:
		print("[usage]: python convert_techqa_to_mrpc.py techqa mrpc_techqa")
	else:
		techqa = open(sys.argv[1], "r")
		techqa = techqa.readlines()
		mrpc = open(sys.argv[2], "w")
		mrpc.write("Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
		i1 = 0
		i2 = 0
		for line in techqa:
			line = line.strip("\n")
			# remove outer closing "[]"
			line = line[1:len(line)-1]
			# split by ", "
			line = line.split(", ")
			if len(line) > 1:
				# start offset:
				start = int(line[-2])
				# end offset:
				end = int(line[-1])
				# question extraction:
				step_token_count = 0
				SEP = [line.index("'[SEP]'")]
				SEP.append(line.index("'[SEP]'", SEP[0]+1))
				SEP.append(line.index("'[SEP]'", SEP[1]+1))
				string1 = convert_to_text(line[2: SEP[0]])
				# change 128 to 64 if it is too long, or may be 256 if it is too large
				for i in range(SEP[2]+1, len(line)-2, 128):
					# string characters
					label = "0"
					end_index = min(i+128, len(line)-2)
					string2 = convert_to_text(line[i: end_index])
					ID1 = str(i1)
					ID2 = str(i2)
					if start != 0 or end != 0:
						if i <= start and end_index >= start:
							label = "1" 
						if i <= end and end_index >= end:
							label = "1"
						if i >= start and end_index <= end:
							label = "1"
						if i <= start and end_index >= end:
							label = "1"
					mrpc.write(label+"\t"+ID1+"\t"+ID2+"\t"+string1+"\t"+string2+"\n")
					i1 += 1
					i2 += 1

if __name__ == '__main__':
	main()





