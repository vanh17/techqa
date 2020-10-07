import json
# for now we import sys for fast research 
import sys
import re

def clean_text(text):
	# ALSO check the length of the old text and does the replace of the new text.
	# convert newline character into special token <newline>
	# DOUBLE check for this if it is one character or the length of 2
	# KEEP THE TWO VERSION: text with tables (after remove the borders and so on) and text with no table at all.
	newline = re.compile(r'\n')
	text = re.sub(newline, " ", text)
	# convert html url inside [] into special token <html>
	html = re.compile(r'\[.*\]')
	text = re.sub(html, "<url> ", text)
	# convert html url into special token <htnml>
	exp = re.compile(r'''(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)\/?$''', re.X)
	text = re.sub(exp, "<url> ", text)
	# remove filename with 3 types .zip, .txt, .exe
	zip_txt_exe = re.compile(r'\S+.zip')
	text = re.sub(zip_txt_exe, "<file> ", text)
	zip_txt_exe = re.compile(r'\S+.xlsx')
	text = re.sub(zip_txt_exe, "<file> ", text)
	zip_txt_exe = re.compile(r'\S+.exe')
	text = re.sub(zip_txt_exe, "<file> ", text)
	zip_txt_exe = re.compile(r'\S+.txt')
	text = re.sub(zip_txt_exe, "<file> ", text)
	zip_txt_exe = re.compile(r'\S+.html')
	text = re.sub(zip_txt_exe, "<file> ", text)
	# convert file name such as bin.exe, etc into special token <file_name>
	filename = re.compile(r'\s+([A-Za-z]+\w*\.)+(\[A-Za-z]+\w*\.)*(\w+)')
	text = re.sub(filename, " <file> ", text)
	# convert version number into special token <version>, whatever not filename 
	# and has w.w.w.w will be turned into <version>
	version = re.compile(r'\s+\w?(\d+.)+\d+')
	text = re.sub(version, " <version> ", text)
	# remove hours minutes for time, not useful
	time = re.compile(r'\s*(\d+:)+(\d+)*')
	text = re.sub(time, "", text)
	# remove bullet point minutes for time, not useful
	number_bullet_point = re.compile(r'\s+(\d+.)\s+')
	text = re.sub(number_bullet_point, "", text)
	# need to remove bracket <br> </br> <a> </a> etc....
	return text

def main():
	# usage: python3 clean_up_doc.py training_dev_technotes.json cleaned_training_dev_technotes.json
	# opening the json file
	file = open(sys.argv[1], "r")

	# return json object (list of dictionary)
	technotes = json.load(file)

	print(technotes["swg21383270"]["text"])

	# iterating each document in the technotes.
	# for doc in technotes.keys():
	# 	# for each doc, we will clean up the text attribute. This will iterate through document and clean it.
	# 	# and store it back to text attribute.
	# 	text = technotes[doc]["text"]
	# 	text = clean_text(text)
	# 	technotes[doc]["text"] = text
	# # close the file
	# file.close()

	# dump cleaned data to new json file
	# with open(sys.argv[2], 'w') as outfile:
	# 	formatted_json = json.dumps(technotes, indent=2)
	# 	outfile.write(formatted_json)


	print("**********************************************************************")
	print(technotes["swg21383270"]["text"])

# call main for function to start:
if __name__ == '__main__':
	main()