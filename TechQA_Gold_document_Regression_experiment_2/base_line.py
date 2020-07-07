from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import sys

def main():
	ans_indicator_list = ["CVSS", "The", "ANSWE", "RESOL", "VULNE", "This", "CAUSE"]
	train = open(sys.argv[1], "r")
	dev = open(sys.argv[2], "r")
	train = train.readlines()
	dev = dev.readlines()
	train_preds =[]
	train_labels =[]
	dev_preds = []
	dev_labels = []
	for instance in train:
		instance = instance.split("\t")
		label = int(instance[0]) # convert to integer
		train_labels.append(label)
		ans_candidate = instance[2].lstrip()
		pred = 0
		for ans_indicator in ans_indicator_list:
			if ans_candidate.startswith(ans_indicator):
				pred = 1
				break
		train_preds.append(pred)
	for instance in dev:
		instance = instance.split("\t")
		label = int(instance[0])
		dev_labels.append(label)
		ans_candidate = instance[2].lstrip()
		pred = 0
		for ans_indicator in ans_indicator_list:
			if ans_candidate.startswith(ans_indicator):
				pred = 1
				break
		dev_preds.append(pred)

	train_f1 = f1_score(train_labels, train_preds)
	train_r = recall_score(train_labels, train_preds)
	train_p = precision_score(train_labels, train_preds)
	train_acc = accuracy_score(train_labels, train_preds)
	print(f'train f1:{train_f1:.3f} / train recall: {train_r:.3f} / train precision: {train_p:.3f} / train acc: {train_acc:.3f}')
	dev_f1 = f1_score(dev_labels, dev_preds)
	dev_r = recall_score(dev_labels, dev_preds)
	dev_p = precision_score(dev_labels, dev_preds)
	dev_acc = accuracy_score(dev_labels, dev_preds)
	print(f'dev f1:{dev_f1:.3f} / dev recall: {dev_r:.3f} / dev precision: {dev_p:.3f} / dev acc: {dev_acc:.3f}')

	# Results from not doing anything.
	# train f1:0.274 / train recall: 0.358 / train precision: 0.222 / train acc: 0.803
	# dev f1:0.229 / dev recall: 0.362 / dev precision: 0.168 / dev acc: 0.797
	
	# Results from without "This"
	# train f1:0.285 / train recall: 0.319 / train precision: 0.256 / train acc: 0.833
	# dev f1:0.254 / dev recall: 0.347 / dev precision: 0.201 / dev acc: 0.830  

	# Results from without "This" + "The"
	# train f1:0.342 / train recall: 0.246 / train precision: 0.563 / train acc: 0.902
	# dev f1:0.329 / dev recall: 0.248 / dev precision: 0.490 / dev acc: 0.916

main()	
