from math import log
from operator import itemgetter
import re
import pandas as pd

def createDataSet():
	mydata = ""
	with open('D:/ML-Practicles/Decision-Tree/lenses.txt') as f:
		mydata = re.split(r'\n',f.read())

	mydata.remove('')
	final = ""
	final_list = []

	for i in range(len(mydata)):
		final = mydata[i]
		final_list.append((re.findall(r'[\d]',final)))

	labels = ['Age', 'Spectacle_Prescription', 'astigmatic', 'Tear_Rate', 'Lenses']

	return final_list,labels

def majoritycnt(classList):
	classCount = {}

	for vote in classList:

		if vote not in classCount.keys():
			classCount[vote] = 1

		else:
			classCount[vote] += 1

	sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def calcshannonEnt(dataset):
	d = {}
	length = len(dataset)
	shannon_ent = 0
	for feature in dataset:
		label = feature[-1]

		if label not in d.keys():
			d[label] = 1
		else:
			d[label] += 1

	for k in d.keys():
		prob = float(d[k] / length)
		shannon_ent -= prob * log(prob,2)

	return shannon_ent

def splitdataset(dataset, axis, value):
	a = list()
	for feature in dataset:
		if feature[axis] == value:
			reducevect = feature[:axis]
			reducevect.extend(feature[axis+1:])
			a.append(reducevect)

	return a

def selectbestFeature(dataset):
	old_entropy = calcshannonEnt(dataset)
	best_feature = -1
	best_info_gain = 0.0
	num_feature = len(dataset[0]) - 1

	for i in range(num_feature):
		example = []
		for feature in dataset:
			example.append(feature[i])
		unique_val = set(example)
		new_entropy = 0.0

		for value in unique_val:
			sub_dataset = splitdataset(dataset, i, value)
			prob = float(len(sub_dataset)) / float(len(dataset))
			new_entropy += prob * calcshannonEnt(sub_dataset)

		info_gain = old_entropy - new_entropy

		if info_gain > best_info_gain:
			best_info_gain = info_gain
			best_feature = i

	return best_feature

def createTree(dataset, labels_test):
	classList = []
	for i in dataset:
		classList.append(i[-1])

	if classList.count(classList[0]) == len(dataset):
		return classList[0]

	if len(dataset[0]) == 1:
		return majoritycnt(classList)

	bestFeature = selectbestFeature(dataset)
	bestFeatureLabel = labels[bestFeature]

	myTree = {bestFeatureLabel:{}}
	del(labels_test[bestFeature])

	featValues = []
	for f in dataset:
		featValues.append(f[bestFeature])

	uniqueValues = set(featValues)

	for value in uniqueValues:
		subLabels = labels_test[:]
		myTree[bestFeatureLabel][value] = createTree(splitdataset(dataset, bestFeature, value), subLabels)

	return myTree

#test using tree for classification
def classify(myTree, labels, testVec):
	featLabel = ""
	for k in myTree.keys():
		featLabel = k
		break

	second_dir = myTree[featLabel]
	feature = labels.index(featLabel)
	value = testVec[feature]
	classLabel = ''

	for k in second_dir.keys():

		if value == int(k):
			if type(second_dir[k]).__name__ == 'dict':
				classLabel = classify(second_dir[k], labels, testVec)
			else:
				classLabel = second_dir[k]
	return classLabel

dataset, labels = createDataSet()
myTree = createTree(dataset, labels)

answer = classify(myTree, ['Age', 'Spectacle_Prescription', 'astigmatic', 'Tear_Rate', 'Lenses'], [3,1,1,1])
if answer == "3":
	print("hard lense")

if answer == "2":
	print("no lense")

if answer == "1":
	print("soft lense")

	
