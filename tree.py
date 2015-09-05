# CSE6242/CX4242 Homework 4 Pseudocode
# You can use this skeleton code for Task 1 of Homework 4.
# You don't have to use this code. You can implement your own code from scratch if you want.

import csv, math
from random import shuffle

class TreeNode:
	def __init__(self, parent=None):
		self.parent = parent
		self.children = {}
		self.split_attr = None
		self.answer = None

# Implement your decision tree below
#class DecisionTree():
	#tree = {}

def learn(training_set, attr_dict, root=None):
	#self.tree = {}
	#tree_index = 0
	threshold = 0.00001
	default_val = majorityVote(training_set)

	if root == None:
		root = TreeNode(None)

	#base case when all data has the same label
	if same_label_data(training_set):
		root.answer = training_set[0][-1]
		return root

	#print len(attr_dict)
	#if training_set is empty or attr_dict is empty
	if len(attr_dict) <= 0:
		root.answer = default_val
		return root

	#get best attribute to split on
	best_attr = get_best_attr(training_set, attr_dict, threshold)
	# -1 is the return value when comparing with threshold
	if best_attr == -1:
		root.answer = default_val
		return root
	else:
		best_attr_idx = attr_dict[best_attr]
		#print "best attr idx ", best_attr, best_attr_idx
		subsets = split(training_set,best_attr_idx)
		root.split_attr = best_attr_idx
		root.data = None
		attr_dict.pop(best_attr, None)
		#print len(attr_dict)
		for key in subsets.keys():
			child = TreeNode(root)
			#child.split_attr_val = subsets[key]
			root.children[key] = child
			learn(subsets[key], attr_dict, child)
		return root

def classify(test_row, tree):
	#result = "no" # baseline: always classifies as no
	curr_tree = tree
	result = tree.answer
	while result == None:
		if tree.split_attr >= len(test_row) or test_row[curr_tree.split_attr] not in curr_tree.children:
			return None
		curr_tree = curr_tree.children[test_row[curr_tree.split_attr]]
		result = curr_tree.answer
	return result


#Calculates entropy for a given data set
def calc_entropy(training_set):
	total_rows = len(training_set)
	#Making a dictionary for the values
	y_val = {}
	ent = 0.0
	for row in training_set:
		y_val.setdefault(row[-1],0)
		y_val[row[-1]] += 1

	#print y_val.keys()
	#print y_val.values()
	for val in y_val.values():
		propor = val/ float(total_rows)
		ent = ent - (propor * math.log(propor))
	return ent

#Calculates gini index for a given data set	
def calc_gini_index(training_set):
	total_rows = len(training_set)
	#Making a dictionary for the values
	y_val = {}
	gini = 0.0
	for row in training_set:
		y_val.setdefault(row[-1],0)
		y_val[row[-1]] += 1

	for val in y_val.values():
		propor = val/ float(total_rows)
		gini += (propor**2)
	gini = 1 - gini
	#print gini
	return gini

#Calculates information gain based on entropy
def information_gain(attr_mapped_name, training_set, threshold):
	entropy = calc_entropy(training_set)
	attr_vals = {}
	total_rows = len(training_set)
	entropy_vals = 0.0
	# for different values of this attr, make a list of corresponding rows.
	for row in training_set:
		attr_vals.setdefault(row[attr_mapped_name], [])
		attr_vals[row[attr_mapped_name]].append(row)

	for val in attr_vals.values():
		#Calculate entropy for each value the attribute can take
		ent_val = calc_entropy(val)
		p = len(val) / float(total_rows)
		entropy_vals += p*ent_val
	info_gain = entropy - entropy_vals
	#print info_gain
	return info_gain

#Calculates information gain based on gini index
def information_gain_gini(attr_mapped_name, training_set, threshold):
	gini_index = calc_gini_index(training_set)
	#print "gini", gini_index
	attr_vals = {}
	total_rows = len(training_set)
	gini_vals = 0.0
	for row in training_set:
		attr_vals.setdefault(row[attr_mapped_name], [])
		attr_vals[row[attr_mapped_name]].append(row)

	for val in attr_vals.values():
		g_val = calc_gini_index(val)
		p = len(val) / float(total_rows)
		gini_vals += p*g_val
	info_gain = gini_index - gini_vals
	#print info_gain
	return info_gain


def get_best_attr(training_set, attr_dict, threshold):
	info_gain_attr_dict = {}
	for attr in attr_dict.keys():
		#print attr, attr_dict[attr]
		info_gain_attr_dict[attr] = 0

		#calculate information gain based on entropy
		info_gain_attr_dict[attr] = information_gain(attr_dict[attr], training_set, threshold)

		#Calculate information gain based on gini impurity
		#info_gain_attr_dict[attr] = information_gain_gini(attr_dict[attr], training_set, threshold)

	#print info_gain_attr_dict
	attr_max_gain = max(info_gain_attr_dict, key=info_gain_attr_dict.get)
	if info_gain_attr_dict[attr_max_gain] < threshold:
		return -1
	else:
		return attr_max_gain

#when splitting stopped , return the label with the maximum votes to label the leaf
def majorityVote(training_set):
	d = {}
	for row in training_set:
		d.setdefault(row[-1],0)
		d[row[-1]] += 1
	max = 0
	for key,val in d.items():
		if val > max:
			max = val
			vote = key
	#print vote
	return vote

#To check if all the given data has the same label value
def same_label_data(data):
	count = len(data)
	d = {}
	for row in data:
		d.setdefault(row[-1],0)
		d[row[-1]] += 1
	#print len(set(d.keys()))
	return (len(set(d.keys())) <= 1)

def split(training_set, attribute):
	subsets = {}
	for row in training_set:
		subsets.setdefault(row[attribute] , [])
		subsets[row[attribute]].append(row)
	return subsets

def run_decision_tree():
	# Load data set
	with open("hw4-data.tsv") as tsv:
	#with open("test.tsv") as tsv:
		data = [tuple(line) for line in csv.reader(tsv, delimiter="\t")]
	print "Number of records: %d" % len(data)
	accuracy_overall = []
	# Split training/test sets
	# You need to modify the following code for cross validation.

	#Shuffle the data. Will change the building of decision tree if a lot of same labels are continuous in the dataset.
	shuffle(data)
	K = 10
	for check in range (0,K):
		training_set = [x for i, x in enumerate(data) if i % K != check]
		test_set = [x for i, x in enumerate(data) if i % K == check]

		attrs = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month','day_of_week', 'duration','campaign', 'pdays','previous','poutcome','emp.var.rate','cons.proce.idx','cons.conf.idx','euribor3m','nr.employed']
		attr_dict = {}
		num = 0
		for attr in attrs:
			attr_dict[attr] = num
			num += 1
		#tree = DecisionTree()
		# Construct a tree using training set
		tree = learn( training_set, attr_dict, None )

		# Classify the test set using the tree we just constructed
		results = []
		for instance in test_set:
			result = classify( instance[:-1], tree )
			#print "result is", result
			results.append( result == instance[-1] )

		# Accuracy
		accuracy = float(results.count(True))/float(len(results))
		#print "accuracy: %.4f" % accuracy
		accuracy_overall.append(accuracy)

	sum=0.0
	for acc in accuracy_overall:
		sum += acc
	accuracy_f = sum/float(len(accuracy_overall))
	# Writing results to a file (DO NOT CHANGE)
	f = open("result.txt", "w")
	f.write("accuracy: %.4f" % accuracy_f)
	f.close()


if __name__ == "__main__":
	run_decision_tree()