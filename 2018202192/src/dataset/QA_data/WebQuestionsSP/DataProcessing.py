import json

train_file = "./WebQSP.train.json"

with open(train_file, "r") as train_obj:
	train_data = json.load(train_obj)

output_file = open("qa_train_webqsp.txt", "w")

for question in train_data["Questions"]:
	if question["Parses"][0]["TopicEntityMid"] is None:
		continue
	output_file.write(question["ProcessedQuestion"] + " ")
	num_parses = len(question["Parses"])
	output_file.write("[" + question["Parses"][num_parses - 1]["TopicEntityMid"] + "]\t")
	answers = []
	for parse in question["Parses"]:
		for answer in parse["Answers"]:
			if answer["AnswerType"] == "Entity":
				answers.append(answer["AnswerArgument"])
	first = True
	for answer in answers:
		if not first:
			output_file.write("|")
		output_file.write(answer)
		first = False
	output_file.write("\n")

output_file.close()


test_file = "./WebQSP.test.json"

with open(test_file, "r") as test_obj:
	test_data = json.load(test_obj)

output_file = open("qa_test_webqsp.txt", "w")

for question in test_data["Questions"]:
	if question["Parses"][0]["TopicEntityMid"] is None:
		continue
	output_file.write(question["ProcessedQuestion"] + " ")
	num_parses = len(question["Parses"])
	output_file.write("[" + question["Parses"][num_parses - 1]["TopicEntityMid"] + "]\t")
	answers = []
	for parse in question["Parses"]:
		for answer in parse["Answers"]:
			if answer["AnswerType"] == "Entity":
				answers.append(answer["AnswerArgument"])
	first = True
	for answer in answers:
		if not first:
			output_file.write("|")
		output_file.write(answer)
		first = False
	output_file.write("\n")

output_file.close()
