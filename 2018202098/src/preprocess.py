import random
import json


def read_dataset_from_text(filename):
    question_list, answers_list = [], []
    f = open(filename)
    for line in f:
        line = line.strip()
        if not line:
            continue
        question, answers = line.split("->")
        question = question.strip()
        answers = [answer.strip() for answer in answers.split("、")]
        question_list.append(question)
        answers_list.append(answers)
    return question_list, answers_list


def encode_question_answers_into_dict(question_list, answers_list):
    assert len(question_list) == len(answers_list), "Should have equal length"
    dataset = []
    for idx, (question, answers) in enumerate(zip(question_list, answers_list)):
        item = {"id": str(idx), "question": question, "answer": answers}
        dataset.append(item)
    return dataset


def split_dataset(dataset, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    assert train_ratio + dev_ratio + test_ratio == 1.0, "Sum should equal to 1"
    random.shuffle(dataset)
    train_idx = int(len(dataset) * train_ratio)
    dev_idx = int(len(dataset) * (train_ratio + dev_ratio))
    train_dataset = dataset[:train_idx]
    dev_dataset = dataset[train_idx:dev_idx]
    test_dataset = dataset[dev_idx:]
    return train_dataset, dev_dataset, test_dataset


if __name__ == "__main__":
    debug = True
    filename = "dataset/chinese-common-sense-qa.txt"
    output_filename_list = ["data/train.json", "data/dev.json", "data/test.json"]
    question_list, answers_list = read_dataset_from_text(filename)
    dataset = encode_question_answers_into_dict(question_list, answers_list)
    train_dataset, dev_dataset, test_dataset = split_dataset(dataset)
    dataset_list = [train_dataset, dev_dataset, test_dataset]
    if debug:
        print([len(i) for i in dataset_list])
    for name, item in zip(output_filename_list, dataset_list):
        with open(name, "w") as f:
            json.dump(item, f, ensure_ascii=False, indent=4)
