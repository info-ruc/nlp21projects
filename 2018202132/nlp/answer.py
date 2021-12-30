from nltk.tree import Tree as Tree
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nlp import *


SNLP = StanfordNLP()


class Answer:
    def __init__(self):
        self.stemmer = 0
        self.remove_punctuation_map = 0
        self.vectorizer = 0

    def parse_text(filename):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        fp = open(filename)
        data = fp.read()
        sentences = tokenizer.tokenize(data)
        sentences = [si for si in sentences if "\n" not in si]
        sentences = [s.encode('ascii', 'ignore') for s in sentences]
        return sentences

    def answer(self, text, filename):
        tree = SNLP.parse(text)
        tree = Tree.fromstring(str(tree))
        if not self.is_q(tree):
            print("This question is not binary.")
        sentences = self.parse_text(filename)
        print(sentences)
        return

    def get_raw_answer(self, question, answer):
        q_tree = SNLP.parse(question)
        q_tree = Tree.fromstring(str(q_tree))
        a_tree = SNLP.parse(answer)
        a_tree = Tree.fromstring(str(a_tree))
        (q_top_level_structure, q_parse_by_structure) = self.get_structure(q_tree)
        (a_top_level_structure, a_parse_by_structure) = self.get_structure(a_tree)
        for i in range(0, len(q_top_level_structure)):
            q_label = q_top_level_structure[i]
            if q_label in a_top_level_structure:
                a_index = a_top_level_structure.index(q_label)
            else:
                print("label not found")
                return False
            if not self.partial_matching(q_parse_by_structure[i], a_parse_by_structure[a_index]):
                return False
        return True

    def partial_matching(self, question_list, answer_list):
        question_list = [word for word in question_list if word not in stopwords.words('english')]
        answer_list = [word for word in answer_list if word not in stopwords.words('english')]
        synonyms_list = []
        antonyms_list = []
        for word in question_list:
            synonyms_list.append(word)
            (synonyms, antonyms) = self.get_syn_ant(word)
            synonyms_list += synonyms
            antonyms_list += antonyms
        for word in answer_list:
            if not word in synonyms_list:
                return False
            if word in antonyms_list:
                return False
        return True

    @staticmethod
    def is_q(tree):
        for s in tree.subtrees(lambda t: t.label() == "SQ"):
            return True
        return False

    @staticmethod
    def get_structure(tree):
        top_level_structure = []
        parse_by_structure = []
        for t in tree.subtrees(lambda t: t.label() == "SQ"):
            for tt in t:
                top_level_structure.append(tt.label())
                parse_by_structure.append(tt.leaves())
            return top_level_structure, parse_by_structure

    @staticmethod
    def get_syn_ant(word):
        synonyms = []
        antonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
            for l in syn.hypernyms():
                synonyms.append(l.name().split(".")[0])
        return synonyms, antonyms