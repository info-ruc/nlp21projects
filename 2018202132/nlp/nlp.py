from stanfordcorenlp import StanfordCoreNLP


class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=30000)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

if __name__ == '__main__':
    SNLP = StanfordNLP()
    text = 'he will be her in china.'
    tree = SNLP.parse(text)