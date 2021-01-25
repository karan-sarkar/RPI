import io
import conllu
import numpy
import tqdm
import operator

class conllu_file:
    def __init__(self, file_name):
        data_file = io.open(file_name, 'r', encoding='utf-8')
        self.sentences = []
        print('Started Sentences')
        for sentence in tqdm.tqdm(conllu.parse_incr(data_file)):
            self.sentences.append(sentence)
        print('Finished Sentences')

def test(tagger, file):
    total = 0
    matches = 0
    print('Started Testing')
    for sentence in tqdm.tqdm(file.sentences):
        total += len(sentence)
        preds = tagger.predict(sentence)
        matches += sum([preds[i] == sentence[i][tagger.tag] for i in range(len(sentence))])
    print('Finished Testing')
    return matches / total

class hmm_tagger:
    
    def __init__(self, file, token, tag):
        self.sentences = file.sentences
        self.token = token
        self.tag = tag
        
        self.tags = {'START': 0, 'END': 1}
        self.tag_list = ['START', 'END']
        print('Started Tags')
        for sentence in tqdm.tqdm(self.sentences):
            for word in sentence:
                if word[tag] not in self.tags.keys():
                    self.tags[word[tag]] = len(self.tags)
                    self.tag_list.append(word[tag])
        print('Finished Tags')
        
        self.words = {}
        print('Started Words')
        for sentence in tqdm.tqdm(self.sentences):
            for word in sentence:
                if word[token] not in self.words.keys():
                    self.words[word[token]] = len(self.words)
        print('Finished Words')
        
        self.tag_counts = numpy.zeros(len(self.tags))
        self.trans_probs = numpy.zeros((len(self.tags), len(self.tags)))
        self.emiss_probs = numpy.zeros((len(self.words), len(self.tags)))
        self.avg_emis_probs = numpy.zeros(len(self.tags))
    
    def predict(self, sentence):
        probs = numpy.zeros(len(self.tags))
        probs[self.tags['START']] = 1
        tag_seqs = [[] for i in range(len(self.tags))]
        for word in sentence:
            probs = self.trans_probs * probs[:, numpy.newaxis]
            if word[self.token] in self.words.keys():
                probs = probs * self.emiss_probs[self.words[word[self.token]], :]
            elif word[self.token][0].isdigit():
                probs = probs * (self._fix('NUM') if self.tag == 'upostag' else self._fix('CD'))
            elif '.' in word[self.token]:
                probs = probs * (self._fix('X') if self.tag == 'upostag' else self._fix('ADD'))
            elif word[self.token] == '-':
                probs = probs * (self._fix('PUNCT') if self.tag == 'upostag' else self._fix('NFP'))
            elif word[self.token].isupper() and len(tag_seqs) != 0:
                 probs = probs * (self._fix('PROPN') if self.tag == 'upostag' else self._fix('NNP'))
            else:
                probs = probs * self.avg_emis_probs
            indices = numpy.argmax(probs, axis = 0)
            probs = numpy.amax(probs, axis = 0)
            tag_seqs = [tag_seqs[indices[i]] + [self.tag_list[i]] for i in range(len(indices))]
        return tag_seqs[numpy.argmax(probs)]
    
    def _fix(self, tag):
        probs = numpy.zeros(len(self.tags))
        probs[self.tags[tag]] = 1
        return probs
    
    def train(self):
        self._count_tags()
        self._calc_trans_probs()
        self._calc_emiss_probs()
      
    def _calc_emiss_probs(self):
        print('Started Emission Probabilities')
        for sentence in tqdm.tqdm(self.sentences):
            for word in sentence:
                self.emiss_probs[self.words[word[self.token]], self.tags[word[self.tag]]] += 1
        print('Finished Emission Probabilities')
        self.emiss_probs = self.emiss_probs / self.tag_counts
        self.avg_emis_probs = numpy.mean(self.emiss_probs, axis = 0)
        closed = ['DET', 'DT', 'PRON', 'PRP', 'AUX', 'MD', 'PUNCT', 'EX' ]
        for tag in closed:
            if tag in self.tags.keys():
                self.avg_emis_probs[self.tags[tag]] = 0

    def _calc_trans_probs(self):
        print('Started Transition Probabilities')
        for sentence in tqdm.tqdm(self.sentences):
            self.trans_probs[self.tags['START'], self.tags[sentence[0][self.tag]]] += 1
            self.trans_probs[self.tags[sentence[-1][self.tag]], self.tags['END']] += 1
            for i in range(len(sentence) - 1):
                self.trans_probs[self.tags[sentence[i][self.tag]], self.tags[sentence[i + 1][self.tag]]] += 1
        print('Finished Transition Probabilities')
        self.trans_probs = self.trans_probs / self.tag_counts[:, numpy.newaxis]

    def _count_tags(self):
        print('Started Tags Counts')
        for sentence in tqdm.tqdm(self.sentences):
            self.tag_counts[self.tags['START']] += 1
            self.tag_counts[self.tags['END']] += 1
            for word in sentence:
                self.tag_counts[self.tags[word[self.tag]]] += 1
        print('Finished Tags Counts')
    


if __name__== '__main__' :
    file = conllu_file('en-ud-train.conllu')
    tagger = hmm_tagger(file, 'form', 'upostag')
    tagger.train()
    file = conllu_file('en-ud-dev.conllu')
    print(test(tagger, file))
