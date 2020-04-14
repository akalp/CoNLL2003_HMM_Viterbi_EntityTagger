import numpy as np


class HMM:
    def __init__(self, sentences, entities, poses=None, chunks=None):
        super().__init__()
        self.sentences, self.poses, self.chunks, self.entities = sentences, poses, chunks, entities
        self.all_words, self.all_poses, self.all_chunks, self.all_entities = self.merge_lists()

        self.n_init_entity = self.init_entity_count()
        self.init_prob, self.s_init_prob = self.cal_init_prob()

        self.n_entity_word = self.entity_word_count()
        self.emis_prob, self.s_emis_prob = self.cal_emission_prob()

        self.n_bi_en = self.bi_en_count()
        self.trans_prob, self.s_trans_prob = self.cal_trans_prob()

    def entity_word_count(self):
        n_entity_word_dict = dict()

        for e, w in zip(self.all_entities, self.all_words):
            if e not in n_entity_word_dict.keys():
                n_entity_word_dict[e] = dict()
            n_entity_word_dict[e][w] = 1 if w not in n_entity_word_dict[e].keys() else n_entity_word_dict[e][w] + 1

        return n_entity_word_dict

    def cal_emission_prob(self):
        res = dict()
        s_res = dict()

        for e in self.n_entity_word.keys():
            if e not in res.keys():
                res[e] = dict()
                s_res[e] = dict()

            total = sum(self.n_entity_word[e].values())
            for w in self.n_entity_word[e].keys():
                res[e][w] = self.n_entity_word[e][w] / total
                s_res[e][w] = (self.n_entity_word[e][w] + 1) / (total + len(set(self.all_words)))
            s_res[e]['NaN'] = 1 / (total + len(set(self.all_words)))
        return res, s_res

    def cal_trans_prob(self):
        res = dict()
        s_res = dict()

        for e1 in self.n_bi_en.keys():
            if e1 not in res.keys():
                res[e1] = dict()
                s_res[e1] = dict()
            total = sum(self.n_bi_en[e1].values())
            for e2 in self.n_bi_en[e1].keys():
                res[e1][e2] = self.n_bi_en[e1][e2] / total
                s_res[e1][e2] = (self.n_bi_en[e1][e2] + 1) / (total + len(set(self.all_entities)))

        return res, s_res

    def bi_en_count(self):
        bi_entity = dict()

        unique_ent = list(set(self.all_entities))
        for ent in unique_ent:
            bi_entity[ent] = dict()
            for ent2 in unique_ent:
                bi_entity[ent][ent2] = 0

        for entity_list in self.entities:
            bi_list = zip(*[entity_list[i:] for i in range(2)])
            for bi in bi_list:
                bi_entity[bi[0]][bi[1]] = 1 if bi[1] not in bi_entity[bi[0]].keys() else bi_entity[bi[0]][bi[1]] + 1

        return bi_entity

    def cal_init_prob(self):
        total = sum(self.n_init_entity.values())
        res = dict()
        s_res = dict()
        for key in self.n_init_entity.keys():
            res[key] = self.n_init_entity[key] / total
            s_res[key] = (self.n_init_entity[key] + 1) / (total + 9)
        return res, s_res

    def init_entity_count(self):
        res = dict()
        unique_ent = list(set(self.all_entities))
        for ent in unique_ent:
            res[ent] = 0

        for entity_list in self.entities:
            res[entity_list[0]] += 1

        return res

    def merge_lists(self):
        all_words = []
        all_pos = []
        all_chunk = []
        all_entities = []

        for se in self.sentences:
            for w in se:
                all_words.append(w)
        if self.poses is not None:
            for po in self.poses:
                for p in po:
                    all_pos.append(p)
        if self.chunks is not None:
            for ch in self.chunks:
                for c in ch:
                    all_chunk.append(c)
        for en in self.entities:
            for e in en:
                all_entities.append(e)

        return all_words, all_pos, all_chunk, all_entities


from math import log2


def viterbi(hmm, t_sentences):
    res = list()
    states = list(hmm.n_init_entity.keys())

    for sent in t_sentences:
        D = np.zeros([len(states), len(sent)])
        E = np.zeros([len(states), len(sent) - 1])

        for s in range(len(states)):
            init = hmm.init_prob[states[s]] if hmm.init_prob[states[s]] != 0 else hmm.s_init_prob[states[s]]
            emis = hmm.emis_prob[states[s]][sent[0]] if sent[0] in hmm.emis_prob[states[s]].keys() else \
                hmm.s_emis_prob[states[s]]["NaN"]
            D[s, 0] = log2(init) + log2(emis)

        for w in range(1, len(sent)):
            for i in range(len(states)):
                temp = np.zeros(len(states))
                past = D[:, w - 1]
                for j in range(len(states)):
                    trans = hmm.trans_prob[states[j]][states[i]] if hmm.trans_prob[states[j]][states[i]] != 0 else \
                        hmm.s_trans_prob[states[j]][states[i]]
                    temp[j] = log2(trans) + past[j]

                emis = hmm.emis_prob[states[i]][sent[w]] if sent[w] in hmm.emis_prob[states[i]].keys() else \
                    hmm.s_emis_prob[states[i]]["NaN"]
                D[i, w] = np.amax(temp) + log2(emis)

        t_res = list()
        for i in range(len(sent)):
            t_res.append(states[np.argmax(D[:, i])])

        res.append(t_res)
    return res


def dataset(conll_file):
    sent = []
    pos = []
    chunk = []
    entity = []
    temp_sent = []
    temp_pos = []
    temp_chunk = []
    temp_entity = []

    with open(conll_file) as f:
        conll_raw_data = f.readlines()
    conll_raw_data = [x.strip() for x in conll_raw_data]

    for line in conll_raw_data:
        if line != '':
            split_line = line.split()
            if len(split_line) == 4:
                if split_line[0] != '-DOCSTART-':
                    temp_sent.append(split_line[0].lower())
                    temp_pos.append(split_line[1])
                    temp_chunk.append(split_line[2])
                    temp_entity.append(split_line[3])
            else:
                raise IndexError('Line split length does not equal 4.')
        else:
            if len(temp_sent) > 0:
                assert (len(sent) == len(pos))
                assert (len(sent) == len(chunk))
                assert (len(sent) == len(entity))
                sent.append(temp_sent)
                pos.append(temp_pos)
                chunk.append(temp_chunk)
                entity.append(temp_entity)
                temp_sent = []
                temp_pos = []
                temp_chunk = []
                temp_entity = []

    return sent, pos, chunk, entity


def accuracy(test, res):
    true = 0
    total = 0

    test = [item for sublist in test for item in sublist]
    res = [item for sublist in res for item in sublist]

    for t, r in zip(test, res):
        if t == r:
            true += 1
        total += 1

    print(true / total)


sentences, _, _, entities = dataset('../data/train.txt')
test_sentences, _, _, test_entities = dataset('../data/test.txt')

hmm = HMM(sentences=sentences, entities=entities)

res = viterbi(hmm=hmm, t_sentences=test_sentences)

accuracy(test=test_entities, res=res)
