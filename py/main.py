# -*- coding: utf-8 -*-
import numpy as np
import math


class HMM:
    def __init__(self, sentences, entities, poses=None, chunks=None):
        """
        Inıtializer of the HMM model. Takes 4 parameters, two of them is necessary which are sentences and entities.
        :param sentences: List of sentences which is list of words.
        :param entities: List of entity tags of each sentence.
        :param poses: List of pos tags of each sentence.
        :param chunks: List of chunk tags of each sentence.
        """
        super().__init__()
        self.sentences, self.poses, self.chunks, self.entities = sentences, poses, chunks, entities
        self.all_words, self.all_poses, self.all_chunks, self.all_entities = self.merge_lists()
        self.unique_entity_count = len(set(self.all_entities))
        self.unique_word_count = len(set(self.all_words))

        self.n_init_entity = self.init_entity_count()
        self.init_prob, self.s_init_prob = self.cal_init_prob()

        self.n_entity_word = self.entity_word_count()
        self.emis_prob, self.s_emis_prob = self.cal_emission_prob()

        self.n_bi_en = self.bi_en_count()
        self.trans_prob, self.s_trans_prob = self.cal_trans_prob()

    def entity_word_count(self):
        """
        This function traveses list of word-entity pairs and counts them.
        :return: Word counts for each entity tag as dict, for example: {"O":{"car":2,"fire":3,...},...}
        """
        n_entity_word_dict = dict()

        for e, w in zip(self.all_entities, self.all_words):
            if e not in n_entity_word_dict.keys():
                n_entity_word_dict[e] = dict()
            n_entity_word_dict[e][w] = 1 if w not in n_entity_word_dict[e].keys() else n_entity_word_dict[e][w] + 1

        return n_entity_word_dict

    def cal_emission_prob(self):
        """
        This function calculates the normal and smoothed probabilities of the selection of the word in the word set of each entity tag.
        :return: Normal and Smoothed Probabilities dicts as tuple in order.

        Example return values:
        Normal: {"O":{"car":0.2,"fire":0.03,...},...}
        Smoothed: {"O":{"car":0.18,"fire":0.025,...},...}
        """
        res = dict()
        s_res = dict()

        for e in self.n_entity_word.keys():
            # Initialize the keys and dicts
            if e not in res.keys():
                res[e] = dict()
                s_res[e] = dict()

            total = sum(self.n_entity_word[e].values())
            for w in self.n_entity_word[e].keys():
                res[e][w] = self.n_entity_word[e][w] / total
                s_res[e][w] = (self.n_entity_word[e][w] + 1) / (total + self.unique_word_count)
            s_res[e]['NaN'] = 1 / (total + self.unique_word_count)
        return res, s_res

    def cal_trans_prob(self):
        """
        This function calculates the normal and smoothed probabilities of which entity tag comes after another entity tag.
        :return: Normal and Smoothed Probabilities dicts as tuple in order.

        Example return values:
        Normal: {"O":{"B-PER":0.2,"O":0.03,...},...}
        Smoothed: {"O":{"B-PER":0.18,"O":0.025,...},...}
        """
        res = dict()
        s_res = dict()

        for e1 in self.n_bi_en.keys():
            # Initialize the keys and dicts
            if e1 not in res.keys():
                res[e1] = dict()
                s_res[e1] = dict()
            total = sum(self.n_bi_en[e1].values())
            for e2 in self.n_bi_en[e1].keys():
                res[e1][e2] = self.n_bi_en[e1][e2] / total
                s_res[e1][e2] = (self.n_bi_en[e1][e2] + 1) / (total + self.unique_entity_count)

        return res, s_res

    def bi_en_count(self):
        """
        This function traverses entities list and creates biagrams of entities for each sentence, then counts biagrams.
        :return: Biagram counts as dict, for example: {"O":{"O":10,...},"B-PER":{"O":8,...}}
        """
        bi_entity = dict()

        unique_ent = list(set(self.all_entities))
        # Initialize the keys and dicts and initial values of leaves
        for ent in unique_ent:
            bi_entity[ent] = dict()
            for ent2 in unique_ent:
                bi_entity[ent][ent2] = 0

        for entity_list in self.entities:
            bi_list = zip(*[entity_list[i:] for i in range(2)])     # Creates list of biagrams
            for bi in bi_list:
                bi_entity[bi[0]][bi[1]] += 1

        return bi_entity

    def cal_init_prob(self):
        """
        Thic function calculates the normal and smoothed probabilities of which tag the sentences have in the first index.
        :return: Normal and Smoothed Probabilities dicts as tuple in order.

        Example return values:
        Normal: {"B-PER":0.2,"O":0.03,...}
        Smoothed: {"B-PER":0.18,"O":0.025,...}
        """
        total = sum(self.n_init_entity.values())
        res = dict()
        s_res = dict()
        for key in self.n_init_entity.keys():
            res[key] = self.n_init_entity[key] / total
            s_res[key] = (self.n_init_entity[key] + 1) / (total + self.unique_entity_count)
        return res, s_res

    def init_entity_count(self):
        """
        This function traverses entities list which contains lists of entities of the training sentences.
        Counts the first entities of each sentence.
        :return: Counts of entities of the first words of sentences as dict -> for example: {"B-PER":3,"O":8,...}
        """
        res = dict()
        unique_ent = list(set(self.all_entities))
        # Set initial values
        for ent in unique_ent:
            res[ent] = 0

        for entity_list in self.entities:
            res[entity_list[0]] += 1

        return res

    def merge_lists(self):
        """
        This function flattens nested lists for use when necessary.
        Model does not need POS tags and Chunk tags, so this function checks are there is initialized or not.
        :return: Flattened sentences, POS tags, Chunk tags and entity tags lists as tuple in order.

        Example return values
        Parsed Sentences : ["a","b","c","x",...]
        POS Values: ["NNP","VP","NNP","NNP",...]
        Chunks: ["B-NP","B-PP","I-NP","B-NP",...]
        Entities: ["O","B-PER","O","O",...]
        """
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


def viterbi(hmm, t_sentences):
    """
    This function dynamically applies the viterbi algorithm.
    :param hmm: Trained HMM model
    :param t_sentences: Test sentences
    :return: Predicted entity tags for each sentence in test sentences
    """
    res = list()
    states = list(hmm.n_init_entity.keys())

    for sent in t_sentences:
        D = np.zeros([len(states), len(sent)])      # Viterbi matrix
        E = np.zeros([len(states), len(sent) - 1])      # Matrix to store where highest calculation come to that cell.

        # Fill the first column of the Viterbi matrix
        for s in range(len(states)):
            init = hmm.init_prob[states[s]] if hmm.init_prob[states[s]] != 0 else hmm.s_init_prob[states[s]]
            emis = hmm.s_emis_prob[states[s]][sent[0]] if sent[0] in hmm.emis_prob[states[s]].keys() else \
                hmm.s_emis_prob[states[s]]["NaN"]
            D[s, 0] = math.log2(init) + math.log2(emis)

        # For each column (except first)
        for w in range(1, len(sent)):
            for i in range(len(states)):
                temp = np.zeros(len(states))
                past = D[:, w - 1]  # Retrieves the column with calculated values ​​for the previous word.

                # It calculates the probability of coming to the cell from the previous column and adds it to the list.
                for j in range(len(states)):
                    trans = hmm.trans_prob[states[j]][states[i]] if hmm.trans_prob[states[j]][states[i]] != 0 else \
                        hmm.s_trans_prob[states[j]][states[i]]
                    temp[j] = math.log2(trans) + past[j]

                emis = hmm.s_emis_prob[states[i]][sent[w]] if sent[w] in hmm.emis_prob[states[i]].keys() else \
                    hmm.s_emis_prob[states[i]]["NaN"]

                # Calculates the value of the current cell using the value of the cell that is most likely to arrive.
                D[i, w] = np.amax(temp) + math.log2(emis)

                # Set the cell to index of the cell that is most likely to arrive.
                E[i, w - 1] = np.argmax(temp)

        max_ind = np.zeros(len(sent))
        max_ind[-1] = np.argmax(D[:, -1])   # Get the index of maximum value at the last column

        # By using backtracking, the indices of the most suitable tags are recorded.
        for n in range(len(sent) - 2, -1, -1):
            max_ind[n] = E[int(max_ind[n + 1]), n]

        # Finds the names of the tags from the indices, creates a list and adds them to the answer list.
        res.append([states[n] for n in max_ind.astype(int)])
    return res


def dataset(conll_file):
    """
    Function to read CoNNL file
    :param conll_file: Path of the CoNNL file
    :return: Parsed sentences, POS tags, Chunk tags and Entity tags in given order as tuple

    Example return values
    Parsed Sentences : [["a","b",...],["c","x"],...]
    POS Values: [["NNP","VP",...],["NNP","NNP"],...]
    Chunks: [["B-NP","B-PP",...],["I-NP","B-NP"],...]
    Entities: [["O","B-PER",...],["O","O"],...]
    """
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
    conll_raw_data = [x.strip() for x in conll_raw_data]  # Clean the left/right spaces if there are

    for line in conll_raw_data:
        if line != '':
            split_line = line.split()
            if len(split_line) == 4:
                if split_line[0] != '-DOCSTART-':  # Do not get the lines which start with -DOCSTART-
                    temp_sent.append(split_line[0].lower())  # Get words in lowercase
                    temp_pos.append(split_line[1])
                    temp_chunk.append(split_line[2])
                    temp_entity.append(split_line[3])
            else:
                raise IndexError(
                    'Line split length does not equal 4.')  # A line must contain a word and 3 tags (POS, Chunk, Entity)
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


def accuracy(org_ent, pred_ent):
    """
    Counts the true predicted entities and returns the calculated accuracy.
    :param org_ent: Original entities of the test sentences as [["O","B-PER",...],["O","O"],...]
    :param pred_ent: Predicted entities of the test sentences as [["O","B-PER",...],["O","O"],...]
    :return: accuracy as float
    """
    true = 0

    org = [item for sublist in org_ent for item in sublist]
    pred = [item for sublist in pred_ent for item in sublist]

    for t, r in zip(org, pred):
        if t == r:
            true += 1
    print(true)
    print(len(org))
    return true / len(org)


def main():
    """
    Main function of the program
    """
    sentences, _, _, entities = dataset('../data/train.txt')
    # dev_sentences, _, _, dev_entities = dataset('../data/dev.txt')
    test_sentences, _, _, test_entities = dataset('../data/test.txt')

    hmm = HMM(sentences=sentences, entities=entities)

    res = viterbi(hmm=hmm, t_sentences=test_sentences)

    print("Acc: {}".format(accuracy(org_ent=test_entities, pred_ent=res)))


main()
