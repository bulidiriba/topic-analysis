__author__ = 'eyob'
# Tested on python3.6

import psutil
print('===================ram used at program start:',float(list(psutil.virtual_memory())[3])/1073741824.0,'GB')

import os
import sys
import pathlib
import csv
import random
import datetime
import time
import json
import logging

import re
import numpy as np
import pandas as pd
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/plsa-service/plsa')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/plsa-service/preprocessing')
# sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[1])+'/plsa-service/plsa')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[1])+'/topic-analysis/plsa-service/preprocessing')

# import example_plsa as pplsa
import cleansing as pclean
import porter_dictionary

class LDA_wrapper:

    def __init__(self, docs,local=False):

        self.docs = docs
        if not local:
            self.root_path = str(pathlib.Path(os.path.abspath('')).parents[0]) + '/appData/lda/'
        else:
            self.root_path = str(pathlib.Path(os.path.abspath('')).parents[1]) + '/appData/lda/'
        print('>>>>>>>>>>>>>self.root_path>>>>>>>>>>>')
        print(self.root_path)
        self.extracted_folder = self.root_path + 'extracted/'
        self.file_dict = self.root_path + 'dict/'
        self.source_texts = self.root_path + 'extracted/'
        self.output_dir = self.root_path + 'cleaned/'
        print(self.output_dir)
        self.folder = self.root_path + 'cleaned/'
        self.dict_path = self.root_path + 'dict/'
        self.lda_parameters_path = self.root_path + 'lda-parameters/'
        self.LDA_PARAMETERS_PATH = ''

        # self.messages
        self.unique_folder_naming = str(datetime.datetime.now()).replace(':','-').replace('.','-') + '^' + str(random.randint(100000000000, 999999999999)) + '/'
        os.mkdir(self.lda_parameters_path + self.unique_folder_naming)
        
        # specific file
        self.status_file = self.lda_parameters_path + self.unique_folder_naming + 'status.txt'
        self.num_topics = None
        self.topic_divider = None
        self.max_iter = None

    def __del__(self):

        # Close db connections
        pass

    def write_to_json(self):



        # self.unique_folder_naming = str(datetime.datetime.now()).replace(':','-').replace('.','-') + '^' + str(random.randint(100000000000, 999999999999)) + '/'
        print(self.unique_folder_naming)

        os.mkdir(self.extracted_folder+self.unique_folder_naming)

        contents_dict = {}

        file = self.extracted_folder + self.unique_folder_naming + 'extracted' + '.json'

        for i in range(len(self.docs)):
            contents_dict[str(i)] = self.docs[i]

        with open(file, "w") as f:
            json.dump(contents_dict, f, indent=4)

        print("len(contents_dict):",len(contents_dict))

    def save_topic_term_matrix(self, topics):
        topic_term_file = open(self.lda_parameters_path + self.unique_folder_naming + 'lda_topics_words&weight.txt', 'w')
        topic_term_file_2 = open(self.lda_parameters_path + self.unique_folder_naming + 'lda_topics_word_only.txt', 'w')
        
        status_file = open(self.status_file, 'a')
        status_file.write("The number of topics: " + str(len(topics)) + "\n")
        
        # first lets save the topics with their words and the probability or imporatance value of each word for that topic
        for topic in topics:
            topic_term_file.write(str(topic))
            topic_term_file.write('\n\n')
        topic_term_file.close()

        # second lets save only the terms of each topic, also take the destemmed form of the word from dictionary

        # discard the probability of each term in the topics and extratracts only the term
        extracted_topics = [[item[0] for item in topic[1]] for topic in topics]

        # gets the destemmed form of each word from the dictionary
        port_dict = porter_dictionary.porter_dictionary()
        port_dict.load_dict(self.dict_path + self.unique_folder_naming[:-1] + '_dict')

        try:
            topics_destemmed = [[port_dict.dictionary[j][0] for j in i] for i in extracted_topics]
        except:
            logging.exception('message')

        status_file.write("The total number of word in each topic: " + str(len(topics_destemmed[0])) + "\n")
       
        for topic in topics_destemmed:
            for term in topic:
                topic_term_file_2.write(str(term) + ", ")

            topic_term_file_2.write('\n')

    def save_doc_topic_matrix(self, doc_topic_mat):
        doc_topic_file = open(self.lda_parameters_path + self.unique_folder_naming + "doc_topic_matrix.txt", 'w')
        # save the total number of documents 
        with open(self.status_file, 'a') as st:
            st.write("The total number of documents: " + str(len(doc_topic_mat)) + "\n")
            
        for doc in doc_topic_mat:
            index = doc_topic_mat.index(doc)+1
            doc_topic_file.write('doc '+str(index)+" "+str(doc))
            doc_topic_file.write('\n')

            v = 0
            for i in doc:
                v += i[1]
            #print("summation of all topic probability in document "+str(index)+" is: "+str(v))

        doc_topic_file.close()

    def save_topic_coherence(self, top_topics):
        topic_prob_file = open(self.lda_parameters_path + self.unique_folder_naming + "topic_coherence.txt", 'w')
        for topic in top_topics:
            index = top_topics.index(topic)
            topic_prob_file.write("coherence of topic "+str(index)+": " + str(topic[1]))
            topic_prob_file.write('\n')
        topic_prob_file.close()

    def generate_topics_gensim(self,num_topics, passes, chunksize,
                               update_every=0, alpha='auto', eta='auto', decay=0.5, offset=1.0, eval_every=1,
                               iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None,
                               minimum_phi_value=0.01, per_word_topics=True, callbacks=None):
        start_time_1 = time.time()

        pclean.file_dict = self.file_dict + self.unique_folder_naming[:-1] + '_dict'
        pclean.source_texts = self.source_texts + self.unique_folder_naming + 'extracted.json'
        pclean.output_dir = self.output_dir + self.unique_folder_naming

        os.mkdir(pclean.output_dir)

        # Do cleansing on the data and turing it to bad-of-words model

        with open(self.status_file, 'w') as f:
            f.write('Preprocessing started.' + '\n')

        pclean.pre_pro()

        with open(self.status_file, 'a') as f:
            f.write('Preprocessing finished.\n') 
            f.write('Topic analysis started.\n')

        with open(pclean.output_dir+'cleaned.json', "r") as read_file:
            ret = json.load(read_file)

        data_lemmatized = []

        for k in ret:
            data_lemmatized.append(ret[k].splitlines())

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # View
        # print(corpus[0:1])
        # print(id2word[1])

        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topics,
                                                    random_state=random_state,
                                                    update_every=update_every,
                                                    chunksize=chunksize,
                                                    passes=passes,
                                                    alpha=alpha,
                                                    eta=eta,
                                                    per_word_topics=per_word_topics,
                                                    decay=decay,
                                                    offset=offset,
                                                    eval_every=eval_every,
                                                    iterations=iterations,
                                                    gamma_threshold=gamma_threshold,
                                                    minimum_probability=minimum_probability,
                                                    minimum_phi_value=minimum_phi_value,
                                                    callbacks=callbacks)

        """ Write topic term matrix into file P(W/Z) """
        topics = self.lda_model.show_topics(num_topics=num_topics,num_words=300,formatted=False)
        # in gensim, term in each topics are ordered in terms of their importance value for that topic
        self.save_topic_term_matrix(topics)

        """Write document topic matrix into file P(D/Z)"""
        doc_topic_mat = []
        for bowd in corpus:
            doc_topic_mat.append(self.lda_model.get_document_topics(bow=bowd, minimum_probability=0.000001))
        self.save_doc_topic_matrix(doc_topic_mat)

        ''' Write topic coherence of each topic in the corpus '''
        top_topics = self.lda_model.top_topics(corpus=corpus, topn=300)
        self.save_topic_coherence(top_topics)

        '''to check that topic_by_term is conditional probability(their summation becomes 1)'''
        term_probability = self.lda_model.get_topics()
        v = 1
        for topic in term_probability:
            value = 0
            for prob in topic:
                value += prob
            #print("summation of all term probability in topic: ",v, " is ", value)
            v +=1 

        """display total processing time took"""
        end_time_1 = time.time()
        total_training_time  = round((end_time_1 - start_time_1) / 60 , 4)
        print('Total training time took: ' + str(total_training_time) + ' minutes')

        with open(self.status_file, 'a') as f:
            f.write('LDA Topic analysis Finished.\n')
            f.write('Total Processing time toook: '+ str(total_training_time) + ' minutes\n')

        '''
        Seems remaining code is to extract any produced parameters from the resulting lda model, like the weights. We need to define the proto formats of course
        for all the returned parameters
        
        also code that writes the final status that shows total running time that elapsed
        
        in general, compare the outputs of plsa and as much as possible try to apply it to the results that are returned by lda
        '''

def run_lda():

    docs = []
    s = LDA_wrapper(docs, local=True)

    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_2.json'
    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_singnet_all.json'
    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_bio_all.json'
    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_hersheys_all.json'
    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_hr_all.json'
    
    path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topic_analysis.json'

    docs = []

    with open(path, "r") as read_file:
        fileList = json.load(read_file)

    for k in fileList:
        docs.append(fileList[k])

    s = LDA_wrapper(docs,local=True)
    # s.topic_divider = 0
    # s.num_topics = 2
    # s.max_iter = 22
    # s.beta = 1
    #s.unique_folder_naming = str(datetime.datetime.now()).replace(':','-').replace('.','-') + '^' + str(random.randint(100000000000, 999999999999)) + '/'
    #os.mkdir(str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/lda/lda-parameters/'+s.unique_folder_naming)
    s.write_to_json()
    # s.generate_topics_gensim(num_topics=3,passes=22,chunksize=200)
    s.generate_topics_gensim(num_topics=5,passes=22,chunksize=200, per_word_topics=300)
    # s.generate_topics_gensim(num_topics=2,passes=22,chunksize=200)
    # s.generate_topics_gensim(num_topics=2,passes=100,chunksize=200,random_state=2)


    # pprint(s.lda_model.print_topics(3,50))
    # topics = s.lda_model.show_topics(2,5,formatted=False)
    # print(topics)
    #print_two_d(s.topics_destemmed)


    # topics_snet_all_plsa_file = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topics/singnet_all_plsa_topics_2.txt'
    # topics_snet_all_plsa_file = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topics/hersheys_all_plsa_topics.txt'
    # topics_snet_all_plsa_file = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topics/bio_all_plsa_topics.txt'

    # topics_snet_all_plsa_file = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topics/hr_all_plsa_topics.txt'
    # with open(topics_snet_all_plsa_file,'r') as f:
    #     temp_list = f.readlines()
    #     topics_snet_all_plsa = []
    #     for l in temp_list:
    #         topics_snet_all_plsa.append(l.split(','))
    #
    #     for i in range(len(topics_snet_all_plsa)):
    #         for j in range(len(topics_snet_all_plsa[0])):
    #             topics_snet_all_plsa[i][j] = topics_snet_all_plsa[i][j].strip()
    #
    # topics_snet_all_plsa_file_2 = str(pathlib.Path(os.path.abspath('')).parents[1]) + '/appData/misc/topics/hr_all_plsa_topics_2.txt'
    # with open(topics_snet_all_plsa_file_2, 'r') as f:
    #     temp_list = f.readlines()
    #     topics_snet_all_plsa_2 = []
    #     for l in temp_list:
    #         topics_snet_all_plsa_2.append(l.split(','))
    #
    #     for i in range(len(topics_snet_all_plsa_2)):
    #         for j in range(len(topics_snet_all_plsa_2[0])):
    #             topics_snet_all_plsa_2[i][j] = topics_snet_all_plsa_2[i][j].strip()



    # two topics
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[0],depth=30))
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[1],depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[1],s.topics_destemmed[0],depth=30))
    # print(dot_product(topics_snet_all_plsa[1],s.topics_destemmed[1],depth=30))
    # two topics

    # three topics
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[0],depth=30))
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[1],depth=30))
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[2],depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[1], s.topics_destemmed[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[1], s.topics_destemmed[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[1], s.topics_destemmed[2], depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[2], s.topics_destemmed[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[2], s.topics_destemmed[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[2], s.topics_destemmed[2], depth=30))
    # print('=========================')
    # three topics

    # plsa self
    # print(dot_product(topics_snet_all_plsa[0], topics_snet_all_plsa_2[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[0], topics_snet_all_plsa_2[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[0], topics_snet_all_plsa_2[2], depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[1], topics_snet_all_plsa_2[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[1], topics_snet_all_plsa_2[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[1], topics_snet_all_plsa_2[2], depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[2], topics_snet_all_plsa_2[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[2], topics_snet_all_plsa_2[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[2], topics_snet_all_plsa_2[2], depth=30))
    # print('=========================')
    # plsa self


def dot_product(list_1,list_2,depth=30):

    count = 0
    for i in list_1[0:depth]:
        if i in list_2[0:depth]:
            count = count + 1
    return count


def print_two_d(two_d):
    for i in two_d:
        print(i)




__end__ = '__end__'


if __name__ == '__main__':

    run_lda()

    pass