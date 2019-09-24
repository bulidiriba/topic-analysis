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

sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/plsa-service/preprocessing')
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

        pclean.file_dict = self.file_dict + self.unique_folder_naming[:-1] + '_dict'
        pclean.source_texts = self.source_texts + self.unique_folder_naming + 'extracted.json'
        pclean.output_dir = self.output_dir + self.unique_folder_naming

    def save_topic_term_matrix(self, topics):
        topic_term_file = open(self.lda_parameters_path + self.unique_folder_naming + 'word_by_topic_conditional2.csv', 'w')
        topic_term = open(self.lda_parameters_path + self.unique_folder_naming + 'word_by_topic_conditional.csv', 'w')
        topic_term_file_2 = open(self.lda_parameters_path + self.unique_folder_naming + 'topics.txt', 'w')
        
        # first lets save the topics with their words and the probability or imporatance value of each word for that topic
        for topic in topics:
            topic_term_file.write(str(topic))
            topic_term_file.write('\n\n')
        topic_term_file.close()

        # second lets save only the terms of each topic, also take the destemmed form of the word from dictionary

        # discarde the terms label in the topics and extracts only the term probability
        extracted_prob = [[item[1] for item in topic[1]] for topic in topics]

        # save only the probability
        #print(len(extracted_prob))
        for topic in extracted_prob:
            v = 0
            for prob in topic:
                topic_term.write(str(prob)+",")
                v += prob
            #print("value", v)
            topic_term.write('\n')

        # print("length of ", len(extracted_prob[0]))
        # discard the probability of each term in the topics and extratracts only the term
        extracted_topics = [[item[0] for item in topic[1]] for topic in topics]

        # gets the destemmed form of each word from the dictionary
        port_dict = porter_dictionary.porter_dictionary()
        port_dict.load_dict(self.dict_path + self.unique_folder_naming[:-1] + '_dict')

        try:
            topics_destemmed = [[port_dict.dictionary[j][0] for j in i] for i in extracted_topics]
        except:
            logging.exception('message')
       
        # save only the term
        for topic in topics_destemmed:
            for term in topic:
                topic_term_file_2.write(str(term) + ", ")

            topic_term_file_2.write('\n')

    def topic_by_doc_matrix(self, model, corpus, non_empty_doc_index, minimum_probability):
        doc_topic_conditional_file = open(self.lda_parameters_path + self.unique_folder_naming + "topic-by-doc-conditional.csv", 'w')
        doc_topic_joint_file = open(self.lda_parameters_path + self.unique_folder_naming + "topic-by-doc-matirx.csv", 'w')
        
        doc_by_topic_mat = []  # matrix to store the doc by topic results returned by gensim library
        extracted_prob = [] # matrix to extract only the probability from the doc_by_topic_mat
        topic_by_doc =[] # matrix to store the transpose of above probability
        topic_by_doc_joint = [] # matrix to store the joint probability of topic by doc

        # return the result from gensim functions get_document_topics
        for bowd in corpus:
            doc_by_topic_mat.append(model.get_document_topics(bow=bowd, minimum_probability=minimum_probability))

        # extract only the probability of each topic in the given documents
        extracted_prob = [[topic[1] for topic in doc] for doc in doc_by_topic_mat]
        
        # transpose the probability distributions to change it in the same format with PLSA topic_by_doc_matrix.
        topic_by_doc = [list(i) for i in zip(*extracted_prob)]

        # to check that the probability summation of all topic in a given document is 1 or not.
        #print("summation of all topic probability in each document is")
        #print(np.sum(np.asarray(topic_by_doc, dtype=np.float32), axis=0))

        
        # calculate the Joint Probability of each documents and each topics
        # p(z0, d0) = p(z0|d0) * p(d0)
       
        # call the document probability
        doc_prob = self.document_probability(corpus)
        
        for topic in topic_by_doc:
            joint_topic_prob = []
            for indx, prob in enumerate(topic):
                res = prob * doc_prob[indx] 
                joint_topic_prob.append(res)
            topic_by_doc_joint.append(joint_topic_prob)
        
        #print("summation of joint probability of topic and document is")
        #print(np.sum(np.sum(np.asarray(topic_by_doc_joint, dtype=np.float32), axis=0)))


        # then now write to file, first write the index of non_empty_doc, then the conditional probability of each topic in this non_empty_doc
        
        #  write the non_empty_doc_index to the file
        for indx in non_empty_doc_index:
            doc_topic_conditional_file.write(","+str(indx))
            doc_topic_joint_file.write(","+str(indx))
        doc_topic_conditional_file.write("\n")
        doc_topic_joint_file.write("\n")

       
        # then write the conditional probability of each topic in the given documents.       
        for topic in topic_by_doc:
            doc_topic_conditional_file.write(str(topic_by_doc.index(topic))+",")
            for doc in topic:
                if(topic.index(doc) != len(topic) - 1):
                    doc_topic_conditional_file.write(str(doc)+",")
                else:
                    doc_topic_conditional_file.write(str(doc))

            doc_topic_conditional_file.write('\n')
        doc_topic_conditional_file.close()

        # then write the Joint probability of each topic and each document.       
        for topic in topic_by_doc_joint:
            doc_topic_joint_file.write(str(topic_by_doc_joint.index(topic))+",")
            for doc in topic:
                if(topic.index(doc) != len(topic) - 1):
                    doc_topic_joint_file.write(str(doc)+",")
                else:
                    doc_topic_joint_file.write(str(doc))

            doc_topic_joint_file.write('\n')
        doc_topic_joint_file.close()

    def save_topic_coherence(self, top_topics):
        topic_prob_file = open(self.lda_parameters_path + self.unique_folder_naming + "topic_coherence.txt", 'w')
        for topic in top_topics:
            index = top_topics.index(topic)
            topic_prob_file.write("coherence of topic "+str(index)+": " + str(topic[1]))
            topic_prob_file.write('\n')
        topic_prob_file.close()

    def document_probability(self, corpus):
        doc_prob_file = open(self.lda_parameters_path + self.unique_folder_naming + "document_probability", 'w')
        doc_prob = []
        
        # counting all the total words in the corpus, this is some how bigger than the number of vocabulary 
        # since one documents can contain duplicated words.
        total_num_words_in_corpus = np.sum([len(doc) for doc in corpus])

        for document in corpus:
            # the total number of words in each document
            n_word_document = len(document)
            
            # probability of each document = n_word_document / total_num_words_in_corpus
            prob = n_word_document / total_num_words_in_corpus

            doc_prob.append(prob)

        #print("summation of all document probability(must be 1)")
        #print(np.sum(np.asarray(doc_prob, dtype=np.float32), axis=0))
        
        for d_prob in doc_prob:
            doc_prob_file.write(str(d_prob)+'\n')
        doc_prob_file.close()

        return doc_prob

    def topic_probability(self, corpus, model, num_topics, minimum_probability):
        dist_file = open(self.lda_parameters_path + self.unique_folder_naming + "topic_probability_pz", 'w')
        '''
            the formula used to calculate this is
            p(zi) = (p(zi|d0)+p(zi|d1)+p(zi|d2)+...+p(zi|dn)) / len(corpus)

            len(corpus) here means length of non-empty documents from the corpus, because empty docs are removed.
        '''

        # list that store the probability for each topic
        topic_prob_dist = [0] * num_topics
        
        # loop through all the documents
        for index, document in enumerate(corpus):
            # loop through all the probability of topics for each documents
            for prob_dist in model.get_document_topics(document, minimum_probability=minimum_probability):
                #print("topic_id: "+str(prob_dist[0])+" probability: "+str(prob_dist[1]))

                # to get the topic id in each document
                indx = prob_dist[0]

                # to get the probability of topic correspond to the index
                prob = topic_prob_dist[indx]
                
                # then sum up all the probability of a specific topic in all the document.
                # p(t1/d1)+p(t1/d2)+p(t1/d3)+p(t1/d4) ... 
                # for each topic like this update again and again until you calculate from all the document
                prob += prob_dist[1]
                
                # store the current calculated probability in the list and update it again and again, 
                # until you calculate it from all the documents. 
                topic_prob_dist[indx] = prob
                
        v = 0
        # then divide each calculated probability by length of corpus 
        # to get the required probability between 0 and 1
        # and then write to file probability of each topic given the corpus
        for index, prob in enumerate(topic_prob_dist):
            prob_topic = prob / len(corpus)
            dist_file.write(str(prob_topic))
            dist_file.write('\n')

            v += prob_topic

        # print("\nsum of all topic probability in a corpus is", v)

        dist_file.close()

    def prepare_corpus(self):
        """
        Input : clean document
        purpose : create term dictionary of our corpus and convert list of docs(corpus) into document-term matrix
        Output : term dictionary and document-term matrix
        """
        os.mkdir(pclean.output_dir)

        # Do cleansing on the data and turing it to bad-of-words model
        with open(self.lda_parameters_path + self.unique_folder_naming + 'status.txt', 'w') as f:
            f.write('Preprocessing started.')

        pclean.pre_pro()

        with open(self.lda_parameters_path + self.unique_folder_naming + 'status.txt', 'w') as f:
            f.write('Preprocessing finished. Topic analysis started.') 
            
        with open(pclean.output_dir+'cleaned.json', "r") as read_file:
            documents = json.load(read_file)

        # change loaded json file into list
        data = []
        for document in documents:
            # append each lines to list elements by splitting with a line.
            data.append(documents[document].splitlines())

        # to store index of non-empty documents
        non_empty_doc_index = []

        # removes empty documents, by checking the length of the document if its empty its length is zero.
        # also remove documents with only single word.
        data_lemmatized = []
        for d in data:
            if len(d) == 0:
                pass
            else:
                data_lemmatized.append(d)
                non_empty_doc_index.append(data.index(d))

        # save lemmatized document
        with open(pclean.output_dir+'lemmatized.txt', 'w') as lem:
            for d in data_lemmatized:
                lem.write(str(d))
                lem.write('\n')

        # create dictionary, where every unique term is assigned an index, you can order it according to alphabets
        id2word = corpora.Dictionary(data_lemmatized)
        # save id2word dictionary
        with open(pclean.output_dir+'id2word.txt', 'w') as dic:
            for i in id2word:
                dic.write(str(id2word[i]))
                dic.write('\n')

        # Create Corpus
        texts = data_lemmatized

        # create document-term matrix using dictionary created above
        corpus = [id2word.doc2bow(text) for text in texts]

        return id2word, corpus, texts, non_empty_doc_index


    def generate_topics_gensim(self,num_topics, passes=22, chunksize=200,
                               update_every=0, alpha='auto', eta='auto', decay=0.5, offset=1.0, eval_every=1,
                               iterations=50, gamma_threshold=0.001, minimum_probability=0.0, random_state=None,
                               minimum_phi_value=0.01, per_word_topics=True, callbacks=None, num_words=300, coherence='u_mass'):
        
        # start the time to check how many seconds it will take.
        start_time_1 = time.time()

        # get the dictionary and corpus
        id2word, corpus, texts, non_empty_doc_index = self.prepare_corpus()

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
        topics = self.lda_model.show_topics(num_topics=num_topics,num_words=num_words,formatted=False)
        # in gensim, term in each topics are ordered in terms of their importance value for that topic
        self.save_topic_term_matrix(topics)
    
        # call the topic_by_doc_matrix functions
        self.topic_by_doc_matrix(self.lda_model, corpus, non_empty_doc_index, minimum_probability)

        '''Topic Probability'''
        self.topic_probability(corpus, self.lda_model, num_topics, minimum_probability)

        ''' Write topic coherence of each topic in the corpus '''
        # here we have two options for coherenece type(c_v or u_mass)
        top_topics = self.lda_model.top_topics(corpus=corpus, topn=num_words, texts=texts, coherence=coherence)
        self.save_topic_coherence(top_topics)

        """display total processing time took and write to file"""
        end_time_1 = time.time()
        total_training_time  = round((end_time_1 - start_time_1) / 60 , 4)
        print('Total training time took: ' + str(total_training_time) + ' minutes')
        with open(self.lda_parameters_path + self.unique_folder_naming + 'status.txt', 'w') as f:
            f.write('Topic analysis finished.\n')
            f.write(str(total_training_time))
    

def run_lda(path):

    docs = []
    print(path)
    
    npath = path.split(".")
    #print(npath)
    # if the file is .json
    if npath[-1] == "json":
        with open(path, "r") as read_file:
            fileList = json.load(read_file)
        
        docs = fileList['docs']

    # if the file is .txt
    elif npath[-1] == "txt":
        with open(path, "r") as file:
            for cnt, line in enumerate(file):
                docs.append(line)

    else:
        print("your file format is not supported")
        exit(0)

     # call the LSA Wrapper
    s = LDA_wrapper(docs,local=True)
    
    s.unique_folder_naming = str(datetime.datetime.now()).replace(':','-').replace('.','-') + '^' + str(random.randint(100000000000, 999999999999)) + '/'
    os.mkdir(str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/lda/lda-parameters/'+s.unique_folder_naming)

    s.write_to_json()

    s.generate_topics_gensim(num_topics=2,passes=22,chunksize=200, per_word_topics=True, num_words=300, coherence="u_mass")
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

    test1 = str(pathlib.Path(os.path.abspath('')).parents[0]) + '/docs/tests/test_doc 2.txt'
    test2 = str(pathlib.Path(os.path.abspath('')).parents[0]) + '/docs/tests/test_doc.txt'
    test3 = str(pathlib.Path(os.path.abspath('')).parents[0])+'/docs/tests/test_doc_2.txt'
    test4 = str(pathlib.Path(os.path.abspath('')).parents[0])+'/docs/tests/topic_analysis.json'
    test5 = str(pathlib.Path(os.path.abspath('')).parents[0])+'/docs/tests/topic_analysis_2.json'
    
    run_lda(test1)

    pass