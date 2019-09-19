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

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/plsa-service/preprocessing')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[1])+'/topic-analysis/plsa-service/preprocessing')

import cleansing as pclean
import porter_dictionary

class LSA_wrapper:

    def __init__(self, docs,local=False):

        self.docs = docs
        if not local:
            self.root_path = str(pathlib.Path(os.path.abspath('')).parents[0]) + '/appData/lsa/'
        else:
            self.root_path = str(pathlib.Path(os.path.abspath('')).parents[1]) + '/appData/lsa/'
        print('>>>>>>>>>>>>>self.root_path>>>>>>>>>>>')
        print(self.root_path)
        self.extracted_folder = self.root_path + 'extracted/'
        self.file_dict = self.root_path + 'dict/'
        self.source_texts = self.root_path + 'extracted/'
        self.output_dir = self.root_path + 'cleaned/'
        print(self.output_dir)
        self.folder = self.root_path + 'cleaned/'
        self.dict_path = self.root_path + 'dict/'
        self.lsa_parameters_path = self.root_path + 'lsa-parameters/'
        self.LSA_PARAMETERS_PATH = ''

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
        topic_term_file = open(self.lsa_parameters_path + self.unique_folder_naming + 'word_by_topic_conditional2.csv', 'w')
        topic_term = open(self.lsa_parameters_path + self.unique_folder_naming + 'word_by_topic_conditional.csv', 'w')
        topic_term_file_2 = open(self.lsa_parameters_path + self.unique_folder_naming + 'topics.txt', 'w')
        
        # first lets save the topics with their words and the probability or imporatance value of each word for that topic
        for topic in topics:
            topic_term_file.write(str(topic))
            topic_term_file.write('\n\n')
        topic_term_file.close()

        # second lets save only the terms of each topic, also take the destemmed form of the word from dictionary

        # discarde the terms label in the topics and extracts only the term probability
       	extracted_prob = []
       	for i in range(len(topics)):
       		a = list(topics[i])
       		b = a[1].split("+")
       		t_prob = []
       		for i in b:
	       		a = i.strip()
	       		if re.match('^[0123456789-]', a):
	       			p = a.split("*")
	       			t_prob.append(p[0])
        	extracted_prob.append(t_prob)

        print(extracted_prob)

        for topic in extracted_prob:
            v = 0
            for doc_prob in topic:
                topic_term.write(str(doc_prob)+",")
                v += float(doc_prob)
            print("summation of all word probability in topic ", extracted_prob.index(topic), "is: ",  v)
            topic_term.write('\n')
        
        
        # discard the probability of each term in the topics and extratracts only the term
       	term = []
       	for i in range(len(topics)):
       		a = list(topics[i])
       		b = a[1].split("\"")
       		t_term = []
       		for i in b:
	       		a = i.strip()
	       		if(len(a)>1):
	       			if(a[0].isalpha()):
	       				t_term.append(a)

        	term.append(t_term)
        # gets the destemmed form of each word from the dictionary
        port_dict = porter_dictionary.porter_dictionary()
        port_dict.load_dict(self.dict_path + self.unique_folder_naming[:-1] + '_dict')

        try:
            topics_destemmed = [[port_dict.dictionary[j][0] for j in i] for i in term]
        except:
            logging.exception('message')
       
        # save only the term
        for topic in topics_destemmed:
            for term in topic:
                topic_term_file_2.write(str(term) + ", ")

            topic_term_file_2.write('\n')

    def prepare_corpus(self):
	    """
	    Input : clean document
	    purpose : create term dictionary of our corpus and convert list of docs(corpus) into document-term matrix
	    Output : term dictionary and document-term matrix
	    """
	    
	    os.mkdir(pclean.output_dir)

	    # Do cleansing on the data and turing it to bad-of-words model
	    with open(self.lsa_parameters_path + self.unique_folder_naming + 'status.txt', 'w') as f:
	    	f.write('Preprocessing started.' + '\n')

	    pclean.pre_pro()

	    with open(self.lsa_parameters_path + self.unique_folder_naming + 'status.txt', 'w') as f:
	    	f.write('Preprocessing finished.\n') 
	    	f.write('Topic analysis started.\n')

	    with open(pclean.output_dir+'cleaned.json', "r") as read_file:
	    	ret = json.load(read_file)

	    data_lemmatized = []
	    for k in ret:
	    	data_lemmatized.append(ret[k].splitlines())

	    # save lemmatized document
	    with open(pclean.output_dir+'lemmatized.txt', 'w') as lem:
	    	for d in data_lemmatized:
	    		lem.write(str(d))
	    		lem.write('\n')

	    # Create Dictionar
	    id2word = corpora.Dictionary(data_lemmatized)
	    # save id2word dictionary
	    with open(pclean.output_dir+'id2word.txt', 'w') as dic:
	    	for i in id2word:
	    		dic.write(str(id2word[i]))
	    		dic.write('\n')

	    # Create Corpus
	    texts = data_lemmatized

	    # Term Document Frequency
	    corpus = [id2word.doc2bow(text) for text in texts]

	    return id2word, corpus

    def generate_topics_gensim(self, num_topics=2, per_words=300, chunksize=20000, decay=1.0, distributed=False, onepass=True, power_iters=2, extra_samples=100):
        start_time_1 = time.time()

        # get the dictionary and corpus
        id2word, corpus = self.prepare_corpus()

        self.lsa_model = gensim.models.lsimodel.LsiModel(corpus=corpus,
                                                    num_topics=num_topics,
                                                    id2word=id2word,
                                                    chunksize=chunksize,
                                                    decay=decay,
                                                    distributed=distributed,
                                                    onepass=onepass,
                                                   	power_iters=power_iters,
                                                    extra_samples=extra_samples)

        """ Write topic term matrix into file P(W/Z) """
        topics = self.lsa_model.print_topics(num_topics=num_topics, num_words=per_words)
        # in gensim, term in each topics are ordered in terms of their importance value for that topic
        self.save_topic_term_matrix(topics)

        """display total processing time took"""
        end_time_1 = time.time()
        total_training_time  = round((end_time_1 - start_time_1) / 60 , 4)
        print('Total training time took: ' + str(total_training_time) + ' minutes')

        with open(self.lsa_parameters_path + self.unique_folder_naming + 'status.txt', 'w') as f:
            f.write('Topic analysis finished.\n')
            f.write(str(total_training_time))
        
        
def run_lsa(path):

    docs = []
    print(path)

    npath = path.split(".")
    print(npath)
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
    s = LSA_wrapper(docs,local=True)
    
    s.unique_folder_naming = str(datetime.datetime.now()).replace(':','-').replace('.','-') + '^' + str(random.randint(100000000000, 999999999999)) + '/'
    os.mkdir(str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/lsa/lsa-parameters/'+s.unique_folder_naming)

    s.write_to_json()
    

    s.generate_topics_gensim(num_topics=2)

__end__ = '__end__'


if __name__ == '__main__':

    test1 = str(pathlib.Path(os.path.abspath('')).parents[0]) + '/docs/tests/test_doc 2.txt'
    test2 = str(pathlib.Path(os.path.abspath('')).parents[0]) + '/docs/tests/test_doc.txt'
    test3 = str(pathlib.Path(os.path.abspath('')).parents[0])+'/docs/tests/test_doc_2.txt'
    test4 = str(pathlib.Path(os.path.abspath('')).parents[0])+'/docs/tests/topic_analysis.json'
    test5 = str(pathlib.Path(os.path.abspath('')).parents[0])+'/docs/tests/topic_analysis_2.json'

    run_lsa(test5)

    pass