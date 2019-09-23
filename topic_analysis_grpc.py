# Tested on python3.6

import grpc
from concurrent import futures
import time
import logging
import sys
import pathlib
import os
import csv
import numpy as np
import datetime
import random
from nltk.tokenize import sent_tokenize

SLEEP_TIME = 86400 # One day


sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/topic-analysis/plsa-service/preprocessing')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/topic-analysis/plsa-service/plsa')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/topic-analysis/lda-service')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/topic-analysis/lsa-service')


print(sys.path)

import plsa_wrapper
import lda_wrapper
import lsa_wrapper
import threading
import multiprocessing as mp

#from service_spec import topic_analysis_pb2
#from service_spec import topic_analysis_pb2_grpc

import topic_analysis_pb2
import topic_analysis_pb2_grpc


class TopicAnalysis(topic_analysis_pb2_grpc.TopicAnalysisServicer):

    def PLSA(self,request,context):

        print('>>>>>>>>>>>>>>In endpoint plsa')
        print(time.strftime("%c"))

        docs = request.docs
        num_topics = request.num_topics
        topic_divider = request.topic_divider
        maxiter = request.maxiter
        beta = request.beta

        param_error = False
        message = ''

        try :
            if len(docs) < 1:
                message = 'Length of docs should be at one'
                param_error =True

            if len(docs) == 1:
                docs = sent_tokenize(docs[0])

            if topic_divider < 0:
                param_error = True
                message = 'topic_divider parameter can not be a negative nubmer'

            if topic_divider == 0 and num_topics < 2:
                param_error = True
                message = 'Number of topics should be at least two'

            if maxiter < 0:
                param_error = True
                message = 'maxiter should be greater than zero'

            if beta < 0 or beta > 1:
                param_error = True
                message = 'beta should have value of (0,1]'


            if param_error:
                print(time.strftime("%c"))
                print('Waiting for next call on port 5000.')
                raise grpc.RpcError(grpc.StatusCode.UNKNOWN, message)


        except Exception as e:

            logging.exception("message")

            print(time.strftime("%c"))
            print('Waiting for next call on port 5000.')

            raise grpc.RpcError(grpc.StatusCode.UNKNOWN, str(e))

        try:

            unique_folder_naming = str(datetime.datetime.now()).replace(':', '-').replace('.', '-') + '^' + str(random.randint(100000000000, 999999999999)) + '_plsa/'

            # thread1 = threading.Thread(target=generate_topics_plsa, args=(docs,unique_folder_naming,num_topics,topic_divider,maxiter,beta))
            p1 = mp.Process(target=generate_topics_plsa, args=(docs,unique_folder_naming,num_topics,topic_divider,maxiter,beta))

            p1.start()

            resp = topic_analysis_pb2.PLSAResponse(status=True, message='success', handle=unique_folder_naming[:-1].replace('-','e').replace(' ','d').replace('^','y'))

            print('\n')
            print('status:',resp.status)
            print('message:',resp.message)
            print('Waiting for next call on port 5000.')
            print('\n')

            return resp

        except Exception as e:

            logging.exception("message")

            print(time.strftime("%c"))
            print('Waiting for next call on port 5000.')

            raise grpc.RpcError(grpc.StatusCode.UNKNOWN, str(e))

    def LDA(self, request, context):
        print('>>>>>>>>>>>>>>In endpoint lda')
        print(time.strftime("%c"))

        docs = request.docs
        num_topics = request.num_topics
        topic_divider = request.topic_divider
        iterations = request.iterations
        num_words = request.num_words
        minimum_probability = request.minimum_probability
        minimum_phi_value = request.minimum_phi_value
        per_word_topics = request.per_word_topics
        chunksize = request.chunksize
        passes = request.passes
        alpha = request.alpha
        eta = request.eta
        decay = request.decay
        gamma_threshold = request.gamma_threshold
        coherence = request.coherence
        eval_every = request.eval_every
        update_every = request.update_every
        offset = request.offset

        param_error = False
        message = ''
        try :
            if len(docs) < 1:
                message = 'Length of docs should be at one'
                param_error =True

            if len(docs) == 1:
                docs = sent_tokenize(docs[0])

            if param_error:
                print(time.strftime("%c"))
                print('Waiting for next call on port 5000.')
                raise grpc.RpcError(grpc.StatusCode.UNKNOWN, message)
        except Exception as e:

            logging.exception("message")

            print(time.strftime("%c"))
            print('Waiting for next call on port 5000.')

            raise grpc.RpcError(grpc.StatusCode.UNKNOWN, str(e))
        try:

            unique_folder_naming = str(datetime.datetime.now()).replace(':', '-').replace('.', '-') + '^' + str(random.randint(100000000000, 999999999999)) + '_lta/'

            # thread1 = threading.Thread(target=generate_topics_plsa, args=(docs,unique_folder_naming,num_topics,topic_divider,maxiter))
            p1 = mp.Process(target=generate_topics_lda, args=(docs,unique_folder_naming,num_topics,topic_divider,iterations, num_words, minimum_probability, 
                minimum_phi_value, per_word_topics, chunksize, passes, alpha, eta, decay, gamma_threshold, coherence, eval_every, update_every, offset))

            p1.start()

            resp = topic_analysis_pb2.LDAResponse(status=True, message='success', handle=unique_folder_naming[:-1].replace('-','e').replace(' ','d').replace('^','y'))

            print('\n')
            print('status:',resp.status)
            print('message:',resp.message)
            print('Waiting for next call on port 5000.')
            print('\n')

            return resp
        except Exception as e:

            logging.exception("message")

            print(time.strftime("%c"))
            print('Waiting for next call on port 5000.')

            raise grpc.RpcError(grpc.StatusCode.UNKNOWN, str(e))

    def LSA(self, request, context):
        print('>>>>>>>>>>>>>>In endpoint lsa')
        print(time.strftime("%c"))

        docs = request.docs
        num_topics = request.num_topics
        num_words = request.num_words
        chunksize = request.chunksize
        distributed = request.distributed
        onepass = request.onepass
        power_iters = request.power_iters
        param_error = False
        message = ''
        try :
            if len(docs) < 1:
                message = 'Length of docs should be at one'
                param_error =True

            if len(docs) == 1:
                docs = sent_tokenize(docs[0])

            if param_error:
                print(time.strftime("%c"))
                print('Waiting for next call on port 5000.')
                raise grpc.RpcError(grpc.StatusCode.UNKNOWN, message)
        except Exception as e:

            logging.exception("message")

            print(time.strftime("%c"))
            print('Waiting for next call on port 5000.')

            raise grpc.RpcError(grpc.StatusCode.UNKNOWN, str(e))
        try:

            unique_folder_naming = str(datetime.datetime.now()).replace(':', '-').replace('.', '-') + '^' + str(random.randint(100000000000, 999999999999)) + '_lsa/'

            # thread1 = threading.Thread(target=generate_topics_plsa, args=(docs,unique_folder_naming,num_topics,topic_divider,maxiter))
            p1 = mp.Process(target=generate_topics_lsa, args=(docs,unique_folder_naming,num_topics))

            p1.start()

            resp = topic_analysis_pb2.LSAResponse(status=True, message='success', handle=unique_folder_naming[:-1].replace('-','e').replace(' ','d').replace('^','y'))

            print('\n')
            print('status:',resp.status)
            print('message:',resp.message)
            print('Waiting for next call on port 5000.')
            print('\n')

            return resp
        except Exception as e:

            logging.exception("message")

            print(time.strftime("%c"))
            print('Waiting for next call on port 5000.')

            raise grpc.RpcError(grpc.StatusCode.UNKNOWN, str(e))

def generate_topics_plsa(docs,unique_folder_naming,num_topics,topic_divider,maxiter,beta):

    # Put try catch here and add status

    s = plsa_wrapper.PLSA_wrapper(docs)

    try:

        os.mkdir(s.plsa_parameters_path+unique_folder_naming)

        # 1/0

        with open(s.plsa_parameters_path+unique_folder_naming+'status.txt','w') as f:
            f.write('Analysis started.')

        s.unique_folder_naming = unique_folder_naming
        s.num_topics = num_topics
        s.topic_divider = topic_divider
        s.max_iter = maxiter
        s.beta = beta
        s.write_to_json()
        s.generate_topics_json()

    except Exception as e:

        logging.exception("message")

        with open(s.plsa_parameters_path+unique_folder_naming+'status.txt','w') as f:
            f.write('Failed.')
            f.write('\n')
            f.write(str(e))

def generate_topics_lda(docs,unique_folder_naming,num_topics,topic_divider,iterations, num_words, minimum_probability, 
                minimum_phi_value, per_word_topics, chunksize, passes, alpha, eta, decay, gamma_threshold, coherence, eval_every, update_every, offset):

    # Put try catch here and add status
    #path = str(pathlib.Path(os.path.abspath('')).parents[0])+'/appData/misc/topic_analysis.json'
    lda = lda_wrapper.LDA_wrapper(docs)

    try:
        os.mkdir(lda.lda_parameters_path+unique_folder_naming)
        # 1/0
        with open(lda.lda_parameters_path+unique_folder_naming+'status.txt','w') as f:
            f.write('Analysis started.')
            f.write('\n')

        lda.unique_folder_naming = unique_folder_naming
        
        #s.num_topics = num_topics

        lda.write_to_json()

        lda.generate_topics_gensim(num_topics=num_topics, passes=passes, chunksize=chunksize, per_word_topics=per_word_topics, iterations=iterations, alpha=alpha, eta=eta,
            minimum_probability=minimum_probability, minimum_phi_value=minimum_phi_value, gamma_threshold=gamma_threshold, eval_every=eval_every, update_every=update_every,
            offset=offset,num_words=num_words, coherence=coherence)

    except Exception as e:

        logging.exception("message")

        with open(lda.lda_parameters_path+unique_folder_naming+'status.txt','w') as f:
            f.write('Failed.')
            f.write('\n')
            f.write(str(e))

def generate_topics_lsa(docs,unique_folder_naming,num_topics):

    # Put try catch here and add status
    #path = str(pathlib.Path(os.path.abspath('')).parents[0])+'/appData/misc/topic_analysis.json'
    lsa = lsa_wrapper.LSA_wrapper(docs)

    try:
        os.mkdir(lsa.lsa_parameters_path+unique_folder_naming)
        # 1/0
        with open(lsa.lsa_parameters_path+unique_folder_naming+'status.txt','w') as f:
            f.write('Analysis started.')
            f.write('\n')

        lsa.unique_folder_naming = unique_folder_naming
        lsa.num_topics = num_topics
        lsa.write_to_json()
        lsa.generate_topics_gensim(num_topics=num_topics)

    except Exception as e:

        logging.exception("message")

        with open(lsa.lsa_parameters_path+unique_folder_naming+'status.txt','w') as f:
            f.write('Failed.')
            f.write('\n')
            f.write(str(e))


def serve():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    topic_analysis_pb2_grpc.add_TopicAnalysisServicer_to_server(TopicAnalysis(), server)
    print('Starting server. Listening on port 5000.')
    server.add_insecure_port('127.0.0.1:5000')
    server.start()
    try:
        while True:
            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        server.stop(0)


def serve_test():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    topic_analysis_pb2_grpc.add_TopicAnalysisServicer_to_server(TopicAnalysis(), server)
    print('Starting server. Listening on port 5000.')
    server.add_insecure_port('127.0.0.1:5000')
    return server


__end__ = '__end__'


if __name__ == '__main__':


    serve()

    pass













