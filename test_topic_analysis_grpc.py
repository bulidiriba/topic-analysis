# Tested on python3.6

import unittest
import grpc

import json
import time
import numpy as np

#from service_spec import topic_analysis_pb2
#from service_spec import topic_analysis_pb2_grpc

import topic_analysis_pb2
import topic_analysis_pb2_grpc

import topic_analysis_grpc
import analysis_results

sleep_time_secs = 10 # This is to allow for topic models to be generated before unit testing occurs in the following code

class TestTopicAnalysisGrpc(unittest.TestCase):

    def setUp(self):

        # common
        self.app = analysis_results.app.test_client()
        self.docs = [] # this is for tokenized one
        self.docs_2 = None # this is for untokenized one
        self.num_topics = 2
        self.topic_divider = 0
        
        # specific to plsa
        self.maxiter = 22
        self.beta = 1
        
        
        # specific to lda
        self.iterations = 50
        self.num_words = 300
        self.minimum_probability = 0.0
        self.minimum_phi_value = 0.01
        self.per_word_topics = True
        self.chunksize = 200
        self.passes = 22
        self.alpha='auto'
        self.eta='auto'
        self.decay = 0.5
        self.gamma_threshold=0.001
        self.coherence = "u_mass"
        self.eval_every = 1
        self.update_every=0
        self.offset=1.0

        # specific to lsa



        sample_doc = 'docs/tests/test_doc 2.txt'
        #sample_doc = 'docs/tests/test_doc.txt'
        #sample_doc = 'docs/tests/test_doc_2.txt'
        #sample_doc = 'docs/tests/topic_analysis.json'
        #sample_doc = 'docs/tests/topic_analysis_2.json'

        # check that if the given file name is .txt or .json file
        splitted_filename = sample_doc.split(".")
        # if the file is .json
        if splitted_filename[-1] == "json":
            with open(sample_doc, "r") as read_file:
                fileList = json.load(read_file)

            self.docs = fileList['docs']

        # if the file is .txt
        elif splitted_filename[-1] == "txt":
            with open(sample_doc, "r") as file:
                for cnt, line in enumerate(file):
                    self.docs.append(line)

        else:
            print("your file format is not supported")
            exit(0)
            
        #with open(sample_doc,'r') as f:
        #    self.docs = f.read().splitlines()
        
        self.docs = list(filter(lambda a: a != '', self.docs))
        
        # now add for untokenized one
        with open(sample_doc,'r') as f:
            self.docs_2 = [f.read()]


        channel = grpc.insecure_channel('localhost:5000')
        self.stub = topic_analysis_pb2_grpc.TopicAnalysisStub(channel)

        self.server = topic_analysis_grpc.serve_test()
        self.server.start()

    def tearDown(self):
        self.server.stop(0)
        print('Server stopped')

    def test_plsa_response_format_grpc(self):

        print("\n\n\t\tPLSA Processing.......\n")
        print("PLSA Test for Tokenized text input\n")
        
        plsa_request = topic_analysis_pb2.PLSARequest(docs=self.docs, num_topics=self.num_topics, maxiter=self.maxiter, beta=self.beta)

        resp = self.stub.PLSA(plsa_request)

        print('////////////// Sleeping till topic analysis finishes')
        time.sleep(sleep_time_secs)
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\  Wide awake now')

        self.assertEqual([resp.status,resp.message],[True,'success'])

        resp2 = self.app.get('/topic-analysis/api/v1.0/results?handle='+resp.handle)
        resp2_data = json.loads(resp2.get_data(as_text=True))
        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')

        # to check the summation of all probability distributions of words in each topic.
        summation = np.sum(np.asarray(resp2_data['wordByTopicConditional'], dtype=np.float32), axis=1)
        self.assertAlmostEqual(summation[0], 1.0, delta=0.1)

        self.assertEqual(resp2_data['status'],'Topic analysis finished.')
        self.assertGreater(resp2_data['total running time in minutes'],0.0)
        #self.assertEqual(resp2_data['docs_list'], [str(i) for i in range(0,44)])
        #self.assertEqual(len(resp2_data['topics']),2)
        self.assertIsInstance(resp2_data['topics'][0],str)
        self.assertIsInstance(resp2_data['topics'][1],str)


        # check the length of topics 
        self.assertEqual(len(resp2_data['topicByDocMatirx']),self.num_topics)
        
        # to check the number of documents with number of non-empty docs
        #self.assertEqual(len(resp2_data['topicByDocMatirx'][0]),44)
        
        # change the topic by doc matrix from list to numpy array, to add each of all the coloumn
        topic_doc_array = np.asarray(resp2_data['topicByDocMatirx'], dtype=np.float32)
        # add each of all the column, means all topic probability in each column and then assert with 1
        #self.assertEqual(len(resp2_data['topicByDocMatirx'][0]),44)
        #self.assertAlmostEqual(sum(sum(resp2_data['topicByDocMatirx'],[])),1.0,delta=0.1)
        #print('sum of p(z,d)=',sum(sum(resp2_data['topicByDocMatirx'],[])))
        
        #self.assertAlmostEqual(resp2_data['topicProbabilities'][0]+ resp2_data['topicProbabilities'][1],1.0,delta=0.1)
        self.assertAlmostEqual(sum(resp2_data['topicProbabilities'][i] for i in range(self.num_topics)), 1.0, delta=0.1)
        
        self.assertEqual(len(resp2_data['wordByTopicConditional']), self.num_topics)
        
        # check the length of word in each topic with num_words given, but this cause an error,
        # if the number of vocabulary is less than the defined number of word per topic which is 300
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.num_words)


        self.assertAlmostEqual(sum(sum(resp2_data['wordByTopicConditional'], [])), 1.0, delta=0.1)
        print('sum of p(w|z)=',sum(sum(resp2_data['wordByTopicConditional'],[])))
        #self.assertEqual(len(resp2_data['logLikelihoods']),23)
        #for i in range(0,23):
        #    self.assertLess(resp2_data['logLikelihoods'][i],0)


        # Test for untokenized text input
        print("\n\nPLSA Test for Untokenized text input\n")

        plsa_request = topic_analysis_pb2.PLSARequest(docs=self.docs_2, num_topics=self.num_topics, maxiter=self.maxiter, beta=self.beta)

        resp = self.stub.PLSA(plsa_request)

        print('////////////// Sleeping till topic analysis finishes')
        time.sleep(sleep_time_secs)
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\  Wide awake now')

        self.assertEqual([resp.status,resp.message],[True,'success'])

        resp2 = self.app.get('/topic-analysis/api/v1.0/results?handle='+resp.handle)
        resp2_data = json.loads(resp2.get_data(as_text=True))

        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')

        self.assertEqual(resp2_data['status'],'Topic analysis finished.')
        self.assertGreater(resp2_data['total running time in minutes'],0.0)
        #self.assertEqual(resp2_data['docs_list'], [str(i) for i in range(0,98)])
        self.assertEqual(len(resp2_data['topics']),self.num_topics)
        self.assertIsInstance(resp2_data['topics'][0],str)
        self.assertIsInstance(resp2_data['topics'][1],str)
        #self.assertEqual(len(resp2_data['topicByDocMatirx']),2)
        #self.assertEqual(len(resp2_data['topicByDocMatirx'][0]),98)
        #self.assertAlmostEqual(sum(sum(resp2_data['topicByDocMatirx'],[])),1.0,delta=0.1)
        #print('sum of p(z,d)=',sum(sum(resp2_data['topicByDocMatirx'],[])))
        self.assertAlmostEqual(resp2_data['topicProbabilities'][0]+ resp2_data['topicProbabilities'][1],1.0,delta=0.1)
        self.assertEqual(len(resp2_data['wordByTopicConditional']), self.num_topics)
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.num_words)
        self.assertAlmostEqual(sum(sum(resp2_data['wordByTopicConditional'], [])), 1.0, delta=0.1)
        print('sum of p(w|z)=',sum(sum(resp2_data['wordByTopicConditional'],[])))
        #self.assertEqual(len(resp2_data['logLikelihoods']),23)
        #for i in range(0,23):
        #    self.assertLess(resp2_data['logLikelihoods'][i],0)
        print("\n\t\t###END of PLSA Processing\n")

    def test_lda_response_format_grpc(self):

        print("\n\n\t\tLDA Processing.......\n")

        print("LDA Test for Tokenized text input\n")
        
        lda_request = topic_analysis_pb2.LDARequest(docs=self.docs, num_topics=self.num_topics, iterations=self.iterations, num_words=self.num_words, passes=self.passes, chunksize=self.chunksize, 
            update_every=self.update_every, alpha=self.alpha, eta=self.eta, decay=self.decay, offset=self.offset, eval_every=self.eval_every, gamma_threshold=self.gamma_threshold, minimum_probability=self.minimum_probability,
            minimum_phi_value=self.minimum_phi_value, per_word_topics=self.per_word_topics, coherence=self.coherence)
        resp = self.stub.LDA(lda_request)

        print('////////////// Sleeping till topic analysis finishes')
        time.sleep(sleep_time_secs)
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\  Wide awake now')

        self.assertEqual([resp.status,resp.message],[True,'success'])

        resp2 = self.app.get('/topic-analysis/api/v1.0/results?handle='+resp.handle)
        resp2_data = json.loads(resp2.get_data(as_text=True))
        
        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')

        # -------------Status-----------------
        # check whether the process is fully accomplished, if so the current status is 'Topic analysis finished' and time taken greater than 0.0
        self.assertEqual(resp2_data['status'],'Topic analysis finished.')
        self.assertGreater(resp2_data['total running time in minutes'],0.0)
        

        # ------------Documents--------------
        # check the length of non-empty documents
        print("Non-empty documents: ", len(resp2_data['documentProbabilities']))

        # check the sum of document probabilities whether its 1 or not
        print("sum of P(D): ", np.sum(np.asarray(resp2_data['documentProbabilities'], dtype=np.float32)))
        self.assertAlmostEqual(np.sum(np.asarray(resp2_data['documentProbabilities'], dtype=np.float32)), 1.0, delta=0.1)
                

        # ------------Topics------------------
        # check the datatype of all elements of the topics whether they are string or not(have to be string)
        for i in range(self.num_topics):
            self.assertIsInstance(resp2_data['topics'][i], str)

        # check the length of topics
        self.assertEqual(len(resp2_data['topicByDocMatirx']),self.num_topics)
        self.assertEqual(len(resp2_data['wordByTopicConditional']), self.num_topics)
        self.assertEqual(len(resp2_data['topics']),self.num_topics)
    

        # ------------Topic by Doc Matrix---------------    
        # check if all the summation of topicByDoc probability is equal to one and its Joint Probability P(Z, D)
        print('sum of p(Z,D)=',sum(sum(resp2_data['topicByDocMatirx'],[])))
        self.assertAlmostEqual(sum(sum(resp2_data['topicByDocMatirx'],[])),1.0,delta=0.1)                


        # ----------- Word by Topic Matrix----------
        # check the length of word in each topic with num_words given, but this cause an error,
        # if the number of vocabulary is less than the defined number of word per topic which is 300
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.num_words)
        
        # check if the summation of wordByTopic probability distiribution in each row is 1 and its conditional P(W|Zi) 
        # but the problem here is wordByTopic contains only the probability of 300 top words for that topic, 
        # but actually that topics contains probability for all the words in dictionary. 
        print('sum of p(W|Zi)=', sum(resp2_data['wordByTopicConditional'][0]), " (for only top ", self.num_words, " Words )")
        self.assertAlmostEqual(sum(resp2_data['wordByTopicConditional'][0]), 1.0, delta=0.1)
        

        print("\n\t\t####End of LDA Processing\n\n")

    def test_lsa_response_format_grpc(self):

        print("\n\n\t\tLSA Processing.......\n")

        print("LSA Test for Tokenized text input\n")
        
        lsa_request = topic_analysis_pb2.LSARequest(docs=self.docs, num_topics=self.num_topics)
        resp = self.stub.LSA(lsa_request)

        print('////////////// Sleeping till topic analysis finishes')
        time.sleep(sleep_time_secs)
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\  Wide awake now')

        self.assertEqual([resp.status,resp.message],[True,'success'])

        resp2 = self.app.get('/topic-analysis/api/v1.0/results?handle='+resp.handle)
        resp2_data = json.loads(resp2.get_data(as_text=True))
        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
        
        #print("sum of P(Z): ", sum(resp2_data['topicProbabilities'][i] for i in range(self.num_topics)))

        self.assertEqual(resp2_data['status'],'Topic analysis finished.')
        self.assertGreater(resp2_data['total running time in minutes'],0.0)
        self.assertEqual(len(resp2_data['topics']),self.num_topics)
        for i in range(self.num_topics):self.assertIsInstance(resp2_data['topics'][i], str) 

        #self.assertEqual(resp2_data['docs_list'], [str(i) for i in range(0,44)])
        
        self.assertEqual(len(resp2_data['topicByDocMatirx'][0]),self.num_topics)
        self.assertAlmostEqual(sum(resp2_data['topicByDocMatirx'][0]), 1.0, delta=0.1)
        #self.assertEqual(len(resp2_data['topicByDocMatirx'][0]),44)
        #self.assertAlmostEqual(sum(sum(resp2_data['topicByDocMatirx'],[])),1.0,delta=0.1)
        #print('sum of p(z,d)=',sum(sum(resp2_data['topicByDocMatirx'],[])))
        
        #self.assertAlmostEqual(resp2_data['topicProbabilities'][0]+ resp2_data['topicProbabilities'][1],1.0,delta=0.1)
        self.assertAlmostEqual(sum(resp2_data['topicProbabilities'][i] for i in range(self.num_topics)), 1.0, delta=0.1)
        
        self.assertEqual(len(resp2_data['wordByTopicConditional']), self.num_topics)
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.num_words)

        #self.assertAlmostEqual(sum(sum(resp2_data['wordByTopicConditional'], [])), 1.0, delta=0.1)

        # the problem of this is to test with one we need to sum up all the probability of 
        # term in the dictionary but now we are summing up only the probability of 300 words 
        #self.assertAlmostEqual(sum(resp2_data['wordByTopicConditional'][0]), 1.0, delta=0.1)

        #print('sum of p(w|z)=',sum(sum(resp2_data['wordByTopicConditional'],[])))
        
        print('sum of p(w|z)=', sum(resp2_data['wordByTopicConditional'][0]))
        
        '''
        self.assertEqual(len(resp2_data['logLikelihoods']),23)
        for i in range(0,23):
            self.assertLess(resp2_data['logLikelihoods'][i],0)
        '''

        # Test for untokenized text input
        print("\n\nLSA Test for untokenized text input\n")

        lsa_request = topic_analysis_pb2.LSRequest(docs=self.docs_2, num_topics=self.num_topics, maxiter=self.maxiter)

        resp = self.stub.LSA(lsa_request)

        print('////////////// Sleeping till topic analysis finishes')
        time.sleep(sleep_time_secs)
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\  Wide awake now')

        self.assertEqual([resp.status,resp.message],[True,'success'])

        resp2 = self.app.get('/topic-analysis/api/v1.0/results?handle='+resp.handle)
        resp2_data = json.loads(resp2.get_data(as_text=True))
        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')

        self.assertEqual(resp2_data['status'],'Topic analysis finished.')
        self.assertGreater(resp2_data['total running time in minutes'],0.0)
        self.assertEqual(len(resp2_data['topics']),self.num_topics)
        for i in range(self.num_topics):self.assertIsInstance(resp2_data['topics'][i], str) 
        
        #self.assertEqual(resp2_data['docs_list'], [str(i) for i in range(0,98)])

        self.assertEqual(len(resp2_data['topicByDocMatirx'][0]),self.num_topics)
        #self.assertEqual(len(resp2_data['topicByDocMatirx'][0]),98)
        #self.assertAlmostEqual(sum(sum(resp2_data['topicByDocMatirx'],[])),1.0,delta=0.1)
        #print('sum of p(z,d)=',sum(sum(resp2_data['topicByDocMatirx'],[])))
        
        self.assertAlmostEqual(sum(resp2_data['topicProbabilities'][i] for i in range(self.num_topics)), 1.0, delta=0.1)
        
        self.assertEqual(len(resp2_data['wordByTopicConditional']), self.num_topics)
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.num_words)

        #self.assertAlmostEqual(sum(sum(resp2_data['wordByTopicConditional'], [])), 1.0, delta=0.1)
        #print('sum of p(w|z)=',sum(sum(resp2_data['wordByTopicConditional'],[])))
        
        '''
        self.assertEqual(len(resp2_data['logLikelihoods']),23)
        for i in range(0,23):
            self.assertLess(resp2_data['logLikelihoods'][i],0)
        '''
        print("\n\t\t####End of LSA Processing\n\n")

__end__ = '__end__'

if __name__ == '__main__':
    unittest.main()