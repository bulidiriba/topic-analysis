# Tested on python3.6

import unittest
import grpc

import json
import time

#from service_spec import topic_analysis_pb2
#from service_spec import topic_analysis_pb2_grpc

import topic_analysis_pb2
import topic_analysis_pb2_grpc

import topic_analysis_grpc
import analysis_results

sleep_time_secs = 10 # This is to allow for topic models to be generated before unit testing occurs in the following code

class TestTopicAnalysisGrpc(unittest.TestCase):

    def setUp(self):

        self.app = analysis_results.app.test_client()
        self.docs = []
        self.docs_2 = None
        self.num_topics = 5
        self.maxiter = 22
        self.beta = 1
        self.per_word_topic = 300

        sample_doc = 'docs/tests/test_doc.txt'
        
        with open(sample_doc,'r') as f:
            self.docs = f.read().splitlines()

        self.docs = list(filter(lambda a: a != '', self.docs))

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

        su = 0
        for i in range(len(resp2_data['wordByTopicConditional'])):
            a = sum(resp2_data['wordByTopicConditional'][i])
            print("summation of probability term in topic: ", i, "is", a)
            su += a 
        print("their summa", su)

        print("lentgh", len(resp2_data['wordByTopicConditional'][0]))

        self.assertEqual(resp2_data['status'],'Topic analysis finished.')
        self.assertGreater(resp2_data['total running time in minutes'],0.0)
        #self.assertEqual(resp2_data['docs_list'], [str(i) for i in range(0,44)])
        #self.assertEqual(len(resp2_data['topics']),2)
        self.assertIsInstance(resp2_data['topics'][0],str)
        self.assertIsInstance(resp2_data['topics'][1],str)
        #self.assertEqual(len(resp2_data['topicByDocMatirx']),2)
        #self.assertEqual(len(resp2_data['topicByDocMatirx'][0]),44)
        #self.assertAlmostEqual(sum(sum(resp2_data['topicByDocMatirx'],[])),1.0,delta=0.1)
        #print('sum of p(z,d)=',sum(sum(resp2_data['topicByDocMatirx'],[])))
        self.assertAlmostEqual(resp2_data['topicProbabilities'][0]+ resp2_data['topicProbabilities'][1],1.0,delta=0.1)
        self.assertEqual(len(resp2_data['wordByTopicConditional']), self.num_topics)
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.per_word_topic)
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
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.per_word_topic)
        self.assertAlmostEqual(sum(sum(resp2_data['wordByTopicConditional'], [])), 1.0, delta=0.1)
        print('sum of p(w|z)=',sum(sum(resp2_data['wordByTopicConditional'],[])))
        #self.assertEqual(len(resp2_data['logLikelihoods']),23)
        #for i in range(0,23):
        #    self.assertLess(resp2_data['logLikelihoods'][i],0)
        print("\n\t\t###END of PLSA Processing\n")

    def test_lda_response_format_grpc(self):

        print("\n\n\t\tLDA Processing.......\n")

        print("LDA Test for Tokenized text input\n")
        
        lda_request = topic_analysis_pb2.LDARequest(docs=self.docs, num_topics=self.num_topics, maxiter=self.maxiter)
        resp = self.stub.LDA(lda_request)

        print('////////////// Sleeping till topic analysis finishes')
        time.sleep(sleep_time_secs)
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\  Wide awake now')

        self.assertEqual([resp.status,resp.message],[True,'success'])

        resp2 = self.app.get('/topic-analysis/api/v1.0/results?handle='+resp.handle)
        resp2_data = json.loads(resp2.get_data(as_text=True))
        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')

        su = 0
        for i in range(len(resp2_data['wordByTopicConditional'])):
            a = sum(resp2_data['wordByTopicConditional'][i])
            print("summation of all probability term in topic: ", i, "is", a)
            su += a 

        print("their total summation is", su)

        print("lentgh", len(resp2_data['wordByTopicConditional'][0]))

        print("sum of all topic probability given corpus: ", sum(resp2_data['topicProbabilities'][i] for i in range(self.num_topics)))

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
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.per_word_topic)

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
        print("\n\nLDA Test for untokenized text input\n")

        lda_request = topic_analysis_pb2.LDARequest(docs=self.docs_2, num_topics=self.num_topics, maxiter=self.maxiter)

        resp = self.stub.LDA(lda_request)

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
        self.assertEqual(len(resp2_data['wordByTopicConditional'][0]), self.per_word_topic)

        #self.assertAlmostEqual(sum(sum(resp2_data['wordByTopicConditional'], [])), 1.0, delta=0.1)
        #print('sum of p(w|z)=',sum(sum(resp2_data['wordByTopicConditional'],[])))
        
        '''
        self.assertEqual(len(resp2_data['logLikelihoods']),23)
        for i in range(0,23):
            self.assertLess(resp2_data['logLikelihoods'][i],0)
        '''
        print("\n\t\t####End of LDA Processing\n\n")

__end__ = '__end__'

if __name__ == '__main__':
    unittest.main()