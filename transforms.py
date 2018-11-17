# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 19:22:36 2018
context = article_content;    cleaned_context_max_length = 292->300; 

@author: Lenovo
"""
import tensorflow as tf
import pandas as pd
import json
import numpy as np
import jieba
import gensim
from string import punctuation

#data = np.load('F:\\portfolio\\tensorflow_practice_demos\\zhongdian28reading\\dataarr.npy')
#data = json.load('/media/chenjiazheng/906E955B6E953B44/portfolio/tensorflow_practice_demos/zhongdian28reading/dataarr.npy')

def read_data():
    
    #path = "/media/chenjiazheng/906E955B6E953B44/portfolio/kaggleFiles/中电28机器阅读/trainingDataSet.json"
    
    path = "F:/portfolio/kaggleFiles/中电28机器阅读/trainingDataSet.json"
    dataopen = open(path,'r',encoding='utf-8').read();
    dataj = json.loads(dataopen,strict=False)
    dataf = pd.DataFrame(dataj)
    return dataf;
    

def w2v(passage):
    
    model = gensim.models.KeyedVectors.load_word2vec_format('F:/portfolio/References/w2vDB/sgns.baidubaike.bigram-char', binary=False)

    #model = gensim.models.KeyedVectors.load_word2vec_format('/media/chenjiazheng/906E955B6E953B44/portfolio/References/w2vDB/sgns.baidubaike.bigram-char', binary=False);
    
    passage = list( jieba.cut(passage, cut_all=False, HMM=True) );
    phrase = []
    for p in passage:
        try:    phrase.append( model[p] )
        except:
            try: phrase.append( model[p] )
            except:
                for e in p:
                   phrase.append( model[e] ) ;
    dictionaire = {'word':passage, 'vector':phrase}
    return phrase, dictionaire;

def wash(message):
    # 定义要删除的标点等字符
    add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
    all_punc = punctuation + add_punc
    all_punc = [e for e in all_punc]
    #mes = [m for m in massage if not m in all_punc] 
    mess = []
    for m in message:
        if m not in all_punc:
            mess.append(m);
    mes = ''
    for m in mess:
        if m not in all_punc:
            mes = str(mes + str(m) );
    return mes

def clean(message):
    # 定义要删除的标点等字符
    add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
    all_punc = punctuation + add_punc
    #mes = [m for m in massage if not m in all_punc] 
    mess = []
    for m in message:
        if m not in all_punc:
            mess.append(m);
    mes = ''
    for m in mess:
        if m not in all_punc:
            mes += m;
    seg = jieba.cut(mes)
    re = []
    for e in seg:
        re.append(e);
    return re;

def preprare(message):
    prepared = w2v( clean(message) )
    return prepared;

def start_idx(context, answer):
    #context = list(clean(context))   
    try: no = context.index(answer)
    except: no=0
    return no;

#l = str('你是谁爸呀?我是你爸爸')
#ls = ['1','2','1','4']
#
#ans  = u'呀'
#idx = convert_idx(l, ans)
#id2 = ls.index('1',1)
#l2 = str('why you are so cute?')
#tk = word_tokenize(l2)
#n = l.find('爸',9)
#s = start_idx(l, '爸') 
#l2 = str('one one one one')
#n2 = l2.find('one')

           
        
'''
data = read_data() #data.size = (20000,5)
p1 = data.iloc[1]
#questions = pd.DataFrame(data['questions'])
questions = data['questions']
quest1 = pd.DataFrame( questions[1] )
sth = quest1['question']
qs = []
for q in questions:
    q = pd.DataFrame(q)
    quest = q['question']
    for u in quest:
        u = clean(u)
        qs.append(u);
q_maxlen = len( max(qs)) # == 7
#context = clean(context)
maxc = max(context) #673
phrs,dictio = w2v(context)

vector = w2v(u'知')
'''
'''
class config(object):
    #N, PL, QL, CL, d, dc, nh = config.batch_size
    def __init__(self):
        self.test_para_limit = 20
        self.test_ques_limit = 20
        self.char_limit = 4
        self.batch_size = [10, 4] # [N, CL]
        self.learning_rate = 0.01
        self.grad_clip = 0.1
        self.ans_limit = 20
        self.l2_norm = 0
        self.decay = 0 
        self.hidden = 4;
        
    def get_batch_size(self):
        N = 10
        PL = 20
        QL = 7
        CL = 4
        d = 128
        dc = 300
        nh = 8
        return [N, PL, QL, CL, d, dc, nh];
        
        
class batch(object):
    def __init__(self):
        #self.data = np.load('/media/chenjiazheng/906E955B6E953B44/portfolio\\tensorflow_practice_demos\\zhongdian28reading\\dataarr.npy')
        self.data = read_data();

    def get_batch(self,n): 
        batch = self.data[n]
        return batch;
    
    def get_next(self,n,m,k): 
        #self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()#
        next_batch = self.data[n]
        c = self.get_context()
        q = self.get_question()
        ch = list(c)
        qh = list(q)
        y1 = self.get_answer(m)[0]
        y2 = self.get_answer(m)[-1]
        qa_id = self.get_qa_id[n,m][k] #??
        return next_batch;
        
    def get_context(self, n):
        bc = self.get_batch(n)
        content = bc['article_content']
        context = prepro(content)
        return context;
    
    def get_question(self, n):
        bc = self.get_batch(n)
        questions = pd.DataFrame(bc['questions'])
        quest = questions['question']
        return quest;
    
    def get_answer(self, n):      
        bc = self.get_batch(n)
        questions = pd.DataFrame(bc['questions'])
        ans = questions['answer']
        return ans;
    
    def get_qa_id(self, n, m):
        bc = self.get_batch(n)
        questions = pd.DataFrame(bc['questions'])
        qa_id = questions['questions_id']
        return qa_id;
        
    def get_charmat():
        pass;
'''
       