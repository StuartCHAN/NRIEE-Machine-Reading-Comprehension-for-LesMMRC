import array
import gc
import json as js
import random
from codecs import open
from collections import Counter
from string import punctuation

import gensim
import jieba
import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from tqdm import tqdm

import transforms as trans
import ujson as json


'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''
'''
nlp = spacy.blank("en")
def word_tokenize(sent):##1
    doc = nlp(sent)
    return [token.text for token in doc]
'''
def word_tokenize(message):
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

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            #raise Exception()
            try: current = tokens.index(token,current)
            except:
                print("Token {} cannot be found".format(token))
                try: current = tokens.index(token)
                except: current = int(0);
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    #with open(filename, "r", encoding='utf-8').read() as fh: #
    path = "F:\\portfolio\\kaggleFiles\\中电28机器阅读\\trainingDataSet.json"
    dataopen = open(path,'r',encoding='utf-8').read()
    source = js.loads(dataopen,strict=False)
    '''
    answers = []
    for para in source:
        for qas in para["questions"]:
            vs = list(qas.values())
            answers.append(vs[2])
            #answers.append(vs[-1])
     '''             
    #source = json.loads(fh, strict=False ) #
    for article in tqdm(source):
        #for para in article:
        para = article
        #context = para["article_content"].replace("''", '" ').replace("``", '" ')
        context = trans.wash(para["article_content"])
        context_tokens = word_tokenize(context)
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        for token in context_tokens:
            word_counter[token] += len(para["questions"])
            for char in token:
                char_counter[char] += len(para["questions"])
        for qa in para["questions"]:
            total += 1
            ques = qa["question"].replace(
                "''", '" ').replace("``", '" ')
            ques_tokens = word_tokenize(ques)
            ques_chars = [list(token) for token in ques_tokens]
            for token in ques_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1
            y1s, y2s = [], []
            answer_texts = []
            #for answer in qa["answer"]:
            answer = list(qa.values())[2] # ==qa["answer"]
            
            answer_text = answer
            answer_start = trans.start_idx(context, answer)#answer['answer_start']
            answer_end = answer_start + len(answer_text)
            answer_texts.append(answer_text)
            answer_span = []
            for idx, span in enumerate(spans):
                if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(idx)
            try:
                y1, y2 = answer_span[0], answer_span[-1]
            except:
                y1=0; y2=0;
            y1s.append(y1)
            y2s.append(y2);
                
            example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                       "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
            examples.append(example)
            eval_examples[str(total)] = {
                "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["questions_id"]};
                    
    random.shuffle(examples)
    print("{} questions in total".format(len(examples)))
    return examples, eval_examples

'''
def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                      token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict
'''
def get_embedding(counter):   
    model = gensim.models.KeyedVectors.load_word2vec_format('F:/portfolio/References/w2vDB/sgns.baidubaike.bigram-char', binary=False)
    #model = gensim.models.KeyedVectors.load_word2vec_format('/media/chenjiazheng/906E955B6E953B44/portfolio/References/w2vDB/sgns.baidubaike.bigram-char', binary=False);
    print('the word to vectors model is OK')
    passage = [k for k, v in counter.items()]
    emb_mat = []
    token2idx_dict = {}
    print('begin to embed...')
    for p in passage:
        try:
            emb=model[p] 
        except:
            emb= np.random.rand(300) ;
        emb_mat.append(emb) 
        token2idx_dict[p] = emb;
    print('this embedding is ok')
    return emb_mat, token2idx_dict;


def convert_to_features(config, data, word2idx_dict, char2idx_dict):
    print('begin to convert the features...')
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.test_para_limit
    ques_limit = config.test_ques_limit
    ans_limit = 100
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1
	
    i1 = int(0)
    for token in enumerate(example["context_tokens"]):
        context_idxs[i1] = _get_word(token)
    i1 = int(i1+1)

    i2 = int(0)
    for token in enumerate(example["ques_tokens"]):
        ques_idxs[i2] = _get_word(token)
        i2 = int(i2+1)

    i3 = int(0)
    for token in enumerate(example["context_chars"]):
        j = int(0)
        for char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i3, j] = _get_char(char)
            j = int(j+1)
        i3 = int(i3+1)
	
    i4 = int(0)
    for token in enumerate(example["ques_chars"]):
        j = int(0)
        for char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i4, j] = _get_char(char)
            j = int(j+1)
        i4 = int(i4+1)
    print('these features are converted.')
    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs

def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    print('begin to build the features...')
    para_limit = 400#config.test_para_limit if is_test else config.para_limit
    ques_limit = 50#config.test_ques_limit if is_test else config.ques_limit
    ans_limit = 30 #if is_test else config.ans_limit
    char_limit = 16#config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               (example["y2s"][0] - example["y1s"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = list(np.zeros([para_limit], dtype=np.int64) )
        context_char_idxs = list(np.zeros([para_limit, char_limit], dtype=np.int64) )
        context_char_idxs = [list(e) for e in context_char_idxs]#
        ques_idxs = list(np.zeros([ques_limit], dtype=np.int64) )
        ques_char_idxs = list(np.zeros([ques_limit, char_limit], dtype=np.int64) )
        ques_char_idxs = [list(e) for e in ques_char_idxs]#
        y1 = list(np.zeros([para_limit], dtype=np.int64) )
        y2 = list(np.zeros([para_limit], dtype=np.int64) )
        
#        context_idxs =  np.zeros([para_limit], dtype=np.int32) 
#        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32) 
#         
#        ques_idxs = np.zeros([ques_limit], dtype=np.int32) 
#        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32) 
#         
#        y1 = np.zeros([para_limit], dtype=np.float32) 
#        y2 = np.zeros([para_limit], dtype=np.float32) 

        def _get_word(word):
            for each in (word):  #
                if each in word2idx_dict.keys():
                    return np.array(word2idx_dict[each])#
            return 1

        def _get_char(char):
            if char in char2idx_dict.keys():
                return np.array(char2idx_dict[char])# 
            return 1

        context_tokens = example["context_tokens"]
        i1 = 0
        for  token in list(context_tokens):            
            #i = int(context_tokens.index(token) )
            if i1 == para_limit: break;
            context_idxs[i1] = _get_word(token)
            i1 += 1
        #? ValueError: setting an array element with a sequence.

        ques_tokens = list(example["ques_tokens"])
        i2 = 0
        for token in ques_tokens:            
            #i = int(ques_tokens.index(token))
            if i2 == ques_limit: break;
            ques_idxs[i2] = _get_word(token)
            i2 += 1

        context_chars = list(example["context_chars"])
        i3 = 0
        for token in context_chars: 
            if i3 == para_limit: break;
            #i = int(context_chars.index(token))			
            j1 = 0
            for char in list(token):
                #j = int(token.index(char))				
                if j1 == char_limit:
                    break
                context_char_idxs[i3][j1] = _get_char(char)
                j1 += 1
            i3 += 1;
            
        ques_chars = list( trans.wash(example["ques_chars"]) )
        i4 = 0
        for token in ques_chars: 
            if i4 == ques_limit: break;
            #i = int(ques_chars.index(token))			
            j2 = 0
            for char in list(token):
                #j = int(token.index(char))				
                if j2 == char_limit:
                    break
                try:
                    ques_char_idxs[i4][j2] = _get_char(char)
                except:
                    try:ques_char_idxs[i4].extend( _get_char(char))
                    except: break;
                j2 += 1
            i4 += 1;
                    
        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1, 1
        
        
        context_idxs = np.array( context_idxs )
        ques_idxs = np.array( ques_idxs ) 
        '''
        cc_idx = []
        for ele in context_char_idxs:
            p = []
            for e in ele:
                cc = bytes(e)
                p.append(cc)
            cc_idx.append(p);'''
        context_char_idxs = np.array(context_char_idxs)
                
        #ques_char_idxs = array.array('b',ques_char_idxs)
        '''
        qc_idx = []
        for ele in ques_char_idxs:
            p = []
            for e in ele:
                qc = bytes(e)
                p.append(qc)
            qc_idx.append(p);'''
        ques_char_idxs = np.array(ques_char_idxs)
        
        y1, y2 = np.array(y1), np.array(y2)

        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        writer.write(record.SerializeToString())
        
#        record_feature={
#              "context_idxs": context_idxs,
#              "ques_idxs": ques_idxs,
#              "context_char_idxs": context_char_idxs,
#              "ques_char_idxs": ques_char_idxs,
#              "y1": y1,
#              "y2": y2,
#              "id": example["id"]
#              }   
#      
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    print('the features are built now')
    return meta,record;


def save(filename, obj, message=None):
    print('the saving begins...')
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)
    print('OK saved')

class ExtendJSONEncoder(js.JSONEncoder):#https://juejin.im/post/5a06d4776fb9a04515435afe
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return list(obj)
        return super(ExtendJSONEncoder, self).default(obj);




def preprocess(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file( config.train_file, "train", word_counter, char_counter )
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.train_examples_file, train_examples, message='train examples')

	
    print('let items get embedding... ')
    word_emb_mat, word2idx_dict = get_embedding(word_counter)
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
	
    char_emb_mat, char2idx_dict = get_embedding(char_counter)
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")

    print('let meta get prepared... ')
    dev_meta = build_features(config, train_examples, "train", config.train_record_file, word2idx_dict, char2idx_dict)    
    save(config.dev_meta, dev_meta, message="dev meta")
    
    end = 'The preprocessing is ok now'
    return end;
    
def prepro(config):
    p = preprocess(config)
    print(p);
