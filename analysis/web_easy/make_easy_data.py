import pandas as pd
import csv
import pickle
import numpy as np
import json
import pprint

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import copy

from easy.key import EASY_KEY, MEDIUM_KEY, ONTOLOGY_KEY, USER_KEY


EASY_Q_BLOCK = ['q_block1', 'q_block2', 'q_block3', 'q_block4', 'q_block5', 'q_block6', 'q_block7', 'q_block8', 'q_block9', 'q_block10', 'q_block11', 'q_block12', 'q_block13',
           'q_block14', 'q_block15', 'q_block16', 'q_block17', 'q_block18', 'q_block19', 'q_block20']

EASY_ID = [1, 2, 3]


EASY_HIST = ['HIST0', 'HIST1', 'HIST2', 'HIST3', 'HIST4', 'HIST5', 'HIST6', 'HIST7', 'HIST8', 'HIST9', 'HIST10', 'HIST11', 'HIST12', 'HIST13', 'HIST14', 'HIST15', 'HIST16',
             'HIST17', 'HIST18', 'HIST19']

EASY_BLOCK = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12',
                  'block13', 'block14', 'block15', 'block16', 'block17', 'block18', 'block19', 'block20']

SOA_BLOCK = ['block1_soa', 'block2_soa', 'block3_soa', 'block4_soa', 'block5_soa', 'block6_soa', 'block7_soa', 'block8_soa', 'block9_soa', 'block10_soa',
             'block11_soa', 'block12_soa', 'block13_soa', 'block14_soa', 'block15_soa', 'block16_soa', 'block17_soa', 'block18_soa', 'block19_soa', 'block20_soa']

EASY_TASK = 'easy-task'
MEDIUM_TASK = 'medium-task'
HARD_TASK = 'hard-task'
ONTOLOGY_TASK = 'ontology-task'
USERS = 'users'

EASY = 20  # Number of Question Set
EASY_Q = 3  # Number of Questions - each set
EASY_SCORE = 8

MEDIUM = 20
MEDIUM_Q = 3
MEDIUM_SCORE = 8

HARD = 20
HARD_Q = 4

ONTO = 12
ONTO_Q = 4
ONTO_SCORE = 8

# JSON_PATH = './config/idea-lab-35902-firebase-adminsdk-8tx4h-8c702e856c.json'
JSON_PATH = 'idea-lab-35902-firebase-adminsdk-8tx4h-8c702e856c.json'
# JSON_PATH =  'flutterstudy-4e686-6caf7d2e98d6.json'
cred = credentials.Certificate(JSON_PATH)

firebase_admin.initialize_app(cred)
db = firestore.client()









def getUserAge(time_user):
    millisecond_len = 13
    t = time_user[:millisecond_len]
    u = time_user[millisecond_len:]

    _doc = db.collection(USERS)
    docs = _doc.stream()

    for doc in docs:
        # user_id <- sha
        user_id = doc.id

        if user_id == u:
            datas = doc.to_dict()
            return datas['age']

def getUserSex(time_user):

    millisecond_len = 13
    t = time_user[:millisecond_len]
    u = time_user[millisecond_len:]

    _doc = db.collection(USERS)
    docs = _doc.stream()

    for doc in docs:
        # user_id <- sha
        user_id = doc.id

        if user_id == u:
            datas = doc.to_dict()
            return datas['sex']

def _ans_save_tot(data, block_size, question_size, score_size):
    res = np.empty([block_size, question_size], dtype=np.object)
    for block in range(block_size):

        for question in range(question_size):


            score = []
            for idx in range(score_size):
                s = data['BLOCK' + str(block) + '_QUESTION' + str(question) + '_SCORE']
            s = ['0' if _s == '' else _s for _s in s]
            score.append(s)
            res[block][question] = score
    print(res.shape)
    return res

def q_block_question(datas, Q_BLOCK, _ID):
    q_block_question = []
    for q in Q_BLOCK:
        q_temp = []
        for id in _ID:
            if (datas[q][id - 1]['_id'] == id):
                q_temp.append(datas[q][id - 1]['question'])
        q_block_question.append(q_temp)
    return q_block_question

def q_block_answer(datas, Q_BLOCK, _ID):
    q_block_answer = []
    for q in Q_BLOCK:
        q_temp = []
        for id in _ID:
            if (datas[q][id - 1]['_id'] == id):
                q_temp.append(datas[q][id - 1]['answer_list'])
                #print("q: ", q)
               # print("datas[q][id - 1]['answer_list']: ", datas[q][id - 1]['answer_list'])
        q_block_answer.append(q_temp)
    return q_block_answer



def easy_block(datas, EASY_BLOCK):
    easy_block = []
    for b in EASY_BLOCK:
        easy_block.append(datas[b])
    return easy_block




def _HIST_schedule(datas, EASY_HIST):
    hist = []
    for h in EASY_HIST:
        hist.append(datas[h])
    return hist

#answer list에서 sti에 해당하는 위치를 담은 배열 리턴


#answer_position과 q_block_answer, order를 가져온다
#answer_position과의 word를 알아내어 order['word'] = 숫자(0~19)
#sequence_id에 해당하는 node에서 node[숫자]

#q_block_question에 answer_position 넣어서 나온 word
#img_number_matching에 word를 넣어 나온 숫자
#배열[숫자] = score




#실험에 걸린 전체 시간을 계산한다.(분으로 반환)
def total_time(datas, KEY):
    start = 0
    finish = 0
    for k in KEY:
        if ("BLOCK0_QUESTION0_REACTION_TIME" in k):
            start = datas[k][0]
        elif ("BLOCK19_QUESTION2_REACTION_TIME" in k):
            finish = datas[k][9]
    # print("시간 차: ", (finish - start) / 1000, "초")
    # print("시간 차: ", ((finish - start) / 1000) / 60, "분")
    return((finish - start))

#각 질문 당 걸린 reaction time을 계산한다.
def reaction_time(datas, KEY):
    time = []
    for k in KEY:
        if ("REACTION_TIME" in k):
            time.append((datas[k][9] - datas[k][0])/1000)
    return(time)



#{'out': [sti, hist number]}
#key값은 output이 되고, value는 [stimulus, sti에 부여된 넘버]이다(0 ~ 4)
#20 번째 블록: {'assets/seed_images/038.png': [['assets/seed_images/036.png', 2], ['assets/seed_images/034.png', 0]], 'assets/seed_images/035.png': [['assets/seed_images/039.png', 3], ['assets/seed_images/040.png', 1]], 'assets/seed_images/037.png': ['assets/seed_images/033.png', 4]}
def out_sti_number_matching(HIST_schedule, easy_block):
    word_number = []
    for i in range(20):
        tmp = {}
        print("easy_block_len: ", (easy_block[0]))
        print("HIST_schedule: ", (HIST_schedule[0]))
        for j in range(len(easy_block[i])):
            if(easy_block[i][j]['out'] not in tmp.keys()):
                tmp[easy_block[i][j]['out']] = [easy_block[i][j]['sti'], int(HIST_schedule[i][j])]
            else:
                print("tmp[easy_block[i][j]['out']][0]: ", tmp[easy_block[i][j]['out']][0])
                print("tmp[easy_block[i][j]['out']]: ", tmp[easy_block[i][j]['out']])
                if(len(tmp[easy_block[i][j]['out']][0]) > 20):#길이가 20이상이면 이미지라는 뜻(= output에 대해 sti가 아직 하나밖에 없다는 뜻)이고 20이하면 배열이라는 뜻이다.
                    if(tmp[easy_block[i][j]['out']][0] != easy_block[i][j]['sti']):#아직 저장되지 않은 sti일 때
                        tmp[easy_block[i][j]['out']] = [tmp[easy_block[i][j]['out']], [easy_block[i][j]['sti'], HIST_schedule[i][j]]]
        word_number.append(tmp)
    print("check_word_number: ", word_number)
    return word_number

#{'sti': hist number}
#key값은 stimulus가 되고 value는 stimulus의 넘버이다.
#16 번째 블록: {'assets/seed_images/001.png': 0, 'assets/seed_images/005.png': 2, 'assets/seed_images/006.png': 1, 'assets/seed_images/002.png': 3}
def sti_number_matching(HIST_schedule, easy_block):
    word_number = []
    for i in range(20):
        tmp = {}
        for j in range(len(easy_block[i])):
            if(easy_block[i][j]['sti'] not in tmp.keys()):
                tmp[easy_block[i][j]['sti']] = int(HIST_schedule[i][j])
        word_number.append(tmp)
    #print("sti_number_matching: ", word_number)
    return word_number


#보기 중 중복되는 img를 찾아 한 번도 display되지 않은 sti를 알아낸다.(out_sti_number 사용)
#input으로 out_sti_number을 사용하는데, 깊은 복사를 하지 않기 떄문에 out_sti_number에 덮어쓰게 된다.
def find_no_display_out(out_sti_number, trial_info_detail, q_block_answer, q_block_question):
    for i in range(len(out_sti_number)):#sbj의 모든 out_sti_num에 대해서
        print("out_sti_number[i].keys(): ", out_sti_number[i].keys())#한번도 보여지지 않은 out-sti는 배열에 저장되어 있지 않다. 따라서 keys에 포함되지 않음
        keys = out_sti_number[i].keys()#key는 output(총 3개)이다.
        question = q_block_question[i]#question == output
        print("check_quesiotn: ", question)
        sti = []
        if(trial_info_detail[i][0] == 7):#oneshot-
            for key in keys:
                if(len(out_sti_number[i][key][0]) == 2):#길이가 2이면 배열이라는 뜻이다. 즉, 2개의 sti가 있다는 뜻
                    print("out_sti_number[i][key][0]: ", out_sti_number[i])
                    sti.append(out_sti_number[i][key][0][0])#첫번째 sti의 이미지를 저장
                    sti.append(out_sti_number[i][key][1][0])#두번째 sti의 이미지를 저장
                else:#그냥 의미 없음
                    sti.append(out_sti_number[i][key][0])
            for out in range(3):
                if(question[out] not in keys):#퀴즈에는 있으나 keys에는 포함되지 않는 것은 한번도 보여지지 않은 out-sti이다.
                    #out_sti_number[i][question[out]] = ['', 4]#한번도 보여지지 않은 out-sti를 out_sti_number에 key로 넣고, id 4로 저장
                    temp = {}
                    for j in range(3):#3개의 퀴즈에 대해서
                        for k in range(8):#모든 보기를 방문
                            if (q_block_answer[i][j][k] not in temp.keys()):#보기 중에서 temp의 key에 포함되지 않는 값이 있다면
                                temp[q_block_answer[i][j][k]] = 1
                            else:#이미 존재한다면 +1 해준다.
                                temp[q_block_answer[i][j][k]] += 1
                    print("temp:", temp)
                    sti_count = 0;
                    for l in temp.keys():
                        if((temp[l] == 3) and (l not in sti)):#보기가 3번 중복으로 나온 것 중에서, sti에 포함되지 않은 보기가 no display sti이다.
                            if(sti_count > 1):
                                print("error_____________________________")
                            sti_count += 1
                            out_sti_number[i][question[out]] = [l, 4]
                            #print(" word_number[i][question[out]]: ",  word_number[i])

    return out_sti_number

#보기 중 중복되는 img를 찾아 한 번도 display되지 않은 sti를 알아낸다.(sti_out 사용)
def find_no_display_sti(sti_out, trial_info_detail, q_block_answer):
    for i in range(len(q_block_answer)):
        keys = sti_out[i].keys()
        temp = {}
        if(trial_info_detail[i][0] == 7):
            for j in range(len(q_block_answer[i])):
                for k in range(8):
                    if (q_block_answer[i][j][k] not in temp.keys()):
                        temp[q_block_answer[i][j][k]] = 1
                    else:
                        temp[q_block_answer[i][j][k]] += 1

                    for l in temp.keys():
                        if(temp[l] == 3 and l not in keys):
                            print("l:", l)
                            sti_out[i][l] = 4
        print("--------------------------")
    return sti_out


#보기 중 중복되는 img를 찾아 한 번도 display되지 않은 sti를 알아낸다.(sti_out 사용)
def soa_find_no_display_sti(pre_out_list, SOA_BLOCK):
    soa_sti_out = []
    for soa in SOA_BLOCK:
        temp = {}
        for seq_num in range(5):
            temp[pre_out_list[soa][seq_num]['sti']] = pre_out_list[soa][seq_num]['seq_num']
        soa_sti_out.append(temp)

    return soa_sti_out

#보기 중 중복되는 img를 찾아 한 번도 display되지 않은 sti를 알아낸다.(out_sti_number 사용)
def soa_find_no_display_out(pre_out_list, SOA_BLOCK):
    soa_out_sti = []
    for soa in SOA_BLOCK:
        temp = {}
        for seq_num in range(5):
            if(pre_out_list[soa][seq_num]['out'] in temp.keys()):
                temp[pre_out_list[soa][seq_num]['out']] = [temp[pre_out_list[soa][seq_num]['out']], [pre_out_list[soa][seq_num]['sti'], pre_out_list[soa][seq_num]['seq_num']]]
            else:
                temp[pre_out_list[soa][seq_num]['out']] = [pre_out_list[soa][seq_num]['sti'], pre_out_list[soa][seq_num]['seq_num']]

        soa_out_sti.append(temp)
    return soa_out_sti



#1개의 질문의 보기 중, 정답인 보기의 인덱스를 저장하는 함수
#find_no_display_out을 이요하기 떄문에 diplay되지 않는 보기에 대해서도 answer_position을 찾을 수 있다.
#형식
#1 번째 블록: [7]
# 1 번째 블록: [4, 6]
# 1 번째 블록: [0, 3]
def answer_position(find_no_display_out, q_block_answer, q_block_question):
    ans_position = []
    for i in range(len(q_block_question)):
        ans_score = []
        for j in range(len(q_block_question[i])):
            ans_score_temp = []
            ans_temp = []
            answer = find_no_display_out[i][q_block_question[i][j]]
            if (len(answer[0]) == 2):
                ans_temp.append(answer[0][0])
                ans_temp.append(answer[1][0])
            else:
                ans_temp.append(answer[0])
            for k in range(8):
                if(q_block_answer[i][j][k] in ans_temp):
                    ans_score_temp.append(k)
            ans_score.append(ans_score_temp)
        ans_position.append(ans_score)
    return ans_position

#sti를 다 고른 것
def answer_position_5(find_no_display_sti, q_block_answer):
    ans_position = []
    non_ans_position = []

    for i in range(len(q_block_answer)):
        print("find_no_display_sti: ", find_no_display_sti[i])
        keys = find_no_display_sti[i].keys()
        ans_score = []
        non_ans_score = []
        for j in range(len(q_block_answer[i])):
            ans_temp = []
            non_ans_temp = []
            for k in range(8):
                if(q_block_answer[i][j][k] in keys):
                    #print("q_block_answer: ", q_block_answer[i][j][k])
                    ans_temp.append(k)
                else:
                    non_ans_temp.append(k)
            #print()
            ans_score.append(ans_temp)
            non_ans_score.append(non_ans_temp)

        ans_position.append(ans_score)
        non_ans_position.append(non_ans_score)
    return ans_position, non_ans_position




################## MAIN CODE ####################

def collectData(LEVEL):
    _doc = db.collection(LEVEL)
    docs = _doc.stream()

    if LEVEL == EASY_TASK:
        BLOCK, QUESTION, SCORE = EASY, EASY_Q, EASY_SCORE
        KEY_LEN, KEY = len(EASY_KEY), EASY_KEY
        Q_BLOCK, ID = EASY_Q_BLOCK, EASY_ID
        _BLOCK, HIST = EASY_BLOCK, EASY_HIST
    elif LEVEL == MEDIUM_TASK:
        BLOCK, QUESTION, SCORE = MEDIUM, MEDIUM_Q, MEDIUM_SCORE
        KEY_LEN, KEY = len(MEDIUM_KEY), MEDIUM_KEY
        Q_BLOCK, ID = EASY_Q_BLOCK, EASY_ID
    # elif LEVEL == HARD_TASK:
    #    BLOCK, QUESTION = HARD, HARD_Q
    #    KEY_LEN, KEY = len(HARD_KEY), HARD_KEY
    elif LEVEL == ONTOLOGY_TASK:
        BLOCK, QUESTION, SCORE = ONTO, ONTO_Q, ONTO_SCORE
        KEY_LEN, KEY = len(EASY_Q_BLOCK), ONTOLOGY_KEY


    n = 0
    for idx, doc in enumerate(docs):

        # time_user <- getUserData input
        time_user = doc.id  # millisecond & userID
        datas = doc.to_dict()  # get Data
        millisecond_len = 13
        t = time_user[:millisecond_len]
        u = time_user[millisecond_len:]
        if (u == 'Avei9QKPpZX5Y1uDY6RqFbIL5JI2' or u == 't63O8XUhvtcoz7tnv5BCQqzXft83'
                or u == 'ZgeqwkjwmwWUAuTfYwPki7JTuTr1' or u == 'Kv6qo8m3xhZvKMqNnioNO4U1CNT2' or u == 'wo0J3zKnGCf86mv9S5ol5BanCIs1' or u == 'N85nzSXJxYTRYWBKWy1mg7CuBa62' or u == 'cg3TwbwOQAbAb3CE6wSsPErdWqJ3' or u == 'uHtn4hABGXPtLjD2Uba08bLEIVq1'):
            continue

        if(u == 'uHtn4hABGXPtLjD2Uba08bLEIVq1' or u == 'cg3TwbwOQAbAb3CE6wSsPErdWqJ3'):
            continue

        # medium 한사람이 easy도 함
        if (u == '2iGoerY8AggFSOX2WNJCQMP3NdI2'):
             continue

        elif len(datas) >= 3 * BLOCK * QUESTION:
                print("number:", n)
                n += 1
                HIST_schedule = _HIST_schedule(datas['pre_out_list']['HIST_schedule'], EASY_HIST)

                trial_info = ([[d + 1] for d in datas['pre_out_list']['trial_info']])

                trial_info_detail = ([[d + 1] for d in datas['pre_out_list']['trial_info_detail']])
                ans_save_tot = _ans_save_tot(datas, BLOCK, QUESTION, SCORE)

                print('----------- ', idx, ' ------------')
                result = {}
                result['age'] = getUserAge(time_user)
                result['sex'] = getUserSex(time_user)
                result['HIST_schedule'] = HIST_schedule
                result['trial_info'] = trial_info
                result['trial_info_detail'] = trial_info_detail
                result['ans_save_tot'] = ans_save_tot


                result['easy_block'] = easy_block(datas['pre_out_list'], EASY_BLOCK)

                result['q_block_question'] =  q_block_question(datas['pre_out_list'], Q_BLOCK, ID)
                result['q_block_answer'] = q_block_answer(datas['pre_out_list'],Q_BLOCK, ID)
                result['total_time'] = total_time(datas, KEY)
                result['reaction_time'] = reaction_time(datas, KEY)

                result['out_sti_number'] = out_sti_number_matching(result['HIST_schedule'], result['easy_block'])
                result['sti_number'] = sti_number_matching(result['HIST_schedule'], result['easy_block'])


                if ('block1_soa' in datas['pre_out_list'].keys()):
                    result['find_no_display_out'] = soa_find_no_display_out(datas['pre_out_list'], SOA_BLOCK)
                    result['find_no_display_sti'] = soa_find_no_display_sti(datas['pre_out_list'], SOA_BLOCK)
                else:
                    result['find_no_display_out'] = find_no_display_out(result['out_sti_number'], result['trial_info_detail'], result['q_block_answer'],  result['q_block_question'])
                    result['find_no_display_sti'] = find_no_display_sti(result['sti_number'],
                                                                    result['trial_info_detail'],
                                                                    result['q_block_answer'])

                result['answer_position'] = answer_position(result['find_no_display_out'] ,result['q_block_answer'],  result['q_block_question'])
                result['answer_position_5'], result['non_answer_position_5']  = answer_position_5(result['find_no_display_sti'], result['q_block_answer'])


                file_name = 'easy_pickle/' + str(LEVEL) + '_user_' + str(n) + '.pickle'
                with open(file_name, 'wb') as f:
                    pickle.dump(result, f)
                    print("pickle saved ", file_name)

        else:  # Test Logs
            print(len(datas), " name : ", time_user, "     ", KEY_LEN)

#uHtn4hABGXPtLjD2Uba08bLEIVq1
def readPickle(filename):
    with open(filename, 'rb') as fr:
        task_data = pickle.load(fr)

        pass


def getUserData(time_user):
    millisecond_len = 13
    t = time_user[:millisecond_len]
    u = time_user[millisecond_len:]

    _doc = db.collection(USERS)
    docs = _doc.stream()

    for doc in docs:
        # user_id <- sha
        user_id = doc.id

        if user_id == u:
            datas = doc.to_dict()
            for k in USER_KEY:
                # get User Data
                print(datas[k])
        else:
            continue


if __name__ == "__main__":
    lv = EASY_TASK  # EASY_TASK, MEDIUM_TASK, HARD_TASK, ONTOLOGY_TASK

    collectData(LEVEL=lv)

#################### MAIN CODE #####################


