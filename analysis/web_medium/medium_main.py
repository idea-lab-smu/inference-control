import json
import pprint

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from medium.key import EASY_KEY, MEDIUM_KEY, ONTOLOGY_KEY, USER_KEY

EASY_TASK = 'easy-task'
MEDIUM_TASK = 'medium-task'
HARD_TASK = 'hard-task'
ONTOLOGY_TASK = 'ontology-task'
USERS = 'users'

EASY = 20 # Number of Question Set
EASY_Q = 3 # Number of Questions - each set

MEDIUM = 20
MEDIUM_Q = 3

HARD = 20
HARD_Q = 4

ONTO = 12
ONTO_Q = 4

#JSON_PATH =  'flutterstudy-4e686-6caf7d2e98d6.json'
JSON_PATH = 'idea-lab-35902-firebase-adminsdk-8tx4h-8c702e856c.json'
cred = credentials.Certificate(JSON_PATH)

firebase_admin.initialize_app(cred)
db = firestore.client()


################### MAIN CODE ####################

def collectData(LEVEL, KEY):
    _doc = db.collection(LEVEL)
    docs = _doc.stream()

    if LEVEL == EASY_TASK:
        BLOCK, QUESTION = EASY, EASY_Q
    elif LEVEL == MEDIUM_TASK:
        BLOCK, QUESTION = MEDIUM, MEDIUM_Q
    elif LEVEL == HARD_TASK:
        BLOCK, QUESTION = HARD, HARD_Q
    elif LEVEL == ONTOLOGY_TASK:
        BLOCK, QUESTION = ONTO, ONTO_Q
    ##########
    i = 0
    user = 1
    ##########
    for doc in docs:

        time_user = doc.id  # millisecond & userID
        datas = doc.to_dict()  # get Data
        millisecond_len = 13
        t = time_user[:millisecond_len]
        u = time_user[millisecond_len:]

        if ( u == 'Kv6qo8m3xhZvKMqNnioNO4U1CNT2'):
            continue

        if len(datas) >= 3 * BLOCK * QUESTION + 1:  # 3 (Question, Reaction Time, Input Score)
        #if(u == '2iGoerY8AggFSOX2WNJCQMP3NdI2'):
        #ePEUcAjs2seTiYLDEeSzbZ2F0WQ2
        #8vRtyMx85TWADxAxOtHC5IBQp6G2
            millisecond_len = 13
            t = time_user[:millisecond_len]
            u = time_user[millisecond_len:]

            #if (u == 'hcc0InrolNZEDjBmaHlJHypByk32'):
            #i += 1
                #print(i, "번째")
            print("number: ", user)
            getUserData(time_user)
            print("len datas: ", len(datas))
            user += 1
            print("--------------------------------------")
            #print(datas['pre_out_list'])
            # for k in KEY:
            #     if ("SCORE" in k):
            #         print(k, " - ", datas[k])
            # calc_time = 0
            # for k in KEY:
            #     if ("REACTION_TIME" in k):
            #         print(datas[k])
            #         start = datas[k][0]
            #         finish = datas[k][9]
            #         print("시간 차: ", (finish - start) / 1000, "초")
            #         calc_time += (finish - start) / 1000

            #for k in KEY:
            #key = sorted(datas.keys())
                # for k in KEY:
                #     if ("REACTION_TIME" in k):
                #         print(k, "- ", datas[k])
            #         #
            #         finish1 = datas[k][0]
            #         finish2 = datas[k][1]
            #         finish3 = datas[k][2]
            #         finish4 = datas[k][3]
            #         finish5 = datas[k][4]
            #         finish6 = datas[k][5]
            #         finish7 = datas[k][6]
            #
            #
            #         finish = max(finish1, finish2, finish3, finish4 ,finish5, finish6, finish7)
            #         start = datas[k][0]
            #         # start = datas[k][0]
            #         # finish = datas[k][9]
            #         print("시간 차: ", (finish - start) / 1000, "초")
            #key = sorted(datas.keys())
            # for k in key:
            #     if ("SCORE" in k):
            #         print(k, " - ", datas[k])
            # for k in key:
            #     if ("REACTION_TIME" in k):
            #             start = datas[k][0]
            #             finish = datas[k][9]
            #             print("시간 차: ", (finish - start) / 1000, "초")



        #for k in KEY:
        # key = sorted(datas.keys())
        # for k in key:
        #         print(k, ": ", datas[k])

    #print("--------------------------------------------------------")

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
                #if(k == 'id' or k == 'name'):
                    print(k, ": ", datas[k])
        else:
            continue

if __name__ == "__main__":
    lv = MEDIUM_TASK  # EASY_TASK, MEDIUM_TASK, HARD_TASK, ONTOLOGY_TASK
    k = MEDIUM_KEY[:]  # array (info - key.py)
    #k = ['BLOCK0_QUESTION0', 'BLOCK0_QUESTION0_REACTION_TIME', 'BLOCK0_QUESTION0_SCORE', 'BLOCK0_QUESTION1', 'BLOCK0_QUESTION1_REACTION_TIME', 'BLOCK0_QUESTION1_SCORE', 'BLOCK0_QUESTION2', 'BLOCK0_QUESTION2_REACTION_TIME', 'BLOCK0_QUESTION2_SCORE', 'BLOCK10_QUESTION0', 'BLOCK10_QUESTION0_REACTION_TIME', 'BLOCK10_QUESTION0_SCORE', 'BLOCK10_QUESTION1', 'BLOCK10_QUESTION1_REACTION_TIME', 'BLOCK10_QUESTION1_SCORE', 'BLOCK10_QUESTION2', 'BLOCK10_QUESTION2_REACTION_TIME', 'BLOCK10_QUESTION2_SCORE', 'BLOCK11_QUESTION0', 'BLOCK11_QUESTION0_REACTION_TIME', 'BLOCK11_QUESTION0_SCORE', 'BLOCK11_QUESTION1', 'BLOCK11_QUESTION1_REACTION_TIME', 'BLOCK11_QUESTION1_SCORE', 'BLOCK11_QUESTION2', 'BLOCK11_QUESTION2_REACTION_TIME', 'BLOCK11_QUESTION2_SCORE', 'BLOCK12_QUESTION0', 'BLOCK12_QUESTION0_REACTION_TIME', 'BLOCK12_QUESTION0_SCORE', 'BLOCK12_QUESTION1', 'BLOCK12_QUESTION1_REACTION_TIME', 'BLOCK12_QUESTION1_SCORE', 'BLOCK12_QUESTION2', 'BLOCK12_QUESTION2_REACTION_TIME', 'BLOCK12_QUESTION2_SCORE', 'BLOCK13_QUESTION0', 'BLOCK13_QUESTION0_REACTION_TIME', 'BLOCK13_QUESTION0_SCORE', 'BLOCK13_QUESTION1', 'BLOCK13_QUESTION1_REACTION_TIME', 'BLOCK13_QUESTION1_SCORE', 'BLOCK13_QUESTION2', 'BLOCK13_QUESTION2_REACTION_TIME', 'BLOCK13_QUESTION2_SCORE', 'BLOCK14_QUESTION0', 'BLOCK14_QUESTION0_REACTION_TIME', 'BLOCK14_QUESTION0_SCORE', 'BLOCK14_QUESTION1', 'BLOCK14_QUESTION1_REACTION_TIME', 'BLOCK14_QUESTION1_SCORE', 'BLOCK14_QUESTION2', 'BLOCK14_QUESTION2_REACTION_TIME', 'BLOCK14_QUESTION2_SCORE', 'BLOCK15_QUESTION0', 'BLOCK15_QUESTION0_REACTION_TIME', 'BLOCK15_QUESTION0_SCORE', 'BLOCK15_QUESTION1', 'BLOCK15_QUESTION1_REACTION_TIME', 'BLOCK15_QUESTION1_SCORE', 'BLOCK15_QUESTION2', 'BLOCK15_QUESTION2_REACTION_TIME', 'BLOCK15_QUESTION2_SCORE', 'BLOCK16_QUESTION0', 'BLOCK16_QUESTION0_REACTION_TIME', 'BLOCK16_QUESTION0_SCORE', 'BLOCK16_QUESTION1', 'BLOCK16_QUESTION1_REACTION_TIME', 'BLOCK16_QUESTION1_SCORE', 'BLOCK16_QUESTION2', 'BLOCK16_QUESTION2_REACTION_TIME', 'BLOCK16_QUESTION2_SCORE', 'BLOCK17_QUESTION0', 'BLOCK17_QUESTION0_REACTION_TIME', 'BLOCK17_QUESTION0_SCORE', 'BLOCK17_QUESTION1', 'BLOCK17_QUESTION1_REACTION_TIME', 'BLOCK17_QUESTION1_SCORE', 'BLOCK17_QUESTION2', 'BLOCK17_QUESTION2_REACTION_TIME', 'BLOCK17_QUESTION2_SCORE', 'BLOCK18_QUESTION0', 'BLOCK18_QUESTION0_REACTION_TIME', 'BLOCK18_QUESTION0_SCORE', 'BLOCK18_QUESTION1', 'BLOCK18_QUESTION1_REACTION_TIME', 'BLOCK18_QUESTION1_SCORE', 'BLOCK18_QUESTION2', 'BLOCK18_QUESTION2_REACTION_TIME', 'BLOCK18_QUESTION2_SCORE', 'BLOCK19_QUESTION0', 'BLOCK19_QUESTION0_REACTION_TIME', 'BLOCK19_QUESTION0_SCORE', 'BLOCK19_QUESTION1', 'BLOCK19_QUESTION1_REACTION_TIME', 'BLOCK19_QUESTION1_SCORE', 'BLOCK19_QUESTION2', 'BLOCK19_QUESTION2_REACTION_TIME', 'BLOCK19_QUESTION2_SCORE', 'BLOCK1_QUESTION0', 'BLOCK1_QUESTION0_REACTION_TIME', 'BLOCK1_QUESTION0_SCORE', 'BLOCK1_QUESTION1', 'BLOCK1_QUESTION1_REACTION_TIME', 'BLOCK1_QUESTION1_SCORE', 'BLOCK1_QUESTION2', 'BLOCK1_QUESTION2_REACTION_TIME', 'BLOCK1_QUESTION2_SCORE', 'BLOCK2_QUESTION0', 'BLOCK2_QUESTION0_REACTION_TIME', 'BLOCK2_QUESTION0_SCORE', 'BLOCK2_QUESTION1', 'BLOCK2_QUESTION1_REACTION_TIME', 'BLOCK2_QUESTION1_SCORE', 'BLOCK2_QUESTION2', 'BLOCK2_QUESTION2_REACTION_TIME', 'BLOCK2_QUESTION2_SCORE', 'BLOCK3_QUESTION0', 'BLOCK3_QUESTION0_REACTION_TIME', 'BLOCK3_QUESTION0_SCORE', 'BLOCK3_QUESTION1', 'BLOCK3_QUESTION1_REACTION_TIME', 'BLOCK3_QUESTION1_SCORE', 'BLOCK3_QUESTION2', 'BLOCK3_QUESTION2_REACTION_TIME', 'BLOCK3_QUESTION2_SCORE', 'BLOCK4_QUESTION0', 'BLOCK4_QUESTION0_REACTION_TIME', 'BLOCK4_QUESTION0_SCORE', 'BLOCK4_QUESTION1', 'BLOCK4_QUESTION1_REACTION_TIME', 'BLOCK4_QUESTION1_SCORE', 'BLOCK4_QUESTION2', 'BLOCK4_QUESTION2_REACTION_TIME', 'BLOCK4_QUESTION2_SCORE', 'BLOCK5_QUESTION0', 'BLOCK5_QUESTION0_REACTION_TIME', 'BLOCK5_QUESTION0_SCORE', 'BLOCK5_QUESTION1', 'BLOCK5_QUESTION1_REACTION_TIME', 'BLOCK5_QUESTION1_SCORE', 'BLOCK5_QUESTION2', 'BLOCK5_QUESTION2_REACTION_TIME', 'BLOCK5_QUESTION2_SCORE', 'BLOCK6_QUESTION0', 'BLOCK6_QUESTION0_REACTION_TIME', 'BLOCK6_QUESTION0_SCORE', 'BLOCK6_QUESTION1', 'BLOCK6_QUESTION1_REACTION_TIME', 'BLOCK6_QUESTION1_SCORE', 'BLOCK6_QUESTION2', 'BLOCK6_QUESTION2_REACTION_TIME', 'BLOCK6_QUESTION2_SCORE', 'BLOCK7_QUESTION0', 'BLOCK7_QUESTION0_REACTION_TIME', 'BLOCK7_QUESTION0_SCORE', 'BLOCK7_QUESTION1', 'BLOCK7_QUESTION1_REACTION_TIME', 'BLOCK7_QUESTION1_SCORE', 'BLOCK7_QUESTION2', 'BLOCK7_QUESTION2_REACTION_TIME', 'BLOCK7_QUESTION2_SCORE', 'BLOCK8_QUESTION0', 'BLOCK8_QUESTION0_REACTION_TIME', 'BLOCK8_QUESTION0_SCORE', 'BLOCK8_QUESTION1', 'BLOCK8_QUESTION1_REACTION_TIME', 'BLOCK8_QUESTION1_SCORE', 'BLOCK8_QUESTION2', 'BLOCK8_QUESTION2_REACTION_TIME', 'BLOCK8_QUESTION2_SCORE', 'BLOCK9_QUESTION0', 'BLOCK9_QUESTION0_REACTION_TIME', 'BLOCK9_QUESTION0_SCORE', 'BLOCK9_QUESTION1', 'BLOCK9_QUESTION1_REACTION_TIME', 'BLOCK9_QUESTION1_SCORE', 'BLOCK9_QUESTION2', 'BLOCK9_QUESTION2_REACTION_TIME', 'BLOCK9_QUESTION2_SCORE', 'pre_out_list']

    collectData(lv, k)





