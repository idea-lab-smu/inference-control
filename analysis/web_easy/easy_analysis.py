import numpy as np
import lib.os_data_utils_5 as odu
import pickle
import lib.score_efficiency as eff
import copy

filename = './web_easy/easy_pickle/easy-task_user_%d.pickle'

B = 0  # begin
N = 24
T = 20  # max trial
O = 3  # max outcome
S = 5  # number of stimulus

start_trial = 0
end_trial = 20

# total number of rounds per each participant
ROUND = 5

# whether to use Random or not
useRandom = True

# supporting sequence history
useSeqHistory = True

# pure uniform sampling
pure_random = [2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 22, 29, 30, 32, 35, 37, 38, 40, 41, 43, 44, 47, 50, 51, 53,  61, 62, 64, 65, 66, 67, 68, 71]

def filtered_by_percentile(conf_map, val_percentile):
    val = 0
    buf = []
    new_conf_map = []

    if val_percentile > 0:

        for cm in conf_map:
            buf.extend(cm[2])

        confidence_distribution = list(filter(lambda x: x != 0, buf))
        val = np.nanpercentile(confidence_distribution, val_percentile)

        for cm in conf_map:
            if max(cm[2]) >= val and max(cm[2]) > min(cm[2]):
                new_conf_map.append(cm)

    else:

        new_conf_map = conf_map[:]

    return new_conf_map


def run_even_list(nlist, fill_val=np.nan):
    lens = np.array([len(item) for item in nlist])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fill_val)
    out[mask] = np.concatenate(nlist)

    return out


def score_list_8(o, q_block_ans1, sti_num1, test_score_ans1):
    score_8 = [-1, -1, -1, -1, -1, -1, -1, -1]
    temp = []
    keys = sti_num1.keys()
    for i in range(8):
        if(q_block_ans1[o][i] in keys):
            score_8[sti_num1[q_block_ans1[o][i]]]= test_score_ans1[i]
        else:
            temp.append(test_score_ans1[i])

    for j in range(3):
        score_8[j+5] = str(temp[j])
    return score_8

#score_list_5(q_block_ans1,  out_sti_num1, cb)
def score_list_5(q_block_q1, out_sti_num1, score):
    score_5 = [-1, -1, -1, -1, -1]
    for i in range(len(q_block_q1)):
        sti = out_sti_num1[q_block_q1[i]]
        if(len(sti[0]) == 2):
            if(sti[0][1] == 0 or sti[0][1] == 2):
                if(sti[0][1] == 0):
                    if(sti[1][1] != 2):
                        print("this is error")
                elif (sti[0][1] == 2):
                    if (sti[1][1] != 0):
                        print("this is error")
                score_5[0] = score[i][0]
                score_5[2] = score[i][2]
            else:
                if (sti[0][1] == 1 or sti[0][1] == 3):
                    if (sti[0][1] == 1):
                        if (sti[1][1] != 3):
                            print("this is error")
                    elif (sti[0][1] == 3):
                        if (sti[1][1] != 1):
                            print("this is error")
                score_5[1] = score[i][1]
                score_5[3] = score[i][3]
        else:
            score_5[4] = score[i][4]
    return score_5

def make_non_display_zero(cm, visit_cnt):
    _cm = copy.deepcopy(cm)
    for i in range(len(_cm)):
        if(visit_cnt[i] == 0):
            _cm[i] = 0
    return _cm


def __validate__(score, visit_cnt):
    res = []
    for i in range(20):
        if((visit_cnt[i]) == 0):
            res.append(0)
        else:
            res.append(float(score[i]) / visit_cnt[i])

    return res
	
def result_cmaps_for_zscore(cls, outlier, efficiency, distance, rd_list=[], rd_indices=[]):
    seq = []
    seq_detail = []
    rating = []
    seqhist = []
    sti_num = []
    out_sti_num = []
    q_block_ans = []
    q_block_q = []
    ans_position = []
    non_ans_position = []
    test_score_ans = []
    #B = 0
    #N = 1
    # read data from the files
    for i in range(B, N):
        sbj_file = filename % (i + 1)
        #sbj_file = filename
        #print(sbj_file)
        with open(sbj_file, 'rb') as fr:
            data = pickle.load(fr)
            sequence = data['trial_info']
            sequence_detail = data['trial_info_detail']
            answer_score = data['ans_save_tot']
            seq_history = data['HIST_schedule']
            q_block_question = data['q_block_question']
            q_block_answer = data['q_block_answer']
            test_score_answer = copy.deepcopy(answer_score)
            sti_number = data['find_no_display_sti']
            out_sti_number = data['find_no_display_out']
            answer_position = data['answer_position_5']
            non_answer_position = data['non_answer_position_5']

            seq.append(sequence)
            seq_detail.append(sequence_detail)
            rating.append(answer_score)
            seqhist.append(seq_history)
            sti_num.append(sti_number)
            out_sti_num.append(out_sti_number)
            q_block_ans.append(q_block_answer)
            ans_position.append(answer_position)
            non_ans_position.append(non_answer_position)
            test_score_ans.append(test_score_answer)
            q_block_q.append(q_block_question)
    #
    # descriptive statistics
    #

    conf_map = []
    conf = [[], [], [], []]  # no, max, min, random
    score = [[], [], [], []]  # no, max, min, random
    for idx in range(N - B):

        idx_subj = idx + B + 1
        #print('subject id = %d' % idx_subj)

        if len(outlier) > 0 and outlier.count(idx_subj) > 0:
            continue

        if len(cls) > 0 and cls.count(idx_subj) <= 0:
            continue

        #1개 sbj의 data 가져옴
        seq_n1 = seq[idx]
        seq_detail_n1 = seq_detail[idx]
        rating_n1 = rating[idx]
        seqhist_n1 = seqhist[idx]
        ans_position_n1 = ans_position[idx]
        non_ans_position_n1 = non_ans_position[idx]
        out_sti_num_n1 = out_sti_num[idx]
        sti_num_n1 = sti_num[idx]
        q_block_ans_n1 = q_block_ans[idx]
        q_block_q_n1 = q_block_q[idx]
        test_score_ans_n1 = test_score_ans[idx]

        sub_buf = []

        #start_trial = 0
        #end_trial = 12
        for j in range(start_trial, end_trial):
            trial_buf = []
            # sequence:  [[1], [3], [2], [4], [3], [1], [2], [4], [4], [1], [2], [3]]
            seq1= seq_n1[j]  # 1, 2, 3, 4 (bayesian, maxos, minos, random)
            # sequence_id:  [[2], [10], [7], [15], [12], [4], [6], [13], [16], [1], [5], [9]]
            seq_detail1 = seq_detail_n1[j]  # 1-16 (sequence index in the sequence buffer)
            ans_position1 = ans_position_n1[j]
            non_ans_position1 = non_ans_position_n1[j]

            # answer: [[list([['8', '7', '10', '4', '6', '3', '5', '8']])
            #           list([['10', '2', '4', '10', '3', '10', '6', '7']])
            #           list([['1', '5', '6', '10', '5', '10', '10', '10']])
            #           list([['8', '10', '7', '7', '5', '2', '5', '10']])]
            rating1 = rating_n1[j]
            out_sti_num1 = out_sti_num_n1[j]
            sti_num1 = sti_num_n1[j]
            q_block_ans1 = q_block_ans_n1[j]
            test_score_ans1 = test_score_ans_n1[j]

            q_block_q1 = q_block_q_n1[j]
            #seq_history:  [[2, 0, 1, 3, 4, 2, 4, 0, 3, 1, 2, 0, 4, 3, 4, 3, 3, 2, 0, 1],
            if useSeqHistory == True:  
                seqhist1 = seqhist_n1[j]
                print (seqhist1)
            else:
                sh = np.array([])
            '''
            if seq1[0] == 4:
                try:
                    rd_idx = rd_list.index(seqhist1)
                    if pure_random.count(int(rd_idx)) == 0:
                        print (rd_idx)
                        continue
                except ValueError:
                    print ("List does not contain value")
                    continue
            '''
            # print visit
            trial_buf.append(seq1[0])
            trial_buf.append(seq_detail1[0])
            confidence_buf = []
            score_buf = []

            cb =[]
            sb = []
            #ra = []
            for o in range(O):
                temp8 = []

                # O = 4  # max outcome
                # S = 20  # number of stimulus
                #print(j, o)  # for debugging


                #8개의 보기에서 score를 가져오기
                for n in range(8):
                    r1 = rating1[o][0][n]
                    r2 = test_score_ans1[o][0][n]
                    # score 중 포함된 문자열 처리
                    if (r1.isdigit() != True):
                        if (r1[1:].isdigit() == True):
                            r1 = r1[1:]
                        elif (r1[0:-1].isdigit() == True):
                            r1 = r1[0:-1]
                        else:
                            r1 = '0'
                        rating1[o][0][n] = r1

                    if (r2.isdigit() != True):
                        # print("error r2:", r2)
                        if (r2[1:].isdigit() == True):
                            r2 = r2[1:]
                        elif (r2[0:-1].isdigit() == True):
                            r2 = r2[0:-1]
                        else:
                            r2 = '0'
                        test_score_ans1[o][0][n] = r2

                    if (int(r1) > 10):
                        r1 = '10'
                        rating1[o][0][n] = r1
                    if (int(r2) > 10):
                        r2 = '10'
                        test_score_ans1[o][0][n] = r2


                cb.append(score_list_8(o, q_block_ans1, sti_num1, rating1[o][0]))
                temp8.append(score_list_8(o, q_block_ans1, sti_num1, test_score_ans1[o][0]))
                sb.append(odu.get_normalised(temp8, o))

            confidence_buf.append(score_list_5(q_block_q1,  out_sti_num1, cb))
            score_buf.extend(score_list_5(q_block_q1, out_sti_num1, sb))
            trial_buf.append(confidence_buf[0])
            #node = eff.visits_on_node_5(seq_detail1[0] - 1, True)
            node = odu.visits_on_each_node(seq_detail1[0] - 1, True)
            score_buf = make_non_display_zero(score_buf, node)
            trial_buf.append(score_buf)
            sub_buf.append(trial_buf)

        conf_map.append(sub_buf)
    
    return  conf_map

def result_cmaps_for_tsne(cls, outlier, efficiency, distance, rd_list=[], rd_indices=[]):
    seq = []
    seq_detail = []
    rating = []
    seqhist = []
    sti_num = []
    out_sti_num = []
    q_block_ans = []
    q_block_q = []
    ans_position = []
    non_ans_position = []
    test_score_ans = []
    #B = 0
    #N = 1
    # read data from the files
    for i in range(B, N):
        sbj_file = filename % (i + 1)
        #sbj_file = filename
        #print(sbj_file)
        with open(sbj_file, 'rb') as fr:
            data = pickle.load(fr)
            sequence = data['trial_info']
            sequence_detail = data['trial_info_detail']
            answer_score = data['ans_save_tot']
            seq_history = data['HIST_schedule']
            q_block_question = data['q_block_question']
            q_block_answer = data['q_block_answer']
            test_score_answer = copy.deepcopy(answer_score)
            sti_number = data['find_no_display_sti']
            out_sti_number = data['find_no_display_out']
            answer_position = data['answer_position_5']
            non_answer_position = data['non_answer_position_5']

            seq.append(sequence)
            seq_detail.append(sequence_detail)
            rating.append(answer_score)
            seqhist.append(seq_history)
            sti_num.append(sti_number)
            out_sti_num.append(out_sti_number)
            q_block_ans.append(q_block_answer)
            ans_position.append(answer_position)
            non_ans_position.append(non_answer_position)
            test_score_ans.append(test_score_answer)
            q_block_q.append(q_block_question)
    #
    # descriptive statistics
    #

    conf_map = []
    conf = [[], [], [], []]  # no, max, min, random
    score = [[], [], [], []]  # no, max, min, random
    for idx in range(N - B):

        idx_subj = idx + B + 1
        #print('subject id = %d' % idx_subj)

        if len(outlier) > 0 and outlier.count(idx_subj) > 0:
            continue

        if len(cls) > 0 and cls.count(idx_subj) <= 0:
            continue

        #1개 sbj의 data 가져옴
        seq_n1 = seq[idx]
        seq_detail_n1 = seq_detail[idx]
        rating_n1 = rating[idx]
        seqhist_n1 = seqhist[idx]
        ans_position_n1 = ans_position[idx]
        non_ans_position_n1 = non_ans_position[idx]
        out_sti_num_n1 = out_sti_num[idx]
        sti_num_n1 = sti_num[idx]
        q_block_ans_n1 = q_block_ans[idx]
        q_block_q_n1 = q_block_q[idx]
        test_score_ans_n1 = test_score_ans[idx]

        sub_buf = []

        #start_trial = 0
        #end_trial = 12
        for j in range(start_trial, end_trial):
            trial_buf = []
            # sequence:  [[1], [3], [2], [4], [3], [1], [2], [4], [4], [1], [2], [3]]
            seq1= seq_n1[j]  # 1, 2, 3, 4 (bayesian, maxos, minos, random)
            # sequence_id:  [[2], [10], [7], [15], [12], [4], [6], [13], [16], [1], [5], [9]]
            seq_detail1 = seq_detail_n1[j]  # 1-16 (sequence index in the sequence buffer)
            ans_position1 = ans_position_n1[j]
            non_ans_position1 = non_ans_position_n1[j]

            # answer: [[list([['8', '7', '10', '4', '6', '3', '5', '8']])
            #           list([['10', '2', '4', '10', '3', '10', '6', '7']])
            #           list([['1', '5', '6', '10', '5', '10', '10', '10']])
            #           list([['8', '10', '7', '7', '5', '2', '5', '10']])]
            rating1 = rating_n1[j]
            out_sti_num1 = out_sti_num_n1[j]
            sti_num1 = sti_num_n1[j]
            q_block_ans1 = q_block_ans_n1[j]
            test_score_ans1 = test_score_ans_n1[j]

            q_block_q1 = q_block_q_n1[j]
            #seq_history:  [[2, 0, 1, 3, 4, 2, 4, 0, 3, 1, 2, 0, 4, 3, 4, 3, 3, 2, 0, 1],
            if useSeqHistory == True:  
                seqhist1 = seqhist_n1[j]
                print (seqhist1)
            else:
                sh = np.array([])
            '''
            if seq1[0] == 4:
                try:
                    rd_idx = rd_list.index(seqhist1)
                    if pure_random.count(int(rd_idx)) == 0:
                        print (rd_idx)
                        continue
                except ValueError:
                    print ("List does not contain value")
                    continue
            '''
            # print visit
            trial_buf.append(seq1[0])
            trial_buf.append(seq_detail1[0])
            confidence_buf = []
            score_buf = []

            cb =[]
            sb = []
            #ra = []
            for o in range(O):
                temp8 = []
				
                #if (o == 2):

                # O = 4  # max outcome
                # S = 20  # number of stimulus
                #print(j, o)  # for debugging


                #8개의 보기에서 score를 가져오기
                for n in range(8):
                    r1 = rating1[o][0][n]
                    # score 중 포함된 문자열 처리
                    if (r1.isdigit() != True):
                        if (r1[1:].isdigit() == True):
                            r1 = r1[1:]
                        elif (r1[0:-1].isdigit() == True):
                            r1 = r1[0:-1]
                        else:
                            r1 = '0'
                        rating1[o][0][n] = r1

                    if (int(r1) > 10):
                        r1 = '10'
                        rating1[o][0][n] = r1

                cb.append(score_list_8(o, q_block_ans1, sti_num1, rating1[o][0]))

            temp = [int(ch) for ch in cb[2][0:5]]
            trial_buf.append(temp)
            sub_buf.append(trial_buf)

        conf_map.append(sub_buf)
    
    return  conf_map


def result(cls, outlier, efficiency, distance, rd_list=[], rd_indices=[]):
    seq = []
    seq_detail = []
    rating = []
    seqhist = []
    sti_num = []
    out_sti_num = []
    q_block_ans = []
    q_block_q = []
    ans_position = []
    non_ans_position = []
    test_score_ans = []
    #B = 0
    #N = 1
    # read data from the files
    for i in range(B, N):
        sbj_file = filename % (i + 1)
        #sbj_file = filename
        #print(sbj_file)
        with open(sbj_file, 'rb') as fr:
            data = pickle.load(fr)
            sequence = data['trial_info']
            sequence_detail = data['trial_info_detail']
            answer_score = data['ans_save_tot']
            seq_history = data['HIST_schedule']
            q_block_question = data['q_block_question']
            q_block_answer = data['q_block_answer']
            test_score_answer = copy.deepcopy(answer_score)
            sti_number = data['find_no_display_sti']
            out_sti_number = data['find_no_display_out']
            answer_position = data['answer_position_5']
            non_answer_position = data['non_answer_position_5']

            seq.append(sequence)
            seq_detail.append(sequence_detail)
            rating.append(answer_score)
            seqhist.append(seq_history)
            sti_num.append(sti_number)
            out_sti_num.append(out_sti_number)
            q_block_ans.append(q_block_answer)
            ans_position.append(answer_position)
            non_ans_position.append(non_answer_position)
            test_score_ans.append(test_score_answer)
            q_block_q.append(q_block_question)
    #
    # descriptive statistics
    #

    conf_map = []
    conf = [[], [], [], []]  # no, max, min, random
    score = [[], [], [], []]  # no, max, min, random
    for idx in range(N - B):

        idx_subj = idx + B + 1
        #print('subject id = %d' % idx_subj)

        if len(outlier) > 0 and outlier.count(idx_subj) > 0:
            continue

        if len(cls) > 0 and cls.count(idx_subj) <= 0:
            continue

        #1개 sbj의 data 가져옴
        seq_n1 = seq[idx]
        seq_detail_n1 = seq_detail[idx]
        rating_n1 = rating[idx]
        seqhist_n1 = seqhist[idx]
        ans_position_n1 = ans_position[idx]
        non_ans_position_n1 = non_ans_position[idx]
        out_sti_num_n1 = out_sti_num[idx]
        sti_num_n1 = sti_num[idx]
        q_block_ans_n1 = q_block_ans[idx]
        q_block_q_n1 = q_block_q[idx]
        test_score_ans_n1 = test_score_ans[idx]

        sub_buf = []

        #start_trial = 0
        #end_trial = 12
        for j in range(start_trial, end_trial):
            trial_buf = []
            # sequence:  [[1], [3], [2], [4], [3], [1], [2], [4], [4], [1], [2], [3]]
            seq1= seq_n1[j]  # 1, 2, 3, 4 (bayesian, maxos, minos, random)
            # sequence_id:  [[2], [10], [7], [15], [12], [4], [6], [13], [16], [1], [5], [9]]
            seq_detail1 = seq_detail_n1[j]  # 1-16 (sequence index in the sequence buffer)
            ans_position1 = ans_position_n1[j]
            non_ans_position1 = non_ans_position_n1[j]

            # answer: [[list([['8', '7', '10', '4', '6', '3', '5', '8']])
            #           list([['10', '2', '4', '10', '3', '10', '6', '7']])
            #           list([['1', '5', '6', '10', '5', '10', '10', '10']])
            #           list([['8', '10', '7', '7', '5', '2', '5', '10']])]
            rating1 = rating_n1[j]
            out_sti_num1 = out_sti_num_n1[j]
            sti_num1 = sti_num_n1[j]
            q_block_ans1 = q_block_ans_n1[j]
            test_score_ans1 = test_score_ans_n1[j]

            q_block_q1 = q_block_q_n1[j]
            #seq_history:  [[2, 0, 1, 3, 4, 2, 4, 0, 3, 1, 2, 0, 4, 3, 4, 3, 3, 2, 0, 1],
            if useSeqHistory == True:  
                seqhist1 = seqhist_n1[j]
                print (seqhist1)
            else:
                sh = np.array([])
            '''
            if seq1[0] == 4:
                try:
                    rd_idx = rd_list.index(seqhist1)
                    if pure_random.count(int(rd_idx)) == 0:
                        print (rd_idx)
                        continue
                except ValueError:
                    print ("List does not contain value")
                    continue
            '''
            # print visit
            trial_buf.append(seq1[0])
            trial_buf.append(seq_detail1[0])
            confidence_buf = []
            score_buf = []

            cb =[]
            sb = []
            #ra = []
            for o in range(O):
                temp8 = []

                # O = 4  # max outcome
                # S = 20  # number of stimulus
                #print(j, o)  # for debugging


                #8개의 보기에서 score를 가져오기
                for n in range(8):
                    r1 = rating1[o][0][n]
                    r2 = test_score_ans1[o][0][n]
                    # score 중 포함된 문자열 처리
                    if (r1.isdigit() != True):
                        if (r1[1:].isdigit() == True):
                            r1 = r1[1:]
                        elif (r1[0:-1].isdigit() == True):
                            r1 = r1[0:-1]
                        else:
                            r1 = '0'
                        rating1[o][0][n] = r1

                    if (r2.isdigit() != True):
                        # print("error r2:", r2)
                        if (r2[1:].isdigit() == True):
                            r2 = r2[1:]
                        elif (r2[0:-1].isdigit() == True):
                            r2 = r2[0:-1]
                        else:
                            r2 = '0'
                        test_score_ans1[o][0][n] = r2

                    if (int(r1) > 10):
                        r1 = '10'
                        rating1[o][0][n] = r1
                    if (int(r2) > 10):
                        r2 = '10'
                        test_score_ans1[o][0][n] = r2


                cb.append(score_list_8(o, q_block_ans1, sti_num1, rating1[o][0]))
                temp8.append(score_list_8(o, q_block_ans1, sti_num1, test_score_ans1[o][0]))
                sb.append(odu.get_normalised(temp8, o))

            confidence_buf.append(score_list_5(q_block_q1,  out_sti_num1, cb))
            score_buf.extend(score_list_5(q_block_q1, out_sti_num1, sb))
            trial_buf.append(confidence_buf[0])
            #node = eff.visits_on_node_5(seq_detail1[0] - 1, True)
            node = odu.visits_on_each_node(seq_detail1[0] - 1, True)
            score_buf = make_non_display_zero(score_buf, node)
            trial_buf.append(score_buf)
            sub_buf.append(trial_buf)


        conf_map.extend(sub_buf)
    ic_conf, os_conf, ic_score, os_score = odu.distinct_ic_os_buf(conf_map, efficiency, distance)
    # for cm in conf_map:
    #     visit_cnt = eff.visits_on_node_5(seq_detail1, efficiency)
    #     if(efficiency == True):
    #         conf[cm[0][0] - 1].extend(odu.__validate__(cm[2], visit_cnt))
    #         score[cm[0][0] - 1].extend(odu.__validate__(cm[3], visit_cnt))
    #     else:
    #         conf[cm[0][0] - 1].extend(cm[2])
    #         score[cm[0][0] - 1].extend(cm[3])
    #     print("conf: ", conf)
    #     print("score: ", score)
    return  ic_conf, os_conf, ic_score, os_score



def get_webeasy_result(cls, outlier, efficiency, distance, rd_list=[], rd_indices=[]):
    print("plot_opt_vs_counteropt")

    ic_conf, os_conf, ic_score, os_score = result(cls, outlier, efficiency, distance, rd_list=rd_list, rd_indices=rd_indices)
    return ic_conf, os_conf, ic_score, os_score




