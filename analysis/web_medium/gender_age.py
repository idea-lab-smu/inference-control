import pickle
import numpy as np
col_rating = 'ans_save_tot'
col_round = 'trial_info'
col_trial_detail = 'trial_info_detail'
col_seq_schedule = 'HIST_schedule'



zero_buf = [[], [], [], [], [], [], [], [], []]

age_total = []
age_man = []
age_woman = []
sex_man = []
sex_woman = []
def readPickle(filename, index, age_total, age_man, age_woman, sex_man, sex_woman):
    sbj_file = filename % (index + 1)

    with open(sbj_file, 'rb') as fr:
            data = pickle.load(fr)
            print(sbj_file)
            sex = data['sex']
            age = data['age']

            if(sex == 'Gender.WOMEN'):
                sex_woman.append(1)
                age_woman.append(int(age))
            elif(sex == 'Gender.MAN'):
                sex_man.append(1)
                age_man.append(int(age))
            if(sex == 'Gender.MAN' or sex == 'Gender.WOMEN'):
                age_total.append(int(age))

