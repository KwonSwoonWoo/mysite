
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import sys

float_precision_decimal_round = 8

if len(sys.argv) != 3:
    print ('Usage: %s CSVDataFile Targets' % sys.argv[0])
    exit(0)

input_data = sys.argv[1]
output_file = '.'.join(input_data.split('.')[:-1]) + '.interaction.csv'
corr_file = '.'.join(input_data.split('.')[:-1]) + '.corr.csv'
print ('> Input CSV Data File : %s' % input_data)
print ('> Output CSV File : %s' % output_file)
targets_data = sys.argv[2].split('/')
print ('> Target Features : %s' % str(targets_data))

try: #DB가 없을 때, 잘못 입력 했을 때.
    data = pd.read_csv(input_data)
except:
    print("error: fail to load CSV File - %s" % input_data)
    exit(1)

feature_org = list(data.columns)
for item in targets_data:
    try:
        feature_org.remove(item)
    except:
        print('error: undefined target - %s' % item)
        exit(1)

# new feature id
nfid = len(feature_org)

for f_1 in feature_org:
    f_1_list = data.loc[:, [f_1]].values
    for f_2 in feature_org:
        if f_1 >= f_2:
            continue
        f_2_list = data.loc[:, [f_2]].values
        f_name = f_1 + '*' + f_2
        print("> Make 2'nd order interaction feature - %s" % f_name)

        data_count = len(f_1_list)
        if data_count != len(f_2_list):
            print('error: invalid data')
            exit(1)

        inter_2 = []
        for i in range(data_count):
            f = round(float(f_1_list[i]*f_2_list[i]), float_precision_decimal_round)
            inter_2.append(f)

        data.insert(nfid, f_name, inter_2)
        nfid += 1

        for f_3 in feature_org:
            if f_2 >= f_3:
                continue
            f_3_list = data.loc[:, [f_3]].values
            f_name = f_1 + '*' + f_2 + '*' + f_3
            print("> Make 3'rd order interaction feature - %s" % f_name)

            data_count = len(f_2_list)
            if data_count != len(f_3_list):
                print('error: invalid data')
                exit(1)

            inter_3 = []
            for i in range(data_count):
                f = round(float(f_1_list[i]*f_2_list[i]*f_3_list[i]), float_precision_decimal_round)
                inter_3.append(f)

            data.insert(nfid, f_name, inter_3)
            nfid += 1

data.to_csv(output_file, index=False)

with open(corr_file, 'w') as fp:
    feature_org = list(data.columns)
    for t in targets_data:
        for f in feature_org:
            if f in targets_data:
                continue
            one_target_data = data.loc[:,[t,f]].dropna()
            t_list = one_target_data.loc[:,[t]].values.flatten().tolist()
            f_list = one_target_data.loc[:,[f]].values.flatten().tolist()
            corr = np.corrcoef(t_list, f_list)[0][1]

            fp.write('%s,%s,%g\n' % (t, f, corr))


            