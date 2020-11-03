import numpy as np
import torch

from utils import *
from matching import *
import math


datadir = "D:/CSM/affinity/test/in/"
facefeat = "D:/CSM/feat/cast_face_feat/"
t_feat = "D:/CSM/feat/tracklet_face_feat/"
file = datadir + "/across_test.json"
testfile = "D:/CSM/meta/test.json"

bank = {}

# run_in_movie(datadir, testfile, 'face', 1)

mid_list, meta_info = read_meta(testfile)

average_mAP = 0
search_count = 0
threshold = 105

for i, mid in enumerate(mid_list):
    # read data
    print(i)
    tnum = meta_info[mid]['num_tracklet']
    pids = meta_info[mid]['pids']
    gt_list, gt_dict = parse_label(meta_info, mid)
    result_dict = {}

    for g in gt_list:
        if g not in bank.keys():
            if g != "others":
                bank[g] = {}
                bank[g] = np.load(facefeat + g + "..npy")

    for index, id in enumerate(gt_list):
        t_feature = np.load(t_feat + mid + "/" + str(index).zfill(5) + ".npy")
        distance_dict = {}

        for bid in bank:

            if bid not in distance_dict.keys():
                distance_dict[bid] = {}

            dist = []
            for feat in t_feature:
                dist.append(np.linalg.norm(bank[bid] - feat))

            distance = min(dist)
            mindex = np.argmin(dist)

            # distance = np.linalg.norm(bank[bid] - t_feature[0])
            distance_dict[bid]['dist'] = distance
            distance_dict[bid]['index'] = mindex

        min_dist = min(d['dist'] for d in distance_dict.values())

        if min_dist <= threshold:
            dist_index = [k for k in distance_dict if distance_dict[k]['dist'] == min_dist]
            dist_index = dist_index[0]
            min_index = distance_dict[dist_index]['index']

            n_feature = (bank[dist_index] + t_feature[min_index]) / 2
            bank[dist_index] = n_feature

        else:
            dist_index = "others"

        if dist_index not in result_dict.keys():
            result_dict[dist_index] = {}
        if result_dict[dist_index] == {}:
            result_dict[dist_index] = {index}
        else:
            result_dict[dist_index].add(index)

    mAP = get_mAP(gt_dict, result_dict)
    average_mAP += mAP * len(pids)
    search_count += len(pids)

average_mAP = average_mAP / search_count
print('Average mAP: {:.4f}'.format(average_mAP))
