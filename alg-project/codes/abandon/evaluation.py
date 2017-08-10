import os
import itertools
import random
import numpy as np
import scipy.special as special
import facenet


def generate_pairs(data_dir, same_num, diff_num):
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        path_list = []
        issame_list = []
        classes = os.listdir(data_dir)
        classes.sort()

        # generate one person samples
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(data_dir, class_name)
            if os.path.isdir(facedir):
                faces = os.listdir(facedir)
                combinations = list(itertools.combinations(faces, 2))
                slice = random.sample(combinations, same_num)
                for s in slice:
                    path_list += (os.path.join(facedir, s[0]), os.path.join(facedir, s[1]))
                    issame_list.append(True)

        # generate two people sampes
        class_combinations = list(itertools.combinations(classes, 2))
        for cc in class_combinations:
            face1_dir = os.path.join(data_dir, cc[0])
            face2_dir = os.path.join(data_dir, cc[1])
            faces_1 = []
            faces_2 = []
            if os.path.isdir(face1_dir):
                faces_1 = os.listdir(face1_dir)
                faces_1 = random.sample(faces_1, diff_num)
            if os.path.isdir(face2_dir):
                faces_2 = os.listdir(face2_dir)
                faces_2 = random.sample(faces_2, diff_num)

            for i in range(diff_num):
                path_list += (os.path.join(face1_dir, faces_1[i]), os.path.join(face2_dir, faces_2[i]))
                issame_list.append(False)
    else:
        raise Exception('evaluate data dir ' + data_dir + ' nonexistent.')
    return path_list, issame_list


def evaluate(embeddings, seed, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), seed, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
                                              np.asarray(actual_issame), 1e-3, seed, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far
