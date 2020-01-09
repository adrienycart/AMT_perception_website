import codecs
import os
import cPickle as pickle

import numpy as np
import tensorflow as tf

from config import MAX_ANSWERS


BATCH_SIZE = 50

def get_y_z_alphas(ratings):

    y = tf.cast(tf.equal(ratings,int(round(MAX_ANSWERS/2)), tf.int32))
    z = tf.cast(tf.less(ratings,int(round(MAX_ANSWERS/2))), tf.int32)
    # alpha is between 0 and 0.5
    alphas = tf.div(tf.abs(ratings-int(round(MAX_ANSWERS/2))),int(round(MAX_ANSWERS)))

    return y,z,alphas

def linear_regression_model(features):
    #features_o and features_t are of shape: [batch_size, n_features]

    assert features.get_shape().as_list()[1] == N_FEATURES

    weights = tf.Variable(tf.ones([N_FEATURES,1]))

    output = tf.sigmoid(tf.matmul(features, weights))

    return output


def contrastive_loss(batch1,batch2,y,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally similar

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.max(alphas-tf.abs(batch1-batch2),0))
    return loss

def contrastive_loss_magnitude(batch1,batch2,y,z,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally
    # z[i] = 1 iff batch1[i] was better rated than batch2[i]

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.max(alphas-tf.abs(batch1-batch2),0)) + \
           (1-y)*(z*tf.square(batch2) + (1-z)*tf.square(batch1))
    return loss

def contrastive_loss_absolute(batch1,batch2,y,z,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally
    # z[i] = 1 iff batch1[i] was better rated than batch2[i]

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.max(alphas-z*(batch1-batch2)-(1-z)*(batch2-batch1),0))

def import_features(example_dir,system):
    results = pickle.load(open(os.path.join(example_dir,system+'.pkl'), "rb"))
    all_feat = []
    for feat in features_to_use:
        value = results[feat]
        if type(value) is tuple:
            all_feat += list(value)
        elif type(value) is float:
            all_feat += [np.float64(value)]
        else:
            all_feat += [value]

    all_feat = [float(elt) for elt in all_feat]

    return all_feat


#### Prepare data:
feature_dir = 'precomputed_features'

filecp = codecs.open('db_csv/answers_data.csv', encoding = 'utf-8')
answers = np.genfromtxt(filecp,dtype=object,delimiter=";")
answers = answers[1:,:]

####       0            1         2          3        4         5        6            7           8        9           10
#### ['question_id' 'example' 'system1' 'system2' 'user_id' 'answer' 'recognised' 'difficulty' 'time'  'F_syst1' , 'F_syst2']

features1=[]
features2=[]
ratings=[]

features_to_use = [
                "framewise",
                "notewise_On_50",
                "notewise_OnOff_50",
                "high_f",
                "low_f",
                "high_n",
                "low_n",

                "loud_fn",
                "loud_ratio_fn",

                "out_key",
                "out_key_bin",

                "repeat",
                "merge",

                "semitone_f",
                "octave_f",
                "third_harmonic_f",
                "semitone_n",
                "octave_n",
                "third_harmonic_n",
                ]

for q_id in np.unique(answers[:,0]):
    data = answers[answers[:,0]==q_id]
    vote = np.sum(data[:,5].astype(int))
    example = data[0,1]
    system1 = data[0,2]
    system2 = data[0,3]

    # print example, system1, system2

    example_dir = os.path.join(feature_dir,example)
    # print len(import_features(example_dir,system1)),import_features(example_dir,system1)

    features1 += [import_features(example_dir,system1)]
    features2 += [import_features(example_dir,system2)]

features1 = np.array(features1,dtype=float)



features1_ph = tf.placeholder(tf.float32,[BATCH_SIZE,N_FEATURES])
features2_ph = tf.placeholder(tf.float32,[BATCH_SIZE,N_FEATURES])
ratings_ph = tf.placeholder(tf.int32,[BATCH_SIZE,1])

model_output1 = linear_regression_model(features1)
model_output2 = linear_regression_model(features2)

y,z,alphas = get_y_z_alphas(ratings)



print model_output
