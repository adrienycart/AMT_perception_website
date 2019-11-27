import tensorflow as tf
from config import MAX_ANSWERS

N_FEATURES = 10
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








features1 = tf.placeholder(tf.float32,[BATCH_SIZE,N_FEATURES])
features2 = tf.placeholder(tf.float32,[BATCH_SIZE,N_FEATURES])
ratings = tf.placeholder(tf.int32,[BATCH_SIZE,1])

model_output1 = linear_regression_model(features1)
model_output2 = linear_regression_model(features2)

y,z,alphas = get_y_z_alphas(ratings)



print model_output
