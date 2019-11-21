import tensorflow as tf


N_FEATURES = 10
BATCH_SIZE = 50


def linear_regression_model(features):
    #features_o and features_t are of shape: [batch_size, n_features]

    assert features.get_shape().as_list()[1] == N_FEATURES

    weights = tf.Variable(tf.ones([N_FEATURES,1]))

    output = tf.sigmoid(tf.matmul(features, weights))

    return output


def contrastive_loss():



features = tf.placeholder(tf.float32,[BATCH_SIZE,N_FEATURES])

model_output = linear_regression_model(features)

print model_output
