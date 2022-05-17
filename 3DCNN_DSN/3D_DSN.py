import tensorflow as tf
import time
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import os


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2_3D(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')


def get_inputOp(filename, batch_size, capacity):
    def read_and_decode(filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={"label": tf.FixedLenFeature([], tf.int64),
                                                                         "img_raw": tf.FixedLenFeature([],
                                                                                                       tf.string), })
        img = tf.decode_raw(features["img_raw"], tf.int16)
        img = tf.reshape(img, [16 * 16 * 16 * 1])
        img = tf.cast(img, tf.float32) * (1. / 255)
        label = tf.cast(features["label"], tf.int32)
        return img, label

    im, l = read_and_decode(filename)
    l = tf.one_hot(indices=tf.cast(l, tf.int32), depth=2)
    data, label = tf.train.batch([im, l], batch_size=batch_size, capacity=capacity)
    return data, label


def Sensitivity_specificity(model_output, equal):
    positive_position = 1
    negative_position = 0
    staticity_T = [0, 0]
    staticity_F = [0, 0]

    for i in range(len(equal)):
        if equal[i] == True:
            staticity_T[model_output[i]] += 1
        else:
            staticity_F[model_output[i]] += 1

    sensitivity = staticity_T[positive_position] / (
            staticity_T[positive_position] + staticity_F[(positive_position + 1) % 2])
    specificity = staticity_T[negative_position] / (
            staticity_T[negative_position] + staticity_F[(negative_position + 1) % 2])
    return sensitivity, specificity


def precision(model_output, equal):
    positive_position = 1
    negative_position = 0
    staticity_T = [0, 0]
    staticity_F = [0, 0]

    for i in range(len(equal)):
        if equal[i] == True:
            staticity_T[model_output[i]] += 1
        else:
            staticity_F[model_output[i]] += 1
    precision = staticity_T[positive_position] / (
            staticity_T[positive_position] + staticity_F[(negative_position + 1) % 2])
    return precision



sess = tf.InteractiveSession()
i = '10'

global_step = tf.Variable(0)
keep_prob = tf.placeholder(tf.float32)

######## input tfrecord
DMQ_train_data, DMQ_train_label = get_inputOp(
    "/home/public/PycharmProjects/tf1.8/HCC_CEMR_3D/ROIs_3D_63/tfrecord/"
    + i + "/TrainDMQ.tfrecords", 64, 1000)
DMQ_test_data, DMQ_test_label = get_inputOp(
    "/home/public/PycharmProjects/tf1.8/HCC_CEMR_3D/ROIs_3D_63/tfrecord/"
    + i + "/TestDMQ.tfrecords", 40, 40)
MMQ_train_data, MMQ_train_label = get_inputOp(
    "/home/public/PycharmProjects/tf1.8/HCC_CEMR_3D/ROIs_3D_63/tfrecord/"
    + i + "/TrainMMQ.tfrecords", 64, 1000)
MMQ_test_data, MMQ_test_label = get_inputOp(
    "/home/public/PycharmProjects/tf1.8/HCC_CEMR_3D/ROIs_3D_63/tfrecord/"
    + i + "/TestMMQ.tfrecords", 40, 40)
PS_train_data, PS_train_label = get_inputOp(
    "/home/public/PycharmProjects/tf1.8/HCC_CEMR_3D/ROIs_3D_63/tfrecord/"
    + i + "/TrainPS.tfrecords", 64, 1000)
PS_test_data, PS_test_label = get_inputOp(
    "/home/public/PycharmProjects/tf1.8/HCC_CEMR_3D/ROIs_3D_63/tfrecord/"
    + i + "/TestPS.tfrecords", 40, 40)


a=tf.Variable(1.0)
###############################################   DMQ    ##############################
x1 = tf.placeholder(tf.float32, [None, 16 * 16 * 16 * 1])
label1 = tf.placeholder(tf.float32, [None, 2])

inputData_1 = tf.reshape(x1, [-1, 16, 16, 16, 1])

kernel_11 = weight_variable([3, 3, 3, 1, 32])
bias_11 = bias_variable([32])
conv_11 = conv3d(inputData_1, kernel_11)
conv_out_11 = tf.nn.relu(conv_11 + bias_11)
pooling_out_11 = max_pool_2x2_3D(conv_out_11)

kernel_12 = weight_variable([3, 3, 3, 32, 64])
bias_12 = bias_variable([64])
conv_12 = conv3d(pooling_out_11, kernel_12)
conv_out_12 = tf.nn.relu(conv_12 + bias_12)
pooling_out_12 = max_pool_2x2_3D(conv_out_12)

kernel_13 = weight_variable([3, 3, 3, 64, 64])
bias_13 = bias_variable([64])
conv_13 = conv3d(pooling_out_12, kernel_13)
conv_out_13 = tf.nn.relu(conv_13 + bias_13)
pooling_out_13 = max_pool_2x2_3D(conv_out_13)

pooling_out_13 = tf.reshape(pooling_out_13, [-1, 2 * 2 * 2 * 64])

w_fc_11 = weight_variable([2 * 2 * 2 * 64, 50])
b_fc_11 = bias_variable([50])
fc_out_11 = tf.nn.relu(tf.matmul(pooling_out_13, w_fc_11) + b_fc_11)

######################################   MMQ   ###################################

x2 = tf.placeholder(tf.float32, [None, 16 * 16 * 16 * 1])

inputData_2 = tf.reshape(x2, [-1, 16, 16, 16, 1])

kernel_21 = weight_variable([3, 3, 3, 1, 32])
bias_21 = bias_variable([32])
conv_21 = conv3d(inputData_2, kernel_21)
conv_out_21 = tf.nn.relu(conv_21 + bias_21)
pooling_out_21 = max_pool_2x2_3D(conv_out_21)

kernel_22 = weight_variable([3, 3, 3, 32, 64])
bias_22 = bias_variable([64])
conv_22 = conv3d(pooling_out_21, kernel_22)
conv_out_22 = tf.nn.relu(conv_22 + bias_22)
pooling_out_22 = max_pool_2x2_3D(conv_out_22)

kernel_23 = weight_variable([3, 3, 3, 64, 64])
bias_23 = bias_variable([64])
conv_23 = conv3d(pooling_out_22, kernel_23)
conv_out_23 = tf.nn.relu(conv_23 + bias_23)
pooling_out_23 = max_pool_2x2_3D(conv_out_23)

pooling_out_23 = tf.reshape(pooling_out_23, [-1, 2 * 2 * 2 * 64])

w_fc_21 = weight_variable([2 * 2 * 2 * 64, 50])
b_fc_21 = bias_variable([50])
fc_out_21 = tf.nn.relu(tf.matmul(pooling_out_23, w_fc_21) + b_fc_21)

######################################   PS   ###################################

x3 = tf.placeholder(tf.float32, [None, 16 * 16 * 16 * 1])

inputData_3 = tf.reshape(x3, [-1, 16, 16, 16, 1])

kernel_31 = weight_variable([3, 3, 3, 1, 32])
bias_31 = bias_variable([32])
conv_31 = conv3d(inputData_3, kernel_31)
conv_out_31 = tf.nn.relu(conv_31 + bias_31)
pooling_out_31 = max_pool_2x2_3D(conv_out_31)

kernel_32 = weight_variable([3, 3, 3, 32, 64])
bias_32 = bias_variable([64])
conv_32 = conv3d(pooling_out_31, kernel_32)
conv_out_32 = tf.nn.relu(conv_32 + bias_32)
pooling_out_32 = max_pool_2x2_3D(conv_out_32)

kernel_33 = weight_variable([3, 3, 3, 64, 64])
bias_33 = bias_variable([64])
conv_33 = conv3d(pooling_out_32, kernel_33)
conv_out_33 = tf.nn.relu(conv_33 + bias_33)
pooling_out_33 = max_pool_2x2_3D(conv_out_33)

pooling_out_33 = tf.reshape(pooling_out_33, [-1, 2 * 2 * 2 * 64])

w_fc_31 = weight_variable([2 * 2 * 2 * 64, 50])
b_fc_31 = bias_variable([50])
fc_out_31 = tf.nn.relu(tf.matmul(pooling_out_33, w_fc_31) + b_fc_31)
############### DSN ######################
w_fc_12 = weight_variable([50, 2])
b_fc_12 = bias_variable([2])
mid1 = tf.matmul(fc_out_11, w_fc_12) + b_fc_12

with tf.name_scope('loss L1'):
    L1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid1, labels=label1))
    tf.summary.scalar('L2', L1)

w_fc_22 = weight_variable([50, 2])
b_fc_22 = bias_variable([2])
mid2 = tf.matmul(fc_out_21, w_fc_22) + b_fc_22

with tf.name_scope('loss L2'):
    L2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid2, labels=label1))
    tf.summary.scalar('L2', L2)
    

w_fc_32 = weight_variable([50, 2])
b_fc_32 = bias_variable([2])
mid3 = tf.matmul(fc_out_31, w_fc_32) + b_fc_32

with tf.name_scope('loss L3'):
    L3= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid3, labels=label1))
    tf.summary.scalar('L3', L3)
    
######################################     Fusion      ####################################
feature_cat = tf.concat([fc_out_11, fc_out_21,fc_out_31], 1)

w_fc_f1 = weight_variable([150, 2])
b_fc_f1 = bias_variable([2])
mid = tf.matmul(feature_cat, w_fc_f1) + b_fc_f1
prediction = tf.nn.softmax(mid)

with tf.name_scope('loss'):
    loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid, labels=label1))

    total_loss = loss_cross_entropy + a*(L1+L2+L3)
    tf.summary.scalar('loss_cross_entropy', loss_cross_entropy)

learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=64, decay_rate=0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

with tf.name_scope('Accuracy'):
    output_position = tf.argmax(prediction, 1)
    label_position = tf.argmax(label1, 1)
    predict = tf.equal(output_position, label_position)
    Accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
    tf.summary.scalar('Accuracy', Accuracy)

sess.run(tf.global_variables_initializer())
merge = tf.summary.merge_all()
board_path = '/home/public/PycharmProjects/tf1.8/code/log/' + 'DSN'
if (not (os.path.exists(board_path))):
    os.mkdir(board_path)

test_board_path = board_path + '/' + 'test'
if (not (os.path.exists(test_board_path))):
    os.mkdir(test_board_path)
test_board_path = test_board_path + '/' + i
if (not (os.path.exists(test_board_path))):
    os.mkdir(test_board_path)

train_board_path = board_path + '/' + 'train'
if (not (os.path.exists(train_board_path))):
    os.mkdir(train_board_path)
train_board_path = train_board_path + '/' + i
if (not (os.path.exists(train_board_path))):
    os.mkdir(train_board_path)

test_writer = tf.summary.FileWriter(test_board_path + '/', tf.get_default_graph())
train_writer = tf.summary.FileWriter(train_board_path + '/', tf.get_default_graph())
# Very important
tf.train.start_queue_runners(sess=sess)

before = time.time()


for times in range(10000):

    global_step = times
    DMQ_test_data_t, DMQ_test_label_t = sess.run([DMQ_test_data, DMQ_test_label])
    MMQ_test_data_t, MMQ_test_label_t = sess.run([MMQ_test_data, MMQ_test_label])
    PS_test_data_t, PS_test_label_t = sess.run([PS_test_data, PS_test_label])
    ####################  obtain raw data  #####################
    DMQ_train_data_t, DMQ_train_label_t = sess.run([DMQ_train_data, DMQ_train_label])
    MMQ_train_data_t, MMQ_train_label_t = sess.run([MMQ_train_data, MMQ_train_label])
    PS_train_data_t, PS_train_label_t = sess.run([PS_train_data, PS_train_label])

    ####################  Train start #########################
    if times % 20 == 0:
        summary, acc, output_position_r, label_position_r, predict_r, p = sess.run(
            [merge, Accuracy, output_position, label_position, predict, prediction],
            feed_dict={x1: DMQ_test_data_t, x2: MMQ_test_data_t, x3: PS_test_data_t,label1: DMQ_test_label_t, keep_prob: 1.0})

        test_writer.add_summary(summary, times)
        sen, spe = Sensitivity_specificity(output_position_r, predict_r)
        fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        print('step : ' + str(times))
        print('Accuracy is: ' + str(acc) + "\n" + 'Sensitivity is: ' + str(
            sen) + "\n" + 'Specificity is: ' + str(spe))
        print('Auc : ' + str(roc_auc))
        print(label_position_r)
        print(output_position_r)
        print(p)
        print('\n')

    ##########################  show  #######################
    if times == 9999:
        pre = precision(output_position_r, predict_r)
        print('precision is ' + str(pre))
        fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
        AUC = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic_3D_CCA')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % AUC)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    if times % 99 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merge, train_step],
                              feed_dict={x1: DMQ_train_data_t, x2: MMQ_train_data_t, x3: PS_train_data_t,label1: DMQ_train_label_t,
                                         keep_prob: 0.5}, options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % times)
        train_writer.add_summary(summary, times)
    else:
        sess.run([train_step],
                 feed_dict={x1: DMQ_train_data_t, x2: MMQ_train_data_t, x3: PS_train_data_t,label1: DMQ_train_label_t, keep_prob: 0.5})

after = time.time()
print('Total time is: ' + str((after - before) / 60) + ' minutes.')
train_writer.close()
test_writer.close()
