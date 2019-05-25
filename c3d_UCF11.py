import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys
from sklearn.utils import shuffle
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
BATCH_SIZE = 10
SEED = None
NUM_UNITS = 100
SEQ_LENGTH = 30
EPOCHES = 10
TOTAL_SIZE = 7200
TRAIN_SIZE = 6000
VALIDATION_SIZE = 200
TEST_SIZE = 1000
EVAL_FREQUENCY = 20
NUM_CLASS  = 11
RNN_WEIGHT_OUT = 30
# load the dataset into memory
np.set_printoptions(threshold=sys.maxsize)


x = tf.placeholder(shape=[BATCH_SIZE, 30, 64, 64, 3], dtype=tf.float32)
y = tf.placeholder(tf.int64, shape=(None))


conv1_weights = tf.Variable(tf.truncated_normal([3 ,3, 3, 3, 32],  stddev=0.01,seed=SEED, dtype=tf.float32))
conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 64], stddev=0.01,seed=SEED, dtype=tf.float32))
conv3a_weights = tf.Variable(tf.truncated_normal([5, 5, 5, 64, 128], stddev=0.01,seed=SEED, dtype=tf.float32))
conv3b_weights = tf.Variable(tf.truncated_normal([5, 5, 5, 128, 128], stddev=0.01,seed=SEED, dtype=tf.float32))

fc1_weights = tf.Variable(tf.truncated_normal([3456, 2048],stddev=0.01,seed=SEED,dtype=tf.float32))
fc2_weights = tf.Variable(tf.truncated_normal([2048, 1024],stddev=0.01,seed=SEED,dtype=tf.float32))
out_weights = tf.Variable(tf.truncated_normal([1024, NUM_CLASS],stddev=0.01,seed=SEED,dtype=tf.float32))


conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
conv2_biases = tf.Variable(tf.zeros([64], dtype=tf.float32))
conv3a_biases = tf.Variable(tf.zeros([128], dtype=tf.float32))
conv3b_biases = tf.Variable(tf.zeros([128], dtype=tf.float32))

fc1_biases = tf.Variable(tf.constant(0.0, shape=[2048], dtype=tf.float32)) 
fc2_biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32)) 
out_biases = tf.Variable(tf.constant(0.0, shape=[NUM_CLASS], dtype=tf.float32)) 


def CNN(data):
    #conv1 
    conv1 = tf.nn.conv3d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1, 1],
                        padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    print('conv1',conv1.shape)
    pool1 = tf.nn.max_pool3d(conv1,ksize=[1, 1, 2, 2, 1],strides=[1, 1, 2, 2, 1],padding='VALID')
    print('pool1',pool1.shape)

    #conv2
    conv2 = tf.nn.conv3d(pool1,conv2_weights,strides=[1, 1, 1, 1, 1],padding='VALID')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    print('conv2',conv2.shape)
    pool2 = tf.nn.max_pool3d(conv2,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],padding='VALID')
    print('pool2',pool2.shape)
    #conv3
    conv3 = tf.nn.conv3d(pool2,conv3a_weights,strides=[1, 1, 1, 1, 1],padding='VALID')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3a_biases))
    print('conv3a',conv3.shape)

    conv3 = tf.nn.conv3d(conv3,conv3b_weights,strides=[1, 1, 1, 1, 1],padding='VALID')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3b_biases))
    print('conv3b',conv3.shape)

    pool3 = tf.nn.max_pool3d(conv3,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],padding='VALID')
    print ('pool3',pool3.shape) 

    # pool3_reshape = tf.transpose(pool3, perm=[0,1,4,2,3])
    # print('pool3 reshaped',pool3_reshape.shape)
    # pool3_shape = pool3.get_shape().as_list()

    #reshape the last pooling layer to form fc1
    fc1=  tf.reshape(pool3, [BATCH_SIZE,3456])
    print('fc1 start shape',fc1.shape)
    fc1 = tf.matmul(fc1, fc1_weights) + fc1_biases
    fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, 0.5)
    print('fc1 end shape',fc1.shape)

    #fc nn
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    #fc2 = tf.nn.dropout(fc2,0.5)
    print('fc2 end shape',fc2.shape)
    print('out weigh shape',out_weights.shape)
    #output logits
    lgt = tf.matmul(fc2, out_weights) + out_biases
    print('logits shape',lgt.shape)
    return lgt

'''------------------output of the graph----------------------

logits : vector of value for each class before softmax 
predict op: softmax of z[m] aka logis
batch: Variable to keep track for learning rate decay 
'''
logits = CNN(x)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=logits))
predict_op = tf.nn.softmax(logits)
one_hot = tf.argmax(predict_op)


regularizers = (tf.nn.l2_loss(out_weights) + tf.nn.l2_loss(out_biases))
loss += 5e-4 * regularizers

batch = tf.Variable(0, dtype=tf.float32)

learning_rate = tf.train.exponential_decay(
    0.0001,                # Base learning rate.
    batch * BATCH_SIZE,  # Current index into the dataset.
    TRAIN_SIZE,          # Decay step.
    0.98,                # Decay rate.
    staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)

#-------------------------------------------------------------#
#-------------------------------------------------------------#
#------------------READ DATa----------------------------------#
#-------------------------------------------------------------#
data_file = open('youtube_action_train_data_part1.pkl', 'rb')
d1, l1 = pickle.load(data_file)
data_file.close()

data_file = open('youtube_action_train_data_part2.pkl', 'rb')
d2, l2 = pickle.load(data_file)
data_file.close()


l = np.concatenate((l1,l2),axis=0)
del l1, l2

d1=d1.astype(np.float16)
d2=d2.astype(np.float16)
d2 = d2/255 
d1 = d1/255

d = np.concatenate((d1,d2), axis = 0)

del d1,d2
train_data ,train_labels= d, l
train_data ,train_labels= shuffle(d, l, random_state=0)
del  d,l

#-------------------------------------------------------------#
#-------------------------------------------------------------#
#------------------Helper Funcions----------------------------#
#-------------------------------------------------------------#
#train_labels = shuffle(train_data, train_labels, random_state=0)

print('shapes for train and test')
print(train_data.shape)
print(train_labels.shape)


#return the error of one batch
def batch_error(GT,y_hat):
    batch_size = GT.shape[0]
    correct = 0
    #print(GT,y_hat)
    for i in range(0,batch_size):
        predict = np.argmax(y_hat[i])
        if predict == GT[i]:
            correct = correct +1 
    return correct/batch_size

#returns the validation error
def val_error_run(train_data,train_labels,sess):
    offset = 6000
    total_loss = 0
    accuracy = 0
    for step in range(0,int(VALIDATION_SIZE/BATCH_SIZE)):
        start = offset + step*BATCH_SIZE
        end = offset + (step+1)*(BATCH_SIZE)
        val_data = train_data[start:end, ...]
        val_labels = train_labels[start:end]
        val_dict = {x: val_data ,y: val_labels}
        one_loss, lr, predictions = sess.run(
            [loss, learning_rate, predict_op],
            feed_dict=val_dict)
        total_loss = total_loss+ one_loss
        accuracy = accuracy +  batch_error(val_labels,predictions) 
    total_loss = total_loss/ step
    accuracy = accuracy / step
    return (total_loss, lr, accuracy)

#returns the train error on this batch
def train_error_run(batch_data,batch_labels,sess):

    train_dict = {x: batch_data ,y: batch_labels}
    ls, predictions = sess.run(
        [loss, predict_op],
        feed_dict=train_dict)
    accuracy = batch_error(batch_labels,predictions) 

    return (ls, accuracy)

#test on the testdata and dump the lables and GT to pkl
def test_run(train_data,train_labels,sess):
    offset = 6200
    GT = train_labels[offset:offset+TEST_SIZE].tolist()
    y_predict = []
    for step in range(0,int(TEST_SIZE/BATCH_SIZE)):
        start = offset + step*BATCH_SIZE
        end = offset + (step+1)*(BATCH_SIZE)
        test_data = train_data[start:end, ...]
        test_labels = train_labels[start:end]
        test_dict = {x: test_data ,y: test_labels}
        #returns the argmax value 
        predictions = sess.run(predict_op,feed_dict = test_dict)

        for idx in range(0,BATCH_SIZE):
            one_predict = np.argmax(predictions[idx])
            y_predict.append(one_predict)
    cm = confusion_matrix(GT, y_predict)
    print(cm)
    filehandler = open("xj2.pkl","wb")
    pickle.dump(GT,filehandler)
    pickle.dump(y_predict, filehandler)
    filehandler.close()

'''
----------------------------------Prepare Saver ----------------------------
'''
tf.get_collection('validation_nodes')
# Add opts to the collection
tf.add_to_collection('validation_nodes', x)
tf.add_to_collection('validation_nodes', y)
tf.add_to_collection('validation_nodes', one_hot)
'''
----------------------------------Start Trainning ---------------------------
'''
saver = tf.train.Saver()
with tf.Session() as sess:

    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    print('steps to go',int(math.floor(EPOCHES*TRAIN_SIZE / BATCH_SIZE)))
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for step in range(0,int(math.floor(EPOCHES*TRAIN_SIZE / BATCH_SIZE))):
        offset = (step * BATCH_SIZE) % (TRAIN_SIZE - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        feed_dict = {x: batch_data,
                       y: batch_labels}
        _, tl ,lgt, y_pred = sess.run([optimizer,loss,logits,predict_op], feed_dict=feed_dict)


        if step%EVAL_FREQUENCY == 0:
            vl, lr ,v_acc= val_error_run(train_data,train_labels,sess)
            tl, t_acc = train_error_run(batch_data,batch_labels,sess)
            print('---------------epoch ', step*BATCH_SIZE/TRAIN_SIZE,'---------------')
            print('val loss',vl,'\nv_acc',v_acc)
            print('train loss',tl,'\ntrain accuracy',t_acc)
            epoch_list.append(step*BATCH_SIZE/TRAIN_SIZE)
            val_loss_list.append(vl)
            train_loss_list.append(tl)
            val_acc_list.append(v_acc)
            train_acc_list.append(t_acc)
            # train_error = train_error_run(batch_data,batch_labels,sess)
            # print('\nstep ', step,   'loss', l ,
            #     '\n\tlearning rate' , lr,
            #     '\n\tepoch ', step*BATCH_SIZE/TRAIN_SIZE,
            #     '\n\tval pixel error', val_error,
            #     '\n\ttrain pixel error', train_error )
            # val_error_list.append(val_error)
            # train_error_list.append(train_error)
            # loss_list.append(l)
            # epoch_list.append(step*BATCH_SIZE/TRAIN_SIZE)
            # step_list.append(step)
    test_run(train_data,train_labels,sess)
    save_path = saver.save(sess, "./xj_model/my_model")    
    plt.plot(epoch_list, val_loss_list ,label='validation loss')
    plt.plot(epoch_list, train_loss_list, label='training loss')
    plt.legend()
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.show()

    plt.plot(epoch_list, val_acc_list ,label='validation accuracy')
    plt.plot(epoch_list, train_acc_list, label='training accuracy')
    plt.legend()
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.show()

# # Save the model
# tf.get_collection('validation_nodes')
# # Add opts to the collection
# tf.add_to_collection('validation_nodes', x)
# tf.add_to_collection('validation_nodes', y)
# tf.add_to_collection('validation_nodes', predict_op)