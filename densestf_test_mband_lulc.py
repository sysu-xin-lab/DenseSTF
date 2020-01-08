import tensorflow as tf
import densestfConfig
import numpy as np
import os
import time
import scipy.io as io

os.environ["CUDA_VISIBLE_DEVICES"] = densestfConfig.cuda_devices
scale = densestfConfig.scale
num_filters = densestfConfig.num_filters
num_epoch = densestfConfig.train_num_epoch
learning_rate = densestfConfig.learning_rate
size_input = densestfConfig.size_input
mwWidth = densestfConfig.mwWidth
mwHeight = densestfConfig.mwHeight
size_label = densestfConfig.size_label
size_out_channels = densestfConfig.size_out_channels
learning_rate = densestfConfig.learning_rate
batch_size = densestfConfig.test_batch_size
num_epoch = densestfConfig.test_num_epoch

rcstart = densestfConfig.rcstart
rcend = densestfConfig.rcend

ckpt_dir = densestfConfig.ckpt_dir
model_name = densestfConfig.model_name
scale = densestfConfig.scale
max_to_keep = densestfConfig.max_to_keep
model_dir = "%s_%s_mband_lulc" % (model_name, size_label)

# import_and_calibration
data = io.loadmat(densestfConfig.filename)
hr1 = data['hr1'] / scale
hr2 = data['hr2'] / scale
hr3 = data['hr3'] / scale
lr1 = data['lr1'] / scale
lr2 = data['lr2'] / scale
lr3 = data['lr3'] / scale
lulc = data['lulc'].astype(np.float32)
meanhr = np.zeros((1, 1, hr2.shape[2]), dtype=np.float32)
meanlr = np.zeros((1, 1, hr2.shape[2]), dtype=np.float32)
for i in range(hr2.shape[2]):
    meanhr[:, :, i] = np.mean(hr1[:, :, i] + hr3[:, :, i]) / 2.0
    meanlr[:, :, i] = np.mean(lr1[:, :, i] + lr2[:, :, i] + lr3[:, :, i]) / 3.0
hr1 -= meanhr
hr2 -= meanhr
hr3 -= meanhr
lr1 -= meanlr
lr2 -= meanlr
lr3 -= meanlr


test = np.dstack([hr1, lr1, lr2, lulc])
size_in_channels = test.shape[-1]


def gen_test():    
    row = epoch    
    result = np.zeros([batch_size, size_input, size_input, size_in_channels], dtype=np.float32)
    label = np.zeros([batch_size, size_label, size_label, size_out_channels], dtype=np.float32)
    r = rcstart + row
    for idx in range(0, batch_size):
        c = rcstart + idx
        result[idx, :, :, :] = test[r - mwHeight:r + mwHeight, c - mwWidth:c + mwWidth, :]
        label[idx, :, :, :] = hr2[r:r + 1, c:c + 1, :]
    return result, label


# Load the saved checkpoint. Reference: https://github.com/tegg89/SRCNN-Tensorflow/blob/master/model.py
def load_ckpt(sess, checkpoint_dir, saver):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    print('checkpoint_dir is', checkpoint_dir)

    # Require only one checkpoint in the directory.
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print('Restoring from', os.path.join(checkpoint_dir, ckpt_name))
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False


# Save the current checkpoint
def save_ckpt(sess, step, saver):
    checkpoint_dir = os.path.join(ckpt_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


# Initialization.
tf.reset_default_graph()
test_images = tf.placeholder(tf.float32, [None, size_input, size_input, size_in_channels],
                             name='test_images')
test_labels = tf.placeholder(tf.float32, [None, size_label, size_label, size_out_channels],
                             name='test_labels')
model = densestfConfig.model(test_images)

pred = np.zeros([size_label * batch_size, size_label * batch_size, size_out_channels])

l2_loss = tf.losses.mean_squared_error(labels=model, predictions=test_labels)

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    print('Test ...')
    start_time = time.time()
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    if load_ckpt(sess, ckpt_dir, saver):
        print('Successfully loaded checkpoint.')
    else:
        print('Failed to load checkpoint.')
    # test    
    for epoch in range(num_epoch):
        epoch_images, epoch_labels = gen_test()
        epoch_pred, epoch_loss = sess.run([model, l2_loss],
                                          feed_dict={
                                              test_images: epoch_images,
                                              test_labels: epoch_labels
                                          })
        r = epoch * size_label
        for idx in range(0, batch_size):
            c = idx * size_label
            pred[r:r + size_label, c:c + size_label, :] = epoch_pred[idx, :, :, :]        
        print('Epoch:', epoch + 1, 'loss:', epoch_loss, 'duration:', time.time() - start_time)    
    pred += meanhr
    pred *= scale
    io.savemat('pred-{}-lulc.mat'.format(model_name), {'pred': pred})
    print('test complete!')
