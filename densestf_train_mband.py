import tensorflow as tf
import densestfConfig
import numpy as np
import os
import time
import scipy.io as io
import random

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
batch_size = densestfConfig.train_batch_size
num_epoch = densestfConfig.train_num_epoch
rcstart = densestfConfig.rcstart
rcend = densestfConfig.rcend
ckpt_dir = densestfConfig.ckpt_dir
model_name = densestfConfig.model_name
scale = densestfConfig.scale
max_to_keep = densestfConfig.max_to_keep
allow_growth = densestfConfig.allow_growth
model_dir = "%s_%s_mband" % (model_name, size_label)
# import_and_calibration
data = io.loadmat(densestfConfig.filename)
hr1 = data['hr1'] / scale
hr2 = data['hr2'] / scale
hr3 = data['hr3'] / scale
lr1 = data['lr1'] / scale
lr2 = data['lr2'] / scale
lr3 = data['lr3'] / scale

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

train1 = np.dstack([hr1, lr1, lr3])
train2 = np.dstack([hr3, lr3, lr1])
test1 = np.dstack([hr1, lr1, lr2])
test2 = np.dstack([hr3, lr3, lr2])
size_in_channels = train1.shape[-1]

# batch train
sz = (rcend - rcstart) * (rcend - rcstart)
perm = [i for i in range(sz * 2)]
batch_curr_pos = 0


def gen_train():
    global perm
    global batch_curr_pos

    result = np.zeros([batch_size, size_input, size_input, size_in_channels], dtype=np.float32)
    label = np.zeros([batch_size, size_label, size_label, size_out_channels], dtype=np.float32)

    for i in range(batch_size):
        if batch_curr_pos == 0:
            random.shuffle(perm)
        idx = perm[batch_curr_pos]
        if idx < sz:
            temp = train1
            templ = hr3
            r = idx // (rcend - rcstart) + rcstart
            c = idx % (rcend - rcstart) + rcstart
        else:
            temp = train2
            templ = hr1
            r = (idx - sz) // (rcend - rcstart) + rcstart
            c = (idx - sz) % (rcend - rcstart) + rcstart
        result[i, :, :, :] = temp[r - mwHeight:r + mwHeight, c - mwWidth:c + mwWidth, :]
        label[i, :, :, :] = templ[r:r + 1, c:c + 1, :]
        batch_curr_pos = (batch_curr_pos + 1) % (sz * 2)
    return result, label


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

# global_steps = tf.train.get_global_step()
# variables
train_images = tf.placeholder(tf.float32, [None, size_input, size_input, size_in_channels],
                              name='train_images')
train_labels = tf.placeholder(tf.float32, [None, size_label, size_label, size_out_channels],
                              name='train_labels')
global_steps = tf.Variable(0, name="global_step", trainable=False)

model = densestfConfig.model(train_images)
# stas_graph
graph = tf.get_default_graph()
flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
params = tf.profiler.profile(
    graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

l2_loss = tf.losses.mean_squared_error(labels=train_labels, predictions=model)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
#    l2_loss, global_step=global_steps)
# optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.piecewise_constant(
#    global_steps, [20000, 40000], [learning_rate, learning_rate * 0.5, learning_rate *
#                                   0.1])).minimize(l2_loss, global_step=global_steps)
optimizer = tf.train.AdamOptimizer(
    learning_rate=tf.train.exponential_decay(learning_rate, global_steps, 20000, 0.5)).minimize(
        l2_loss, global_step=global_steps)
tf.summary.scalar('loss', l2_loss)
merged = tf.summary.merge_all()
gpu_options = tf.GPUOptions(allow_growth=allow_growth)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    file_writer = tf.summary.FileWriter(os.path.join(ckpt_dir, model_dir), sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=max_to_keep)

    if load_ckpt(sess, ckpt_dir, saver):
        print('Successfully loaded checkpoint.')
    else:
        print('Failed to load checkpoint.')

    print('Training ...')
    start_time = time.time()
    # Training
    for epoch in range(num_epoch):
        epoch_images, epoch_labels = gen_train()
        _, epoch_loss, g_step = sess.run([optimizer, l2_loss, global_steps],
                                         feed_dict={
                                             train_images: epoch_images,
                                             train_labels: epoch_labels
                                         })

        # Save the checkpoint every 500 steps.
        if g_step % 500 == 0:
            summary = sess.run(merged,
                               feed_dict={
                                   train_images: epoch_images,
                                   train_labels: epoch_labels
                               })
            file_writer.add_summary(summary, g_step)
            print(' epoch： ', g_step, 'loss:', epoch_loss, ' duration:', time.time() - start_time)

        if g_step % 1000 == 0:
            save_ckpt(sess, g_step, saver)

        if g_step > num_epoch:
            print(' training completed! ', 'loss:', epoch_loss, 'duration:',
                  time.time() - start_time)
            break
