import math
import densestf

cuda_devices = "0"
work_dir = './'
scale = 0.6
rcstart = 250
rcend = 750
size_input = 64
mwWidth = 32
mwHeight = 32
size_label = 1
size_out_channels = 6

layerlist = [8, 12, 16, 6]
num_filters = 32

allow_growth = True
# parameters
learning_rate = 5e-4
train_batch_size = 128
train_num_epoch = 80000

test_batch_size = math.ceil((rcend - rcstart) / size_label)
test_num_epoch = test_batch_size
max_to_keep = None
ckpt_dir = '{}checkpoint/'.format(work_dir)
filename = './data.mat'
model_name = 'densestf'


# net
def model(x):
    return densestf.densestf(x,
                             layerlist=layerlist,
                             filters=num_filters,
                             num_output_channels=size_out_channels)
