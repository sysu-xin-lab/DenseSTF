# DenseSTF

DenseSTF impelemented by TensorFlow 1.14 

Attached file namely "data.mat" shows the high resolution landsat(hr1~hr3) and low resolution MODIS(lr1~lr3) images for cropland scene.

The pre-processing of cloudmask and gapfilling has been conducted in MATLAB. Surface reflectence for all images are ranging 0~0.6.

Orginal images in the "data.mat" have a size of 1000*1000. A subset of 500*500 was used for train and test, i.e, data[250:750,250:750,:]

Train :
python densestf_train_mband.py && python densestf_train_mband_lulc.py

Test :
python densestf_test_mband.py && python densestf_test_mband_lulc.py

Evaluation using MATLAB:
load('data.mat','hr2')
load('pred-densestf.mat')
Metrics = computeMetric(hr2[251:750,251:750,:],predData); % matlab index starts from 1


