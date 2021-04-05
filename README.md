# C3D-UCF11-Tensorflow
Very rudimentary Tensorflow implementation of C3D on UCF11 video dataset

Original paper:  https://arxiv.org/abs/1412.0767  "Learning Spatiotemporal Features with 3D Convolutional Networks"

This is the final project for RPI's ECSE DL course. Hail Qiang Ji!!!!

The following info should be enough for you to have a taste of an oversimplified C3D implementation. 
1. The code includes both training and testing/plotting. Feel free to modify it using tf.saver etc for fine-tuning and testing.
2. Due to memory issue I used fp16 instead of fp32 for data loading. If you have >16G memory feel free to use FP32. 
3. Converge time for batchsize=5 and ephoches=10 on RTX2080: <20 Mins.
4. Pickle file for data: https://drive.google.com/drive/u/1/folders/17Ul1bps7ONxQ3Ktt_QlJHnNWfl1tqHSq 
5. data shape: 10(Batch)x30(Frames per video sequence)x64x64x3(Image size)
6. The PDF summerizes the design choice, dataset splitting, parameter choice etc. I did not really fine tune hyper parameters. 
7. Major changes compared to the model proposed by the original paper: see PDF file
