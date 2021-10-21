3.4.1:
Real runtime 2.735s
This is a lot slower than in tenserflow. Tenserflow computed a single image in about 4-5ms
I think the difference in time is because tensorlfow does lots of optimization, some of those that we will be doing in this lab. 

3.4.2
compiled design with -std=c++11 to enable our timing
The second convolutional layer takes up most of the compute time
convolution took about 43% of time time
conv1 = .169s
conv2 = 1.683s
pooling1 = .002s
conv3 = .252s
conv4 = .453s
pooling2 = .001s
conv5 = .08s
conv6 = .103s
pooling3 = 0s
flat = 0s
dense1 = .014s
dense2 = .002s

2 threads:
conv1 = .361

4 threads:
conv1 = .197

