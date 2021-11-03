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

3.5.1
values of zero seconds were small enough to be approximated as such
due to insignificant time yield
2 threads:
conv1 = .365s
conv2 = 3.33s
pooling1 = .003s
conv3 = .556s
conv4 = .93s
pooling2 = .001s
conv5 = .25s
conv6 = .233s.
pooling3 = 0s
flat = 0s
dense1 = .016s
dense2 = 0s

4 threads:
conv1 = .2s
conv2 = 1.787s 
pooling1 = .002s
conv3 = .283s
conv4 = .504s
pooling2 = .001s
conv5 = .087s
conv6 = .111s
pooling3 = 0s
flat = 0s
dense1 = .015s
dense2 = .002s

8 threads:
conv1 = .198s
conv2 = 1.88s
pooling1 = .002s
conv3 = .289s
conv4 = .512s
pooling2 = .001s
conv5 = .013s
conv6 = .12s
pooling3 = 0s
flat = 0s
dense1 = .015s
dense2 = .002s

16 threads:
conv1 = .193s
conv2 = 1.731s
pooling1 = .003s
conv3 = .293s
conv4 = .482s
pooling2 = .001s
conv5 = .088s
conv6 = .119s
pooling3 = 0s
flat = 0s
dense1 = .017s 
dense2 = .002s

32 threads:
conv1 = .192s
conv2 = 1.756s
pooling1 = .002s
conv3 = .285s
conv4 = .506s
pooling2 = .001s
conv5 = .092s
conv6 = .118s
pooling3 = 0s
flat = 0s
dense1 = .016s
dense2 = .002s

3.6.2
100 iterations w/o tiling: 9m 10.869s
100 iterations w/tiling block size = 2: 4m 27.374s
100 iterations w/tiling block size = 4: 4m 19.695s
100 iterations w/tiling block size = 8: 4m 20.465s
100 iterations w/tiling block size = 16: 4m 31.558s
100 iterations w/tiling block size = 32: 4m 25.815s

3.7.1
Memory size needed to store TinyImageNet:
fp32 weights/biases: 3.08 MB
int8 weights/biases: 770.216 KB
int2 weights/biases: 192.554 KB

Total weights & biases: 770216
Total inputs: 361416
Total: 1131632

Given 1MB of memory, we can fit the entire TinyImageNet if we quantize all parameters to a max of 8 bits.
On top of storing weighs and biases we also need to store the first input activation and the last output activation. All others should already be in memory.

3.7.2
