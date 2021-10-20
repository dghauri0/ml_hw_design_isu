/*
 *
 * CPRE 482X
 * Lab 1 (Section 2) - C++ Implementation
 * Authors: Ryan Hunt, Jake Larimore, Dawood Ghauri
 *
 */


#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<math.h>
#include <errno.h>
#include "compare1d.h"
#include "compare3d_diff.h"
#include <chrono>
#include <pthread.h>

#define MAX_THREAD 2

using namespace std;

int part = 0;

// Function Declaration
vector<vector<vector<float> > > image_import(const char* fileName);
vector<vector<vector<float> > > ofmap_gen_conv(const vector<vector<vector<float> > > & input_fmap, const vector<vector<vector<vector<float> > > > & weights, const vector<float>& bias);
vector<float> ofmap_gen_dense(vector<float>& input_fmap, vector<vector<float> > & weights, vector<float> & bias, int output_size, bool last_layer);
vector<vector<vector<vector<float> > > > conv_weights(const char* filename, const int x, const int y, const int z, const int w);
vector<vector<float> > dense_weights(const char* filename, int x, int y);
vector<vector<vector<float> > > intermediate_compare_reshape(const char* filename, int x, int y, int z);
vector<float> flatten(vector<vector<vector<float> > > & in_layer);
vector<float> get_biases(const char* filename, int x);
vector<vector<vector<float> > > max_pooling_2D(vector<vector<vector<float> > > & ofmap_in);
vector<float> softmax(vector<float> & input);

// Thread-Specific Output Map Gen Function
void* ofmap_gen_conv_threaded(void *arg);

struct conv_layer {
	
	vector<vector<vector<float> > > *fmap;
	vector<vector<vector<vector<float> > > > *weights;
	vector<float>  *bias;
	vector<vector<vector<float> > > *output;
};

int main()

{  


	vector<vector<vector<float> > > conv1_image(64, vector<vector<float> >(64, vector<float>(3, 0)));
	vector<vector<vector<vector<float> > > > conv1_weights(5, vector<vector<vector<float> > >(5, vector<vector<float> >(3, vector<float>(32, 0))));
	vector<float> conv1_biases(32, 0);
	vector<vector<vector<float> > > conv1_out(60, vector<vector<float> >(60, vector<float>(32, 0)));
	vector<vector<vector<float> > > conv1_out_threaded(60, vector<vector<float> >(60, vector<float>(32, 0)));


	struct conv_layer *conv1_struct = (struct conv_layer *) malloc (sizeof (struct conv_layer));
	

	auto begin = std::chrono::high_resolution_clock::now(); // Start measuring time

	conv1_image = image_import("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/input.bin");
	conv1_weights = conv_weights("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv1_weights.bin", 5, 5, 3, 32);
	conv1_biases = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv1_biases.bin", 32);

	conv1_struct->fmap = &conv1_image;
 	conv1_struct->weights = &conv1_weights;
	conv1_struct->bias = &conv1_biases;
	conv1_struct->output = &conv1_out_threaded;

	// First Convolutional Layer Output
	conv1_out = ofmap_gen_conv(conv1_image, conv1_weights, conv1_biases);

	// Thread Handler Start
	pthread_t threads[MAX_THREAD];

	int t_i = 0;
	for (t_i = 0; t_i < MAX_THREAD; t_i++) {
		pthread_create(&threads[t_i], NULL, ofmap_gen_conv_threaded, (void *)conv1_struct);
	}

	for (t_i = 0; t_i < MAX_THREAD; t_i++) {
		pthread_join(threads[t_i], NULL);
	}
	// Thread Handler End 

	conv1_out_threaded = *conv1_struct->output;
	
	auto end = std::chrono::high_resolution_clock::now();	// End measuring time
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

	printf("conv1out: %f\n", conv1_out[0][0][0]);
	printf("conv1out_threaded: %f\n", conv1_out_threaded[0][0][0]);

	/*
	vector<vector<vector<float> > > test1_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_0_output.bin", 60, 60, 32);

	// Comparison
	int i = 0;
	int f = 0;
	int k = 0;
	float epsilon = 0.0001f;

	float max_diff = 0;
	float curr_diff = 0;

	for (i = 0; i < 60; ++i) {
		for (f = 0; f < 60; ++f) {
			for (k = 0; k < 32; ++k) {
				curr_diff = fabs(test1_inputs[i][f][k] - conv1_out[i][f][k]);
				if (curr_diff < epsilon) {
					// The values are equal
				}
				else {
					// The values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("conv1 diff: %f\n", max_diff);
	printf("conv1time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.

	vector<vector<vector<vector<float> > > > conv2_weights(5, vector<vector<vector<float> > >(5, vector<vector<float> >(32, vector<float>(32, 0))));
	vector<float> conv2_biases(32, 0);
	vector<vector<vector<float> > > conv2_out(56, vector<vector<float> >(56, vector<float>(32, 0)));

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time

	conv2_weights = conv_weights("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv2_weights.bin", 5, 5, 32, 32);
	conv2_biases = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv2_biases.bin", 32);

	// Second Convlolutional Layer Output 
	conv2_out = ofmap_gen_conv(conv1_out, conv2_weights, conv2_biases);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

	printf("conv2out: %f\n", conv2_out[0][0][0]);


	vector<vector<vector<float> > > test2_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_1_output.bin", 56, 56, 32);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 56; ++i) {
		for (f = 0; f < 56; ++f) {
			for (k = 0; k < 32; ++k) {
				curr_diff = fabs(test2_inputs[i][f][k] - conv2_out[i][f][k]);
				if (curr_diff < epsilon) {
					// The values are equal
				}
				else {
					// The values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("conv2 diff: %f\n", max_diff);
	printf("conv2time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.

	// First Pooling Layer Output  
	vector<vector<vector<float> > > pooling_out1(28, vector<vector<float> >(28, vector<float>(32, 0)));

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time
	
	pooling_out1 = max_pooling_2D(conv2_out);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

	vector<vector<vector<float> > > test3_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_2_output.bin", 28, 28, 32);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 28; ++i) {
		for (f = 0; f < 28; ++f) {
			for (k = 0; k < 32; ++k) {
				curr_diff = fabs(test3_inputs[i][f][k] - pooling_out1[i][f][k]);
				if (curr_diff < epsilon) {
					// The values are equal
				}
				else {
					// The values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("pooling1 diff: %f\n", max_diff);
	printf("pooling1time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.


	vector<vector<vector<vector<float> > > > conv3_weights(3, vector<vector<vector<float> > >(3, vector<vector<float> >(32, vector<float>(64, 0))));
	vector<float> conv3_biases(64, 0);
	vector<vector<vector<float> > > conv3_out(26, vector<vector<float> >(26, vector<float>(64, 0)));

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time

	conv3_weights = conv_weights("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv3_weights.bin", 3, 3, 32, 64);
	conv3_biases = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv3_biases.bin", 64);

	// Third Convolutional Layer Output 
	conv3_out = ofmap_gen_conv(pooling_out1, conv3_weights, conv3_biases);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

	vector<vector<vector<float> > > test4_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_3_output.bin", 26, 26, 64);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 26; ++i) {
		for (f = 0; f < 26; ++f) {
			for (k = 0; k < 64; ++k) {
				curr_diff = fabs(test4_inputs[i][f][k] - conv3_out[i][f][k]);
				if (curr_diff < epsilon) {
					// The values are equal
				}
				else {
					// The values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("conv3 diff: %f\n", max_diff);
	printf("conv3time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.

	vector<vector<vector<vector<float> > > > conv4_weights(3, vector<vector<vector<float> > >(3, vector<vector<float> >(64, vector<float>(64, 0))));
	vector<float> conv4_biases(64, 0);
	vector<vector<vector<float> > > conv4_out(24, vector<vector<float> >(24, vector<float>(64, 0)));

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time

	conv4_weights = conv_weights("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv4_weights.bin", 3, 3, 64, 64);
	conv4_biases = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv4_biases.bin", 64);

	// Fourth Convlolutional Layer Output 
	conv4_out = ofmap_gen_conv(conv3_out, conv4_weights, conv4_biases);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

	vector<vector<vector<float> > > test5_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_4_output.bin", 24, 24, 64);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 24; ++i) {
		for (f = 0; f < 24; ++f) {
			for (k = 0; k < 64; ++k) {
				curr_diff = fabs(test5_inputs[i][f][k] - conv4_out[i][f][k]);
				if (curr_diff < epsilon) {
					// The values are equal
				}
				else {
					// The values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("conv4 diff: %f\n", max_diff);
	printf("conv4time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.


	// Second Pooling Layer Output 
	vector<vector<vector<float> > > pooling_out2(12, vector<vector<float> >(12, vector<float>(64, 0)));

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time

	pooling_out2 = max_pooling_2D(conv4_out);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);	

	vector<vector<vector<float> > > test6_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_5_output.bin", 12, 12, 64);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 12; ++i) {
		for (f = 0; f < 12; ++f) {
			for (k = 0; k < 64; ++k) {
				curr_diff = fabs(test6_inputs[i][f][k] - pooling_out2[i][f][k]);
				if (curr_diff < epsilon) {
					//the values are equal
				}
				else {
					//the values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("pooling2 diff: %f\n", max_diff);
	printf("pooling2time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.


	vector<vector<vector<vector<float> > > > conv5_weights(3, vector<vector<vector<float> > >(3, vector<vector<float> >(64, vector<float>(64, 0))));
	vector<float> conv5_biases(64, 0);
	vector<vector<vector<float> > > conv5_out(10, vector<vector<float> >(10, vector<float>(64, 0)));

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time

	conv5_weights = conv_weights("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv5_weights.bin", 3, 3, 64, 64);
	conv5_biases = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv5_biases.bin", 64);

	// Fifth Convolutional Layer Output 
	conv5_out = ofmap_gen_conv(pooling_out2, conv5_weights, conv5_biases);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);		

	vector<vector<vector<float> > > test7_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_6_output.bin", 10, 10, 64);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 10; ++i) {
		for (f = 0; f < 10; ++f) {
			for (k = 0; k < 64; ++k) {
				curr_diff = fabs(test7_inputs[i][f][k] - conv5_out[i][f][k]);
				if (curr_diff < epsilon) {
					// The values are equal
				}
				else {
					// The values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("conv5 diff: %f\n", max_diff);
	printf("conv5time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.


	vector<vector<vector<vector<float> > > > conv6_weights(3, vector<vector<vector<float> > >(3, vector<vector<float> >(64, vector<float>(128, 0))));
	vector<float> conv6_biases(128, 0);
	vector<vector<vector<float> > > conv6_out(8, vector<vector<float> >(8, vector<float>(128, 0)));

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time

	conv6_weights = conv_weights("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv6_weights.bin", 3, 3, 64, 128);
	conv6_biases = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/conv6_biases.bin", 128);

	// Sixth Convlolutional Layer Output 
	conv6_out = ofmap_gen_conv(conv5_out, conv6_weights, conv6_biases);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

	vector<vector<vector<float> > > test8_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_7_output.bin", 8, 8, 128);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 8; ++i) {
		for (f = 0; f < 8; ++f) {
			for (k = 0; k < 128; ++k) {
				curr_diff = fabs(test8_inputs[i][f][k] - conv6_out[i][f][k]);
				if (curr_diff < epsilon) {
					// The values are equal
				}
				else {
					// The values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("conv6 diff: %f\n", max_diff);
	printf("conv6time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.


	// Third Pooling Layer Output  
	vector<vector<vector<float> > > pooling_out3(4, vector<vector<float> >(4, vector<float>(128, 0)));
	
	begin = std::chrono::high_resolution_clock::now(); // Start measuring time
	
	pooling_out3 = max_pooling_2D(conv6_out);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

	vector<vector<vector<float> > > test9_inputs = intermediate_compare_reshape("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_8_output.bin", 4, 4, 128);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 4; ++i) {
		for (f = 0; f < 4; ++f) {
			for (k = 0; k < 128; ++k) {
				curr_diff = fabs(test9_inputs[i][f][k] - pooling_out3[i][f][k]);
				if (curr_diff < epsilon) {
					// The values are equal
				}
				else {
					// The values are different
					printf("%d, %d, %d\n", i, f, k);
				}
				if (curr_diff > max_diff) {
					max_diff = curr_diff;
				}
			}
		}
	}
	printf("pooling3 diff: %f\n", max_diff);
	printf("pooling3time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time

	vector<float> flat = flatten(pooling_out3);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

	vector<float> test10_inputs = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_9_output.bin", 4 * 4 * 128);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 4 * 4 * 128; ++i) {
		curr_diff = fabs(test10_inputs[i] - flat[i]);
		if (curr_diff < epsilon) {
			// The values are equal
		}
		else {
			// The values are different
			printf("%d, %d, %d\n", i, f, k);
		}
		if (curr_diff > max_diff) {
			max_diff = curr_diff;
		}
	}
	printf("flat diff: %f\n", max_diff);
	printf("flattime: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.


	vector<vector<float> > dense1_weights(2048, vector<float>(256, 0));
	vector<float> dense1_biases(256, 0);
	vector<float> dense1_out(256, 0);

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time


	dense1_weights = dense_weights("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/dense1_weights.bin", 2048, 256);
	dense1_biases = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/dense1_biases.bin", 256);

	// First Dense Layer Output 
	dense1_out = ofmap_gen_dense(flat, dense1_weights, dense1_biases, 256, false);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);	

	vector<float> test11_inputs = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_10_output.bin", 256);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 256; ++i) {
		curr_diff = fabs(test11_inputs[i] - dense1_out[i]);
		if (curr_diff < epsilon) {
			// The values are equal
		}
		else {
			// The values are different
			printf("%d, %d, %d\n", i, f, k);
		}
		if (curr_diff > max_diff) {
			max_diff = curr_diff;
		}
	}
	printf("dense1 diff: %f\n", max_diff);
	printf("dense1time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.


	vector<vector<float> > dense2_weights(256, vector<float>(200, 0));
	vector<float> dense2_biases(200, 0);
	vector<float> dense2_out(200, 0);

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time


	dense2_weights = dense_weights("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/dense2_weights.bin", 256, 200);
	dense2_biases = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/dense2_biases.bin", 200);

	// Second Dense Layer Output
	dense2_out = ofmap_gen_dense(dense1_out, dense2_weights, dense2_biases, 200, true);

	end = std::chrono::high_resolution_clock::now();	// End measuring time
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);		

	vector<float> test12_inputs = get_biases("/local/jupyter/cpre482x-lab1/Inference/Template_Visual_Studio/Test_Input0/layer_11_output.bin", 200);

	// Comparison 
	max_diff = 0;
	for (i = 0; i < 200; ++i) {
		curr_diff = fabs(test12_inputs[i] - dense2_out[i]);
		if (curr_diff < epsilon) {
			// The values are equal
		}
		else {
			// The values are different
			printf("%d, %d, %d\n", i, f, k);
		}
		if (curr_diff > max_diff) {
			max_diff = curr_diff;
		}
	}
	printf("dense2 diff: %f\n", max_diff);
	printf("dense2time: %.3f seconds.\n", elapsed.count() * 1e-9);	// Report time.

	*/
	printf("done");
	return 0;

}

// pthread_create(&tid, NULL, hello, (void *)Allen);

void* ofmap_gen_conv_threaded(void *arg) {
	int thread_part = part++;
	struct conv_layer data = *((struct conv_layer*) arg);

	int i=0;
	int x=0;
	int y=0;
	int z=0;
	int a=0;
	int b=0;
	int filter_length = data.weights->size();
	int filter_height = (data.weights->at(0)).size();
	int filter_channel = ((data.weights->at(0)).at(0)).size();
	int filter_num = (((data.weights->at(0)).at(0)).at(0)).size();
	int ifmap_lenght = data.fmap->size();
	int ifmap_height = (data.fmap->at(0)).size();
	int ifmap_channel = ((data.fmap->at(0)).at(0)).size();
	float sum = 0;
	
	for(i=thread_part*(filter_num/MAX_THREAD); i< (thread_part+1)*(filter_num/MAX_THREAD); ++i) {
		for (x = 0; x <= ifmap_lenght - filter_length; x++) {									/* length of output */
			for (y = 0; y <= ifmap_height - filter_height; y++) {								/* height of output */
				for (z = 0; z < ifmap_channel; z++) {											/* input channel */
					for (a = 0; a < filter_length; a++) {										/* filter length */
						for (b = 0; b < filter_height; b++) {									/* filter height */
							sum += ((data.fmap->at(x+a)).at(y+b)).at(z) * (((data.weights->at(a)).at(b)).at(z)).at(i);	/* MultSum Accumulation */
						}
					}
				}
				((data.output->at(x)).at(y)).at(i) = sum + data.bias->at(i);
				sum = 0;

				/* ReLU */
				if (((data.output->at(x)).at(y)).at(i) < 0) {
					((data.output->at(x)).at(y)).at(i) = 0;
				}
			}
		}
	}
}

vector<vector<vector<float> > > image_import(const char* fileName) {

	/* Input Data */
	float conv1_inputs[12288]; // reshape back to x*y*z
	vector<vector<vector<float> > > reshaped_inputs(64, vector<vector<float> >(64, vector<float>(3, 0)));

	FILE* ptr_input = fopen(fileName, "rb");  // r for read, b for binary
	int r2 = fread(conv1_inputs, sizeof(float), 12288, ptr_input);
	printf("Read images: %d\n", r2);
	fclose(ptr_input);

	int i = 0;
	int f = 0;
	int j = 0;
	int count = 0;

	for (i = 0; i < 64; i++) {
		for (f = 0; f < 64; f++) {
			for (j = 0; j < 3; j++) {
				reshaped_inputs[i][f][j] = conv1_inputs[count];
				count++;
			}
		}
	}
	return reshaped_inputs;
}

/*
Generate
 */
vector<vector<vector<float> > > ofmap_gen_conv(const vector<vector<vector<float> > >& input_fmap, const vector<vector<vector<vector<float> > > >& weights, const vector<float>& bias) {
	int x = 0;
	int y = 0;
	int z = 0;
	int filter = 0;
	int a = 0;
	int b = 0;
	int c = 0;
	int filter_length = (int)weights.size();
	int filter_height = (int)weights[0].size();
	int filter_channel = (int)weights[0][0].size();
	int filter_num = (int)weights[0][0][0].size();
	int ifmap_lenght = (int)input_fmap.size();
	int ifmap_height = (int)input_fmap[0].size();
	int ifmap_channel = (int)input_fmap[0][0].size();
	float sum = 0;

	//printf("Output map height: %d\n", ifmap_height - filter_height);

	vector<vector<vector<float> > > output((ifmap_lenght - filter_length) + 1, vector<vector<float> >((ifmap_height - filter_height) + 1, vector<float>(filter_num, 0)));
	//vector<vector<vector<float> > > fmap_3d_section(filter_length, vector<vector<float> >(filter_height, vector<float>(filter_channel, 0)));

	for (filter = 0; filter < filter_num; filter++) {											/* # of filters and # of output channels */
		for (x = 0; x <= ifmap_lenght - filter_length; x++) {									/* length of output */
			for (y = 0; y <= ifmap_height - filter_height; y++) {								/* height of output */
				for (z = 0; z < ifmap_channel; z++) {											/* input channel */
					for (a = 0; a < filter_length; a++) {										/* filter length */
						for (b = 0; b < filter_height; b++) {									/* filter height */
							sum += input_fmap[x + a][y + b][z] * weights[a][b][z][filter];		/* MultSum Accumulation */
						}
					}
				}
				output[x][y][filter] = sum + bias[filter];
				sum = 0;

				/* ReLU */
				if (output[x][y][filter] < 0) {
					output[x][y][filter] = 0;
				}
			}
		}
	}
	return output;
}


vector<float> ofmap_gen_dense(vector<float>& input_fmap, vector<vector<float> >& weights, vector<float>& bias, int output_size, bool last_layer) {
	int x = 0;
	int y = 0;
	int z = 0;

	vector<float> output(output_size, 0);

	for (x = 0; x < output_size; x++) {
		for (y = 0; y < weights.size(); y++) {
			output[x] += input_fmap[y] * weights[y][x];
		}
		output[x] += bias[x];
		if (!last_layer) {
			if (output[x] < 0) {
				output[x] = 0;
			}
		}
	}

	if (last_layer) {
		output = softmax(output);
	}
	return output;

}

/*
Import weights from binary file (1D) and shape into 4D vector.
 */
vector<vector<vector<vector<float> > > > conv_weights(const char* filename, const int x, const int y, const int z, const int w) {

	/* Weights Data */
	int temp = x * y * z * w;
	float conv1_weights[temp]; // reshape back to x*y*z*w
	vector<vector<vector<vector<float> > > > reshaped_weights(x, vector<vector<vector<float> > >(y, vector<vector<float> >(z, vector<float>(w, 0))));
	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(conv1_weights, sizeof(float), x * y * z * w, ptr_weights);
	printf("Read weights: %d\n", r2);
	fclose(ptr_weights);

	int i = 0;
	int f = 0;
	int j = 0;
	int k = 0;
	int count = 0;

	for (i = 0; i < x; i++) {
		for (f = 0; f < y; f++) {
			for (j = 0; j < z; j++) {
				for (k = 0; k < w; k++) {
					reshaped_weights[i][f][j][k] = conv1_weights[count];
					count++;
				}
			}
		}
	}

	return reshaped_weights;
}

/*
Import weights from binary file (1D) and shape into 2D vector for the dense layer
 */
vector<vector<float> > dense_weights(const char* filename, int x, int y) {

	/* Weights Data */
	float dense_weight[x*y];
	vector<vector<float> > reshaped_weights(x, vector<float>(y, 0));
	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(dense_weight, sizeof(float), x * y, ptr_weights);
	printf("Read dense weights: %d\n", r2);
	fclose(ptr_weights);
	int i = 0;
	int f = 0;
	int count = 0;

	for (i = 0; i < x; i++) {
		for (f = 0; f < y; f++) {
			reshaped_weights[i][f] = dense_weight[count];
			count++;
		}
	}
	return reshaped_weights;
}

/*
 Purpose of this function is to take the intermediate layer outputs that are given to us to compare.
 */
vector<vector<vector<float> > > intermediate_compare_reshape(const char* filename, int x, int y, int z) {

	/* Weights Data */
	float intermediate[x*y*z];
	vector<vector<vector<float> > > reshaped_intermediate(x, vector<vector<float> >(y, vector<float>(z, 0)));
	FILE* ptr_intermediate = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(intermediate, sizeof(float), x * y * z, ptr_intermediate);
	printf("Read intermediate images: %d\n", r2);
	fclose(ptr_intermediate);

	int i = 0;
	int f = 0;
	int j = 0;
	int count = 0;

	for (i = 0; i < x; i++) {
		for (f = 0; f < y; f++) {
			for (j = 0; j < z; j++) {
				reshaped_intermediate[i][f][j] = intermediate[count];
				count++;
			}
		}
	}

	return reshaped_intermediate;
}

vector<float> flatten(vector<vector<vector<float> > >& in_layer) {
	vector<float> out(in_layer.size() * in_layer[0].size() * in_layer[0][0].size(), 0);
	int x = 0;
	int y = 0;
	int z = 0;
	int count = 0;
	for (x = 0; x < (int)in_layer.size(); x++) {
		for (y = 0; y < (int)in_layer[0].size(); y++) {
			for (z = 0; z < (int)in_layer[0][0].size(); z++) {
				out[count] = in_layer[x][y][z];
				count++;
			}
		}
	}
	return out;
}


/*
 Import biases from binary file.
 */
vector<float> get_biases(const char* filename, int x) {

	/* Weights Data */
	float conv1_biases[x];
	vector<float>biases(x, 0);

	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(conv1_biases, sizeof(float), x, ptr_weights);
	printf("Read biases: %d\n", r2);
	fclose(ptr_weights);

	int i = 0;
	for (i = 0; i < x; ++i) {
		biases[i] = conv1_biases[i];
	}

	return biases;
}

/*
Performs 2D max pooling (i.e. on each output channel).
 */
vector<vector<vector<float> > > max_pooling_2D(vector<vector<vector<float> > >& ofmap_in) {
	int x = 0;		/* Length ofmap_in */
	int y = 0;		/* Height ofmap_in */
	int z = 0;		/* Channel ofmap_in */
	int x_sec = 0;
	int y_sec = 0;

	vector<vector<vector<float> > > output(ofmap_in.size() / 2, vector<vector<float> >(ofmap_in[0].size() / 2, vector<float>(ofmap_in[0][0].size(), 0)));

	for (z = 0; z < (int)ofmap_in[0][0].size(); z++) {
		for (x = 0; x < (int)ofmap_in.size(); x = x + 2) {
			for (y = 0; y < (int)ofmap_in[0].size(); y = y + 2) {
				float max = 0;
				for (x_sec = x; x_sec < x + 2; x_sec++) {
					for (y_sec = y; y_sec < y + 2; y_sec++) {

						if (ofmap_in[x_sec][y_sec][z] > max) {
							max = ofmap_in[x_sec][y_sec][z];

						}
					}
				}
				output[x / 2][y / 2][z] = max;
			}
		}
	}

	return output;
}


vector<float> softmax(vector<float>& input) {
	vector<float> output(input.size(), 0);
	int x = 0;
	int y = 0;
	float sum = 0;
	for (x = 0; x < (int)input.size(); x++) {
		sum = 0;
		for (y = 0; y < (int)input.size(); y++) {
			sum += (float)exp(input[y]);
		}
		output[x] = (float)exp(input[x]) / sum;
	}

	return output;
}
