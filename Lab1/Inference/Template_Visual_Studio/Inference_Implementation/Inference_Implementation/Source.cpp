#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<math.h>

using namespace std;



//template<int D, int N, int M, int s, int f>

// Implement the layers as functions

vector<vector<vector<float>>> image_import(char* fileName);

int main()

{
	// Do this for all the three chosen inputs:
	

		// Load the input data from binary file.
			// First load the flattened array then reshape according to input dimensions.


		// Load the weights data from binary files.
			// First load the flattened array then reshape according to weights/bias dimensions.


		// Load the intermediate feature map data from binary files.
		 // First load the flattened array then reshape according to intermediate feature map dimensions.

		 
		/* Sample code to read a binary file (test.bin) exported from python into C: */

		// SAMPLE CODE: Read a flattened 5*5*3*32 (2400 float values) convolution weight binary file (test.bin) into an array in C
		vector<vector<vector<float>>> conv1_image(64, vector<vector<float>>(64, vector<float>(3,0)));
		vector<vector<vector<vector<float>>>> conv1_weights(5, vector<vector<vector<float>>>(5, vector<vector<float>>(3, vector<float>(32, 0))));
		vector<float> conv1_biases(32, 0);
		vector<vector<vector<float>>> conv1_out(60, vector<vector<float>>(60, vector<float>(32,0)));
		
		conv1_image = image_import("input.bin");
		conv1_weights = conv_weights("conv1_weights.bin", 5, 5, 3, 32);
		conv1_biases = get_biases("conv1_biases.bin", 32);
		//First Convolutional Layer Output
		conv1_out = ofmap_gen_conv(conv1_image, conv1_weights, conv1_biases);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/
		
		vector<vector<vector<vector<float>>>> conv2_weights(5, vector<vector<vector<float>>>(5, vector<vector<float>>(32, vector<float>(32, 0))));
		vector<float> conv2_biases(32, 0);
		vector<vector<vector<float>>> conv2_out(56, vector<vector<float>>(56, vector<float>(32,0)));
		
		conv2_weights = conv_weights("conv2_weights.bin", 5, 5, 32, 32);
		conv2_biases = get_biases("conv2_biases.bin", 32);
		//second Convlolutional Layer Output
		conv2_out = ofmap_gen_conv(conv1_out, conv2_weights, conv2_biases);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/

		//POOLLING!!!!!!!!!!!!!
		vector<vector<vector<float>>> pooling_out1(28, vector<vector<float>>(28, vector<float>(32,0)));
		//First Pooling Layer Done
		pooling_out1 = max_pooling_2D(conv2_out);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/
		
		vector<vector<vector<vector<float>>>> conv3_weights(3, vector<vector<vector<float>>>(3, vector<vector<float>>(32, vector<float>(64, 0))));
		vector<float> conv3_biases(64, 0);
		vector<vector<vector<float>>> conv3_out(26, vector<vector<float>>(26, vector<float>(64,0)));
		
		conv3_weights = conv_weights("conv3_weights.bin", 3, 3, 32, 64);
		conv3_biases = get_biases("conv3_biases.bin", 64);
		//Third Convlolutional Layer Output
		conv3_out = ofmap_gen_conv(pooling_out1, conv3_weights, conv3_biases);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/
		
		vector<vector<vector<vector<float>>>> conv4_weights(3, vector<vector<vector<float>>>(3, vector<vector<float>>(64, vector<float>(64, 0))));
		vector<float> conv4_biases(64, 0);
		vector<vector<vector<float>>> conv4_out(24, vector<vector<float>>(24, vector<float>(64,0)));
		
		conv4_weights = conv_weights("conv4_weights.bin", 3, 3, 64, 64);
		conv4_biases = get_biases("conv4_biases.bin", 64);
		//Fourth Convlolutional Layer Output
		conv4_out = ofmap_gen_conv(conv3_out, conv4_weights, conv4_biases);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/
		
		//POOLLING!!!!!!!!!!!!!
		vector<vector<vector<float>>> pooling_out2(12, vector<vector<float>>(12, vector<float>(64, 0)));
		//Second Pooling Layer Done
		pooling_out2 = max_pooling_2D(conv4_out);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/


		vector<vector<vector<vector<float>>>> conv5_weights(3, vector<vector<vector<float>>>(3, vector<vector<float>>(64, vector<float>(64, 0))));
		vector<float> conv5_biases(64, 0);
		vector<vector<vector<float>>> conv5_out(10, vector<vector<float>>(10, vector<float>(64,0)));
		
		conv5_weights = conv_weights("conv5_weights.bin", 3, 3, 64, 64);
		conv5_biases = get_biases("conv5_biases.bin", 64);
		//Fifth Convlolutional Layer Output
		conv5_out = ofmap_gen_conv(pooling_out2, conv5_weights, conv5_biases);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/

		vector<vector<vector<vector<float>>>> conv6_weights(3, vector<vector<vector<float>>>(3, vector<vector<float>>(64, vector<float>(128, 0))));
		vector<float> conv6_biases(128, 0);
		vector<vector<vector<float>>> conv6_out(8, vector<vector<float>>(8, vector<float>(128,0)));
		
		conv6_weights = conv_weights("conv6_weights.bin", 3, 3, 64, 128);
		conv6_biases = get_biases("conv6_biases.bin", 128);
		//Sixth Convlolutional Layer Output
		conv6_out = ofmap_gen_conv(conv5_out, conv6_weights, conv6_biases);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/


		//POOLLING!!!!!!!!!!!!!
		vector<vector<vector<float>>> pooling_out3(4, vector<vector<float>>(4, vector<float>(128, 0)));
		//Third Pooling Layer Done
		pooling_out3 = max_pooling_2D(conv6_out);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/

		vector<float> flat = flatten(pooling_out3);

		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/

		vector<vector<float>> dense1_weights(2048, vector<float>(256, 0));
		vector<float> dense1_biases(256, 0);
		vector<float> dense1_out(256, 0);
		

		dense1_weights = conv_weights("dense1_weights.bin", 2048, 256);
		dense1_biases = get_biases("dense1_biases.bin", 256);
		//First Dense Layer Output
		dense1_out = ofmap_gen_dense(flat, dense1_weights, dense1_biases, 256, false);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/

		vector<vector<float>> dense2_weights(256, vector<float>(200, 0));
		vector<float> dense2_biases(200, 0);
		vector<float> dense2_out(200, 0);
		
		dense2_weights = conv_weights("dense2_weights.bin", 256, 200);
		dense2_biases = get_biases("dense2_biases.bin", 200);
		//First Dense Layer Output
		dense2_out = ofmap_gen_dense(dense1_out, dense2_weights, dense2_biases, 200, true);
		/**
		=========================================
		Can do Comparisons with there output here
		=========================================
		**/
		
	return 0;

		// Execute the inference code and validate against the imported inference output. 
		// For each of the input, for all of the intermediate feature maps provide the binary files for both the imported feature maps from python (true value) and the ones predicted by your own C/C++ implementation.
		// Were you able to get similar final classification probability as the python version executing? if not what was the difference.
}

vector<vector<vector<float>>> image_import(char * fileName) {
	
	/* Input Data */
	float conv1_inputs[12288] = { 0 }; // reshape back to x*y*z
	vector<vector<vector<float>>> reshaped_inputs(64, vector<vector<float>>(64, vector<float>(3, 0)));

	
	FILE* ptr_input = fopen(fileName, "rb");  // r for read, b for binary
	int r2 = fread(conv1_inputs, sizeof(float), 12288, ptr_input);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_input);

	int i = 0;
	int f = 0;
	int j = 0;
	int count = 0;

	for (i = 0; i < 64; ++i) {
		for (f = 0; f < 64; ++f) {
			for (j = 0; j < 3; ++j) {
				reshaped_inputs[i][f][j] = conv1_inputs[count];
				count++;
			}
		}
	}

	return reshaped_inputs;
}

/*
Perform element-wise multiplication and partial-sum accumulation on singular filter with singular input fmap cross-section.
 */
float mult_and_accumulate(vector<vector<vector<float>>> weights_C, vector<vector<vector<float>>> input_fmap_C_partial) {

	/* Matrix Multiplication */
	float sum = 0;
	int x = 0;
	int y = 0;
	int z = 0;

	for (x = 0; x < weights_C.size(); x++) {
		for (y = 0; y < weights_C[0].size(); y++) {
			for (z = 0; z < weights_C[0][0].size(); z++) {
				sum += weights_C[x][y][z] * input_fmap_C_partial[x][y][z];
			}
		}
	}
	return sum;
}

/*
Generate 
 */
vector<vector<vector<float>>> ofmap_gen_conv(vector<vector<vector<float>>> input_fmap, vector<vector<vector<vector<float>>>> weights, vector<float> bias) {
	int x = 0;
	int y = 0;
	int z = 0;
	int filter_num = 0;
	int l = 0;
	int m = 0;
	int q = 0;

	vector<vector<vector<float>>> output((input_fmap.size() - weights.size()) + 1), vector<vector<float>>((input_fmap[0].size() - weights[0].size()) + 1), <vector<float>>(weights[0][0][0].size(), 0);

	for (filter_num = 0; filter_num < weights[0][0][0].size(); ++filter_num) { //filter number
		for (x = 0; x < weights.size(); x++) {  // S (length) of filter.
			for (y = 0; y < input_fmap[0].size() - weights[0].size(); y++) { //height
				for (z = 0; z < input_fmap[0][0].size() - weights.size(); z++) { //channel
					vector<vector<vector<float>>> fmap_3d_section(weights.size(), vector<vector<float>>(weights[0].size(), vector<float>(weights[0][0].size(), 0)));
					vector<vector<vector<float>>> 3d_weights(weights.size(), vector<vector<float>>(weights[0].size(), vector<float>(weights[0][0].size(), 0)));
					for (l = 0; l < weights.size(); ++l) {
						for (m = 0; m < weights[0].size(; ++m) {
							for(q = 0; q < weights[0][0].size(); ++q) {
								3d_weights[l][m][q] = weights[l][m][q][filter_num];
							}
						}
					}
					for (l = 0; l < x - input_fmaps.size(); ++l) {
						for (m = y; m < y - weights[0].size(); ++m) {
							for (q = x; q < weights[0][0].size(); ++q) {
								fmap_3d_section[l][m][q] = input_fmap[l][m][q];
							}
						}
					}
					output[x][y][filter_num] = ReLU(mult_and_accumulate(3d_weights, fmap_3d_section) + bias[filter_num]);
				}
			}
		}
	}
	return output;
}

vector<float> ofmap_gen_dense(vector<float> input_fmap, <vector<vector<float>> weights, vector<float> bias, int output_size, bool last_layer) {
	int x = 0;
	int y = 0;
	int z = 0;
	int multsum = 0;

	vector<float> output(output_size);
		for (x=0; x<output_size; x++) {
		multsum = 0;
			for (y=0; y<input_fmap.size(); y++) {
				for (z=0; z<output_size; z++) {
					multsum += input_fmap[y]*weights[y][z];
				}
			}
			if (!last_layer) {
				output[x] = ReLU(multsum + bias[x]);
			} else {
				output[x] = multsum + bias[x];
			}
		}
		if(last_layer) {
			output = softmax(output);
		}
	return output;
}

/*
Import weights from binary file (1D) and shape into 4D vector.
 */
vector<vector<vector<vector<float>>>> conv_weights(char * filename, int x, int y, int z, int w) {
	/* Weights Data */
	float conv1_weights[x*y*z*w] = { 0 }; // reshape back to x*y*z*w
	vector<vector<vector<vector<float>>>> reshaped_weights(x, vector<vector<vector<float>>>(y, vector<vector<float>>(z, vector<float>(w, 0))));

	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(conv1_weights, sizeof(float), x*y*z*w, ptr_weights);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_weights);

	int i = 0;
	int f = 0;
	int j = 0;
	int k = 0;
	int count = 0;

	for(i=0; i<x; ++i) {
		for (f = 0; f<y; ++f) {
			for (j=0; j<z; ++j) {
				for(k=0; k<w; ++k) {
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
vector<vector<float>> dense_weights(char * filename, int x, int y) {
	/* Weights Data */
	float dense_weights[x*y] = { 0 }; // reshape back to x*y
	vector<vector<float>> reshaped_weights(x, <vector<float>>(y, 0));

	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(dense_weights, sizeof(float), x*y, ptr_weights);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_weights);		
	int f = 0;
	int count = 0;

	for(i=0; i<x; ++i) {
		for (f = 0; f<y; ++f) {
			reshaped_weights[i][f] = dense_weights[count];
			count++;
		}
	}
	return reshaped_weights;
}

/*
 Purpose of this function is to take the intermediate layer outputs that are given to us to compare.
 */
vector<vector<vector<float>>> intermediate_compare_reshape(char * filename, int x, int y, int z) {
	/* Weights Data */
	float intermediate[x*y*z] = { 0 }; // reshape back to x*y*z
	vector<vector<vector<float>>> reshaped_intermediate(x, vector<vector<vector<float>>>(y, vector<vector<float>>(z, 0)));

	FILE* ptr_intermediate = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(intermediate, sizeof(float), x*y*z, ptr_intermediate);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_intermediate);

	int i = 0;
	int f = 0;
	int j = 0;
	int count = 0;

	for(i=0; i<x; ++i) {
		for (f = 0; f<y; ++f) {
			for (j=0; j<z; ++j) {
				reshaped_intermediate[i][f][j] = intermediate[count];
				count++;
			}
		}
	}

	return reshaped_intermediate;
}

vector<float> flatten(vector<vector<vector>>> in_layer) {
	vector<float> out(in_layer.size()*in_layer[0].size()*in_layer[0][0].size(), 0);
	int x = 0;
	int y = 0;
	int z = 0;
	count = 0;
	for(x=0; x<in_layer.size(); ++x) {
		for(y=0; y<in_layer[0].size(); ++y) {
			for(z=0; z<in_layer[0][0].size(); ++z) {
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
vector<float> get_biases(char * filename, int x)
/* Weights Data */
	float conv1_biases[x] = { 0 };  
	vector<float>biases(x, 0);

	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(conv1_biases, sizeof(float), x, ptr_weights);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_weights);

	int i=0;
	for(i=0; i<x; ++i) {
		biases[i] = conv1_biases[i];
	}

	return biases;
}

float ReLU (float num) {
	if (num < 0) {
		return 0;
	} else {
		return num;
	}
}

/*
Performs 2D max pooling (i.e. on each output channel).
 */
vector<vector<vector<float>>> max_pooling_2D(vector<vector<vector<float>>> ofmap_in) {
	int x = 0;	// Length ofmap_in
	int y = 0;  // Height ofmap_in
	int z = 0;  // Channel ofmap_in
	int x_sec = 0;
	int y_sec = 0;

	vector<vector<vector<float>>> output(ofmap_in.size() / 2, vector<vector<vector<float>>>(ofmap_in[0].size() / 2, vector<vector<float>>(ofmap_in[0][0].size(), 0)));

	for (z = 0; z < ofmap_in[0][0].size(); z++) {
		for (x = 0; x < ofmap_in.size(); x = x + 2) {
			for (y = 0; y < ofmap_in[0].size(); y = y + 2) {
				int max = 0;
				for (x_sec = x; x_sec < x + 2; x_sec++) {
					for (y_sec = y; y_sec < y + 2; y_sec++) {
						if (ofmap_in[x_sec][y_sec] > max) {
							max = ofmap_in[x_sec][y_sec];
						}
					}
				}
				output[x / 2][y / 2][z] = max;
			}
		}
	}

	return output;
}


vector<float> softmax(vector<float> input) {
	vector<float> output;
	int x = 0;
	int y = 0;
	int sum = 0;
	for (x=0; x<input.size(); x++) {
	sum = 0;
		for (y=0; y<input.size(); y++) {
			sum += exp(input[y]);
		}
		output[x] exp(input[x])/sum;
	}
}