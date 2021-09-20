#include<stdio.h>
#include<stdlib.h>

template<int D, int N, int M, int s, int f>

// Implement the layers as functions


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

		/*
		Questions:
		
		1. Are we feeding the outputs of previous layers into the next layer OR are we just taking the given input for each layer based on the given bin files?
			a. Are we using the given bin files inside the Test_Input0,1,2 directories? Or are we supposed to be generating all of those bin files from python?
		2. Why are the intermediate files flattened?
		3. Are we outputing a bin file after each convolutional layer?
		*/
	return 0;

		// Execute the inference code and validate against the imported inference output. 
		// For each of the input, for all of the intermediate feature maps provide the binary files for both the imported feature maps from python (true value) and the ones predicted by your own C/C++ implementation.
		// Were you able to get similar final classification probability as the python version executing? if not what was the difference.
}

vector<vector<vector<float>>> image_import(char * fileName) {
	f
	
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

// DONE
float mult_and_accumulate(vector<vector<vector<float>>> weights_M, vector<vector<vector<float>>> input_fmap_M_partial) {

	/* Matrix Multiplication */
	float sum = 0;
	int x = 0;
	int y = 0;
	int z = 0;

	for (z = 0; z < weights_C[0][0].size(); z++) {
		for (y = 0; y < weights_C[0].size(); y++) {
			for (x = 0; x < weights_C.size(); x++) {
				sum += weights_C[x][y][z] * input_fmap_C_partial[x][y][z];
			}
		}
	}
	return sum;
}

vector<vector<vector<float>>> ofmap_gen(vector<vector<vector<float>>> input_fmap, vector<vector<vector<vector<float>>>> weights, float bias[2050]) {
	int x = 0;
	int y = 0;
	int z = 0;
	int filter_num = 0;
	int l = 0;
	int m = 0;
	int q = 0;
	vector<vector<vector<float>>> output(weights[0][0][0].size(), vector<vector<float>>((input_fmap[0].size() - weights[0].size()) + 1), <vector<float>>((input_fmap.size() - weights.size()) + 1));
	 

	for (filter_num = 0; filter_num < weights[0][0][0].size(); ++filter_num) {
		for (z = 0; z < weights[0][0].size(); z++) {
			for (y = 0; y < input_fmap[0].size() - weights[0].size(); y++) {
				for (x = 0; x < input_fmap.size() - weights.size(); x++) {
					vector<vector<vector<float>>> fmap_3d_section(input_fmap.size(), vector<vector<float>>(input_fmap[0].size(), vector<float>(input_fmap[0][0].size(), 0)));
					vector<vector<vector<float>>> 3d_weights(weights.size(), vector<vector<float>>(weights[0].size(), vector<float>(weights[0][0].size(), 0)));
					for (l = 0; l < weights[0][0].size(); ++l) {
						for (m = 0; m < weights[0].size(; ++m) {
							for(q = 0; q < weights.size(); ++q) {
								3d_weights[q][m][l] = weights[q][m][l][filter_num];
							}
						}
					}
					for (l = 0; l < input_fmaps[0][0].size(); ++l) {
						for (m = y; m < y - weights[0].size(); ++m) {
							for (q = x; q < x - weights.size(); ++q) {
								fmap_3d_section[q][m][l] = input_fmap[q][m][l];
							}
						}
					}
					//TODO: filter_num and x need to be switched when comparing
					output[x][y][filter_num] = ReLU(mult_and_accumulate(3d_weights, fmap_3d_section) + bias[filter_num]);
				}
			}
		}
	}
	return output;
}


vector<vector<vector<vector<float>>>> conv_weights(char * filename, int x, int y, int z, int w) {
	/* Weights Data */
	float conv1_weights[x*y*z*w] = { 0 }; // reshape back to x*y*z*w
	vector<vector<vector<vector<float>>>> reshaped_weights(z, vector<vector<vector<float>>>(y, vector<vector<float>>(x, vector<float>(w, 0))));

	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(conv1_weights, sizeof(float), x*y*z*w, ptr_weights);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_weights);

	int i = 0;
	int f = 0;
	int j = 0;
	int k = 0;
	int count = 0;

	for(i=0; i<z; ++i) {
		for (f = 0; f<y; ++f) {
			for (j=0; j<x; ++j) {
				for(k=0; k<w; ++k) {
					reshaped_weights[j][f][i][k] = conv1_weights[count];
					count++;
				}
			}
		}
	}

	return reshaped_weights;
}

float conv_biases(char * filename, int x)
/* Weights Data */
	float conv1_biases[x] = { 0 }; // reshape back to x*y*z*w

	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(conv1_biases, sizeof(float), x, ptr_weights);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_weights);

	return conv1_biases;
}

float ReLU (float num) {
	if (num < 0) {
		return 0;
	} else {
		return num;
	}
}



