#include<stdio.h>
#include<stdlib.h>

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


		// Execute the inference code and validate against the imported inference output. 
		// For each of the input, for all of the intermediate feature maps provide the binary files for both the imported feature maps from python (true value) and the ones predicted by your own C/C++ implementation.
		// Were you able to get similar final classification probability as the python version executing? if not what was the difference.
}

vector<vector<vector<float>>> conv_input(char * fileName, int x, int y, int z) {
	
	/* Input Data */
	float conv1_inputs[x*y*z] = { 0 }; // reshape back to x*y*z
	vector<vector<vector<float>>> reshaped_inputs(z, vector<vector<float>>(y, vector<float>(x, 0)));

	
	FILE* ptr_input = fopen(fileName, "rb");  // r for read, b for binary
	int r2 = fread(conv1_inputs, sizeof(float), x*y*z, ptr_input);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_input);

	int i = 0;
	int f = 0;
	int j = 0;
	int count = 0;

	for (i = 0; i < z; ++i) {
		for (f = 0; f < y; ++f) {
			for (j = 0; j < x; ++j) {
				reshaped_inputs[i][f][j] = conv1_inputs[count];
				count++;
			}
		}
	}

	return reshaped_inputs;
}

float mult_and_accumulate(vector<<vector<vector<float>>> weights_M, vector<vector<vector<float>>> input_fmap_M_partial) {
	
	/* Matrix Multiplication */
	float sum = 0;
	int x = 0;
	int y = 0;
	int z = 0;

	for (z = 0; z < weights_C.size(); z++) {
		for (y = 0; y < weights_C[0].size(); y++) {
			for (x = 0; x < weights_C[0][0].size(); x++) {
				sum += weights_C[z][y][x] * input_fmap_C_partial[][][];
			}
		}
	}
	return sum;
}

vector<vector<vector<vector<float>>>> ofmap_gen(vector<vector<vector<float>>> input_fmap, vector<vector<vector<vector<float>>>> weights) {
	int x = 0;
	int y = 0;
	int z = 0;
	int filter_num = 0;
	int l = 0;
	int m = 0;

	for (i = 0; i < weights[0][0].size(); i++) {

		for (j = 0; j < input_fmap.size() - weights.size(); j++) {
			for (k = 0; k < input_fmap.size() - weights.size(); k++) {
				< vector<vector<float>> cross_section (input_fmap.size(), vector<float>(input_fmap[0].size(), 0));
				< vector<vector<float>> fmap_cross (weights.size(), vector<float>(weights[0].size(), 0));
				for (l=0; l<weights.size(); ++l) {
					for(m=0; m<weights[0].size; ++m) {
						cross_section[l][m] = weights[l][m][i][0];
					}
				}
				for (l = j; l < j + weights.size(); ++l) {
					for (m = k; m < k + weights.size(); ++m) {
						fmap_cross[l][m] = input_fmap[l][m][i];
					}
				}
				mult_and_accumulate(cross_section, fmap_cross);
			}
		}
	}
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
					reshaped_weights[i][f][j][k] = conv1_weights[count];
					count++;
				}
			}
		}
	}

	return reshaped_weights;
}





