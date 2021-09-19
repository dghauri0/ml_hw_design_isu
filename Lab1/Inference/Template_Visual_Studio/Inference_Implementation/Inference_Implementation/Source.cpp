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

vector<vector<vector<float>>> conv1_input(char * fileName) {
	
	/* Input Data */
	float conv1_inputs[12288] = { 0 }; // reshape back to 64*64*3
	vector<vector<vector<float>>> reshaped_inputs(3, vector<vector<float>>(64, vector<float>(64, 0)));

	
	FILE* ptr_input = fopen(fileName, "rb");  // r for read, b for binary
	int r2 = fread(conv1_inputs, sizeof(float), 393216, ptr_input);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_input);

	int i = 0;
	int f = 0;
	int j = 0;
	int count = 0;

	for (i = 0; i < 3; ++i) {
		for (f = 0; f < 64; ++f) {
			for (j = 0; j < 64; ++j) {
				reshaped_inputs[i][f][j] = conv1_inputs[count];
				count++;
			}
		}
	}

	return reshaped_inputs;
}

vector<vector<vector<vector<float>>>> conv1_weights(char * filename) {
	/* Weights Data */
	float conv1_weights[393216] = { 0 }; // reshape back to 64*64*3*32
	vector<vector<vector<vector<float>>>> reshaped_weights(3, vector<vector<vector<float>>>(64, vector<vector<float>>(64, vector<float>(32, 0))));

	FILE* ptr_weights = fopen(filename, "rb");  // r for read, b for binary
	int r2 = fread(conv1_weights, sizeof(float), 393216, ptr_weights);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_weights);

	int i = 0;
	int f = 0;
	int j = 0;
	int k = 0;
	int count = 0;

	for(i=0; i<3; ++i) {
		for (f = 0; f<64; ++f) {
			for (j=0; j<64; ++j) {
				for(k=0; k<32; ++k) {
					reshaped_weights[i][f][j][k] = conv1_weights[count];
					count++;
				}
			}
		}
	}

	return reshaped_weights;
}

vector<vector<vector<float>>> conv2_input(char* fileName) {

	/* Input Data */
	float conv1_inputs[12288] = { 0 }; // reshape back to 64*64*3
	vector<vector<vector<float>>> reshaped_inputs(3, vector<vector<float>>(64, vector<float>(64, 0)));


	FILE* ptr_input = fopen(fileName, "rb");  // r for read, b for binary
	int r2 = fread(conv1_weights, sizeof(float), 393216, ptr_input);
	printf("Read weight values: %d\n", r2);
	fclose(ptr_input);

	int i = 0;
	int f = 0;
	int j = 0;
	int count = 0;

	for (i = 0; i < 3; ++i) {
		for (f = 0; f < 64; ++f) {
			for (j = 0; j < 64; ++j) {
				reshaped_inputs[i][f][j] = conv1_inputs[count];
				count++;
			}
		}
	}

	return reshaped_inputs;
}



