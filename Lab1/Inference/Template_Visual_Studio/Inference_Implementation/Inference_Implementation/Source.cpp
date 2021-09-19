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


		/* Sample code to read a binary file (test.bin) exported from python into C:

		// SAMPLE CODE: Read a flattened 5*5*3*32 (2400 float values) convolution weight binary file (test.bin) into an array in C

		float conv1_weights[2400] = { 0 }; // reshape back to 5*5*3*32

		FILE* ptr_weights = fopen("C:\\Users\\mailr\\Downloads\\Conv_experiement\\conv1_weights.bin", "rb");  // r for read, b for binary
		int r2 = fread(conv1_weights, sizeof(float), 2400, ptr_weights);
		printf("Read weight values: %d\n", r2);
		fclose(ptr_weights);

		*/


		// Execute the inference code and validate against the imported inference output. 
		// For each of the input, for all of the intermediate feature maps provide the binary files for both the imported feature maps from python (true value) and the ones predicted by your own C/C++ implementation.
		// Were you able to get similar final classification probability as the python version executing? if not what was the difference.
}