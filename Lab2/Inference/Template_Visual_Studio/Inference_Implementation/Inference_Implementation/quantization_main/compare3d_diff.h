#pragma once
template<int D, int N>
// Parameters: D, N: compares two 3d arrays of size D,D,N and returns the maximum difference
float compare(float Input1[D][D][N], float Input2[D][D][N]) {
    // display the maximum diff
    double max_diff = 0;
    double curr_diff = 0;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            for (int k = 0; k < N; k++) {
                /*if (output[i][j][k] == conv1_output_reshaped[i][j][k])
                    printf("Matches\n");*/
                    /*fprintf(fp2, "%.4f\n", output[i][j][k]);*/
             /*   printf("%f\n", conv2_output_reshaped[i][j][k]);*/
                curr_diff = abs(Input1[i][j][k] - Input2[i][j][k]);
                if (curr_diff > max_diff) {
                    max_diff = curr_diff;
                }


            }
        }
    }
    return max_diff;

}