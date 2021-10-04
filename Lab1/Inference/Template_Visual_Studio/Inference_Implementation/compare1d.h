#pragma once
template<int N>
// Parameters: N: compares two 1d arrays of size N and returns the maximum difference
float compare1d(float Input1[N], float Input2[N]) {
    // display the maximum diff
    double max_diff = 0;
    double curr_diff = 0;
    for (int i = 0; i < N; i++) {
       
                curr_diff = abs(Input1[i] - Input2[i]);
                if (curr_diff > max_diff) {
                    max_diff = curr_diff;
                }


      
    }
    return max_diff;

}