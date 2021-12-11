#include<stdio.h>255
#include<stdlib.h>
#include<vector>
#include<math.h>
#include <errno.h>
#include <chrono>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <inttypes.h>

using namespace std;

struct int4 {
	int8_t nibble : 4;
	//int number;
};

//RANDOM test1;

int main() {

        vector<vector<int4> > test(5, vector<int4> (5, 0));
	//uint4_t test = 8;
	//printf("test (uint4): %" PRIu4 "\n", uint4_t);
	return 0;

}

