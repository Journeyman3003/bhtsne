#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "tsne.h"

// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

    // Define some variables
	int origN, N, D, no_dims, max_iter;
	double perplexity, theta, lying_factor, *data;
    int rand_seed = -1;

    // Read the parameters and the dataset
	if(TSNE::load_data(&data, &origN, &D, &no_dims, &theta, &perplexity, &rand_seed, &max_iter, &lying_factor)) {

		// Make dummy landmarks
        N = origN;
        int* landmarks = (int*) malloc(N * sizeof(int));
        if(landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        for(int n = 0; n < N; n++) landmarks[n] = n;

		// Now fire up the SNE implementation
		double* Y = (double*) malloc(N * no_dims * sizeof(double));
		double* costs = (double*) calloc(N, sizeof(double));
        if(Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		// start lying iter set to 1001 to be unaffected, but beware when setting max iter > 1000!
		TSNE::run(data, N, D, Y, costs, landmarks, no_dims, perplexity, theta, rand_seed, false, max_iter, lying_factor, 250, 1001, 250);

		// Save the results
		// skipped for now as it is done after each 50 iterations within TSNE::run
		//TSNE::save_data(Y, landmarks, costs, N, no_dims, max_iter);

        // Clean up the memory
		free(data); data = NULL;
		free(Y); Y = NULL;
		free(costs); costs = NULL;
		free(landmarks); landmarks = NULL;
    }
}
