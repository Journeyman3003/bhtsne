#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "tsne.h"

// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

    // Define parameters
	int origN, N, D, no_dims, max_iter, stop_lying_iter, restart_lying_iter, momentum_switch_iter, lying_factor;
	double perplexity, theta, eta, momentum, final_momentum, *data, *Y;
    int rand_seed = -1;
	// Define buildingblock adjusting parameters
	int input_similarities = 0; // 0 = Gaussian, 1 = Laplacian, 2 = Student (df = no_dims)
	int output_similarities = 0; // 0 = Student (df = 1), 1 = Laplacian
	int cost_function = 0; // 0 = KL
	int optimization = 0; // 0 = gradient descent


	//by default, conduct a random initialization
	bool skip_random_init = false;

    // Read the parameters and the dataset
	if(TSNE::load_data(&data, &origN, &D, &Y, &no_dims, &theta, &perplexity, &eta, &momentum, &final_momentum, &rand_seed,
		               &max_iter, &stop_lying_iter, &restart_lying_iter, &momentum_switch_iter, &lying_factor, &skip_random_init,
					   &input_similarities, &output_similarities, &cost_function, &optimization)) {

		// Make dummy landmarks
        N = origN;
        int* landmarks = (int*) malloc(N * sizeof(int));
        if(landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        for(int n = 0; n < N; n++) landmarks[n] = n;

		// Now fire up the SNE implementation
		//double* Y = (double*) malloc(N * no_dims * sizeof(double));
		double* costs = (double*) calloc(N, sizeof(double));
		if (costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        //if(Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }

		TSNE::run(data, N, D, Y, costs, landmarks, no_dims, perplexity, eta, momentum, final_momentum, theta, rand_seed, skip_random_init,
			      max_iter, lying_factor, stop_lying_iter, restart_lying_iter, momentum_switch_iter, input_similarities, output_similarities,
				  cost_function, optimization);

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
