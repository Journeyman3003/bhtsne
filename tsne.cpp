/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include "vptree.h"
#include "sptree.h"
#include "tsne.h"


using namespace std;

static double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

static void zeroMean(double* X, int N, int D);
static void computeGaussianInputSimilarity(double* X, int N, int D, double* P, double perplexity);
static void computeGaussianInputSimilarity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K);
static void computeLaplacianInputSimilarity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K);
static void computeStudentInputSimilarity(double* X, int no_dims, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int K);
static double randn();
static void computeExactGradient(double* P, double* Y, int N, int D, double* dC);
static void computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
static double evaluateError(double* P, double* Y, int N, int D, double* costs);
static double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
static void symmetrizeMatrix(unsigned int** row_P, unsigned int** col_P, double** val_P, int N);

// Perform t-SNE
void TSNE::run(double* X, int N, int D, double* Y, double* costs, int* landmarks, int no_dims, double perplexity, double eta,
			   double momentum, double final_momentum, double theta, int rand_seed, bool skip_random_init, int max_iter, 
			   int lying_factor, int stop_lying_iter, int start_lying_iter, int mom_switch_iter, int input_similarities, 
			   int output_similarities, int cost_function, int optimization) {

    // Set random seed
    if (skip_random_init != true) {
      if(rand_seed >= 0) {
          printf("Using random seed: %d\n", rand_seed);
          srand((unsigned int) rand_seed);
      } else {
          printf("Using current time as random seed...\n");
          srand(time(NULL));
      }
    }

    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) { printf("Perplexity too large for the number of data points!\n"); exit(1); }
    //printf("Using no_dims = %d, perplexity = %f, exaggeration factor = %d, theta = %f\nlearning rate = %f, momentum = %f, final momentum = %f, momentum switch iter = %d\nstop lying iter = %d, restart lying iter = %d\n", no_dims, perplexity, lying_factor, theta, eta, momentum, final_momentum, mom_switch_iter, stop_lying_iter, start_lying_iter);
    bool exact = (theta == .0) ? true : false;

    // Set learning parameters
    float total_time = .0;
    clock_t start, end;

    // Allocate some memory
    double* dY    = (double*) malloc(N * no_dims * sizeof(double));
    double* uY    = (double*) malloc(N * no_dims * sizeof(double));
    double* gains = (double*) malloc(N * no_dims * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
    for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

    // Normalize input data (to prevent numerical problems)
    printf("Computing input similarities...\n");
    start = clock();
    zeroMean(X, N, D);
    double max_X = .0;
    for(int i = 0; i < N * D; i++) {
        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
    }
    for(int i = 0; i < N * D; i++) X[i] /= max_X;

    // Compute input similarities for exact t-SNE
    double* P; unsigned int* row_P; unsigned int* col_P; double* val_P;
    if(exact) {

        // Compute similarities
        printf("Exact?");
        P = (double*) malloc(N * N * sizeof(double));
        if(P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		computeGaussianInputSimilarity(X, N, D, P, perplexity);

        // Symmetrize input similarities
        printf("Symmetrizing...\n");
        int nN = 0;
        for(int n = 0; n < N; n++) {
            int mN = (n + 1) * N;
            for(int m = n + 1; m < N; m++) {
                P[nN + m] += P[mN + n];
                P[mN + n]  = P[nN + m];
                mN += N;
            }
            nN += N;
        }
        double sum_P = .0;
        for(int i = 0; i < N * N; i++) sum_P += P[i];
        for(int i = 0; i < N * N; i++) P[i] /= sum_P;
    }

    // Compute input similarities for approximate t-SNE
    else {
		// Compute asymmetric pairwise input similarities
		switch (input_similarities) {
			case 1: 
				computeLaplacianInputSimilarity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int)(3 * perplexity));
				break;
			case 2: 
				computeStudentInputSimilarity(X, no_dims, N, D, &row_P, &col_P, &val_P, (int)(3 * perplexity));
				break;
			default: 
				computeGaussianInputSimilarity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int)(3 * perplexity));
		}

        // Symmetrize input similarities
        symmetrizeMatrix(&row_P, &col_P, &val_P, N);
        double sum_P = .0;
        for(int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
        for(int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
    }
    end = clock();

    // Lie about the P-values
    if(exact) { for(int i = 0; i < N * N; i++)        P[i] *= lying_factor; } //default was 12.0
    else {      for(int i = 0; i < row_P[N]; i++) val_P[i] *= lying_factor; } //default was 12.0						

    if(exact) printf("Input similarities computed in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
    else printf("Input similarities computed in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N));
    
	// Initialize solution (randomly)
	if (skip_random_init != true) {
		printf("Initializing Y at random!\n");
		for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
	}
	else {
		printf("Skip random initialization of Y!\n");
	}

	// Save results of iteration 0
	double C = .0;
	if (exact) C = evaluateError(P, Y, N, no_dims, costs);
	else       C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta, costs);  // doing approximate computation here!
	printf("Initial Solution: error is %f\n", C);
	save_data(Y, landmarks, costs, N, no_dims, 0);

	// Perform main training loop
	start = clock();
	for(int iter = 0; iter < max_iter; iter++) {

        // Compute (approximate) gradient
        if(exact) computeExactGradient(P, Y, N, no_dims, dY);
        else computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta);

        // Update gains
        for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

        // Make solution zero-mean
		zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
        if(iter + 1 == stop_lying_iter) {
            if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= lying_factor; } //default was 12.0
            else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= lying_factor; } //default was 12.0
        }
        if(iter + 1 == mom_switch_iter) momentum = final_momentum;

		// Start lying againg about the P-values for the final iterations
		if (iter + 1 == start_lying_iter) {
			if (exact) { for (int i = 0; i < N * N; i++)        P[i] *= lying_factor; } //default was 12.0
			else { for (int i = 0; i < row_P[N]; i++) val_P[i] *= lying_factor; } //default was 12.0
		}

        // Print out progress and save intermediate results
        // if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
		if ((iter == 0 || (iter + 1) % 50 == 0 || iter == max_iter - 1)) {
            end = clock();
            double C = .0;
            if(exact) C = evaluateError(P, Y, N, no_dims, costs);
            else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta, costs);  // doing approximate computation here!
            if(iter == 0)
                printf("Iteration %d: error is %f\n", iter + 1, C);
            else {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter + 1, C, (float) (end - start) / CLOCKS_PER_SEC);
            }
			//no matter whether iteration is 0 or % 50 == 0 or max_iter - 1, save the current results
			save_data(Y, landmarks, costs, N, no_dims, iter + 1);

			start = clock();
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    if(exact) free(P);
    else {
        free(row_P); row_P = NULL;
        free(col_P); col_P = NULL;
        free(val_P); val_P = NULL;
    }
    printf("Fitting performed in %4.2f seconds.\n", total_time);
}




// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
static void computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{

    // Construct space-partitioning tree on current map
    SPTree* tree = new SPTree(D, Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

    // Compute final t-SNE gradient
    for(int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Compute gradient of the t-SNE cost function (exact)
static void computeExactGradient(double* P, double* Y, int N, int D, double* dC) {

	// Make sure the current gradient contains zeros
	for(int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc(N * N * sizeof(double));
    if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    double sum_Q = .0;
    int nN = 0;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	// Perform the computation of the gradient
    nN = 0;
    int nD = 0;
	for(int n = 0; n < N; n++) {
        int mD = 0;
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
                for(int d = 0; d < D; d++) {
                    dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                }
            }
            mD += D;
		}
        nN += N;
        nD += D;
	}

    // Free memory
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
}


// Evaluate t-SNE cost function (exactly)
static double evaluateError(double* P, double* Y, int N, int D, double* costs) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
            else Q[nN + m] = DBL_MIN;
        }
        nN += N;
    }
    for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;

    // Sum t-SNE error
    double C = .0;
	// i = Data Point i
	for(int i = 0; i < N; i++) {
		// aggregated Cost of Point i
		double Ci = .0;
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			C += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (Q[i * N + j] + FLT_MIN));
			Ci += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (Q[i * N + j] + FLT_MIN));
		}
		//write to costs
		costs[i] = Ci;
	}

    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

// Evaluate t-SNE cost function (approximately)
static double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs)
{

    // Get estimate of normalization term
    SPTree* tree = new SPTree(D, Y, N);
    double* buff = (double*) calloc(D, sizeof(double));
    double sum_Q = .0;
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for(int n = 0; n < N; n++) {
        ind1 = n * D;
		// variable to store cost of point n
		double Cn = .0;
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * D;
            for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
            for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
            for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
			Cn += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));

        }
		costs[n] = Cn;
    }

    // Clean up memory
    free(buff);
    delete tree;
    return C;
}


// Compute gaussian input similarities with a fixed perplexity
static void computeGaussianInputSimilarity(double* X, int N, int D, double* P, double perplexity) {

	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
    int nN = 0;
	for(int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
        double sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += P[nN + m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row normalize P
		for(int m = 0; m < N; m++) P[nN + m] /= sum_P;
        nN += N;
	}

	// Clean up memory
	free(DD); DD = NULL;
}


// Compute gaussian input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
static void computeGaussianInputSimilarity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);
	printf("GAUSSIAN...\n");
    // Loop over all points to find nearest neighbors
    printf("Building tree...\n");
    vector<DataPoint> indices;
    vector<double> distances;
    for(int n = 0; n < N; n++) {

        if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
            double H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}

// Compute laplacian input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
static void computeLaplacianInputSimilarity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K) {

	if (perplexity > K) printf("Perplexity should be lower than K!\n");

	// Allocate the memory we need
	*_row_P = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
	*_col_P = (unsigned int*)calloc(N * K, sizeof(unsigned int));
	*_val_P = (double*)calloc(N * K, sizeof(double));
	if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	unsigned int* row_P = *_row_P;
	unsigned int* col_P = *_col_P;
	double* val_P = *_val_P;
	double* cur_P = (double*)malloc((N - 1) * sizeof(double));
	if (cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	row_P[0] = 0;
	for (int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int)K;

	// Build ball tree on data set
	VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
	vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
	for (int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
	tree->create(obj_X);

	// Loop over all points to find nearest neighbors
	printf("Building tree...\n");
	printf("LAPLACIAN...\n");
	vector<DataPoint> indices;
	vector<double> distances;
	for (int n = 0; n < N; n++) {

		if (n % 10000 == 0) printf(" - point %d of %d\n", n, N);

		// Find nearest neighbors
		indices.clear();
		distances.clear();
		tree->search(obj_X[n], K + 1, &indices, &distances);

		// Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while (!found && iter < 200) {

			// Compute Laplacian kernel row
			for (int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1]);

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < K; m++) sum_P += cur_P[m];
			double H = .0;
			for (int m = 0; m < K; m++) H += beta * (distances[m + 1] * cur_P[m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if (Hdiff > 0) {
					min_beta = beta;
					if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize current row of P and store in matrix
		for (unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
		for (unsigned int m = 0; m < K; m++) {
			col_P[row_P[n] + m] = (unsigned int)indices[m + 1].index();
			val_P[row_P[n] + m] = cur_P[m];
		}
	}

	// Clean up memory
	obj_X.clear();
	free(cur_P);
	delete tree;
}

// Compute student input similarities with a fixed t = no_dims using ball trees (this function allocates memory another function should free)
static void computeStudentInputSimilarity(double* X, int no_dims, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int K) {

	// Allocate the memory we need
	*_row_P = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
	*_col_P = (unsigned int*)calloc(N * K, sizeof(unsigned int));
	*_val_P = (double*)calloc(N * K, sizeof(double));
	if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	unsigned int* row_P = *_row_P;
	unsigned int* col_P = *_col_P;
	double* val_P = *_val_P;
	double* cur_P = (double*)malloc((N - 1) * sizeof(double));
	if (cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	row_P[0] = 0;
	for (int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int)K;

	// Build ball tree on data set
	VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
	vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
	for (int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
	tree->create(obj_X);
	printf("STUDENT...\n");
	// Loop over all points to find nearest neighbors
	printf("Building tree...\n");
	vector<DataPoint> indices;
	vector<double> distances;
	for (int n = 0; n < N; n++) {

		if (n % 10000 == 0) printf(" - point %d of %d\n", n, N);

		// Find nearest neighbors
		indices.clear();
		distances.clear();
		tree->search(obj_X[n], K + 1, &indices, &distances);

		// Compute Student kernel row with df = no_dims
		for (int m = 0; m < K; m++) cur_P[m] = pow(1 + dist[m + 1] * dist[m + 1] / no_dims, -(no_dims + 1)/2)

		double sum_P = DBL_MIN;
		for (int m = 0; m < K; m++) sum_P += cur_P[m];

		// Row-normalize current row of P and store in matrix
		for (unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
		for (unsigned int m = 0; m < K; m++) {
			col_P[row_P[n] + m] = (unsigned int)indices[m + 1].index();
			val_P[row_P[n] + m] = cur_P[m];
		}
	}

	// Clean up memory
	obj_X.clear();
	free(cur_P);
	delete tree;
}


// Symmetrizes a sparse matrix
static void symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
    unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    const double* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}


// Makes data zero-mean
static void zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}
    free(mean); mean = NULL;
}


// Generates a Gaussian random number
static double randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(double** data, int* n, int* d, double** initial_solution, int* no_dims, double* theta, double* perplexity, double* eta, double* momentum, double* final_momentum,
					 int* rand_seed, int* max_iter, int* stop_lying_iter, int* restart_lying_iter, int* momentum_switch_iter, int* lying_factor, bool* skip_random_init,
					 int* input_similarities, int* output_similarities, int* cost_function, int* optimization) {
	// Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
	if((h = fopen("data.dat", "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
	// Load hyperparameters

	fread(n, sizeof(int), 1, h);											// number of datapoints
	printf("number of data points = %d\n", *n);

	fread(d, sizeof(int), 1, h);											// original dimensionality
	printf("original dimensionality = %d \n", *d);

    fread(theta, sizeof(double), 1, h);										// gradient accuracy
	printf("gradient accuracy = %f\n", *theta);

	fread(perplexity, sizeof(double), 1, h);								// perplexity
	printf("perplexity = %f\n", *perplexity);

	fread(eta, sizeof(double), 1, h);										// learning rate eta
	printf("eta = %f\n", *eta);

	fread(momentum, sizeof(double), 1, h);									// momentum
	printf("momentum = %f\n", *momentum);

	fread(final_momentum, sizeof(double), 1, h);							// final momentum
	printf("final momentum = %f\n", *final_momentum);

	fread(no_dims, sizeof(int), 1, h);                                      // output dimensionality
	printf("output dimensionality = %d\n", *no_dims);

    fread(max_iter, sizeof(int),1,h);                                       // maximum number of iterations
	printf("max iterations = %d\n", *max_iter);

	fread(stop_lying_iter, sizeof(int), 1, h);                              // iteration when to stop lying about P values
	printf("iteration when to stop lying about P values = %d\n", *stop_lying_iter);

	fread(restart_lying_iter, sizeof(int), 1, h);                           // iteration when to restart lying about P values
	printf("iteration when to restart lying about P values = %d\n", *restart_lying_iter);

	fread(momentum_switch_iter, sizeof(int), 1, h);                         // iteration when to switch momentum to final momentum
	printf("iteration when to switch momentum to final momentum = %d\n", *momentum_switch_iter);

	fread(lying_factor, sizeof(int), 1, h);									// lying/exaggeration factor
	printf("lying factor = %d\n", *lying_factor);

	// Load building block definitions

	printf("loading building block definitions:\n");
	fread(input_similarities, sizeof(int), 1, h);							// input similarities
	printf("input similarities = %d\n", *input_similarities);

	fread(output_similarities, sizeof(int), 1, h);							// output similarities
	printf("output similarities = %d\n", *output_similarities);

	fread(cost_function, sizeof(int), 1, h);								// cost function
	printf("cost function = %d\n", *cost_function);

	fread(optimization, sizeof(int), 1, h);									// optimization
	printf("optimization = %d\n", *optimization);

	// Data and optional randseed/initial solution

	*data = (double*) malloc(*n * *d * sizeof(double));
    if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    fread(*data, sizeof(double), *n * *d, h);                               // the data

	fread(rand_seed, sizeof(int), 1, h);
	printf("Random seed set to %i\n", *rand_seed);							// random seed (even if no seed is passed, it is set to -1 (thanks to fread default passing amount of successfully read elements)
	
	*initial_solution = (double*)malloc(*n * *no_dims * sizeof(double));
	if (*initial_solution == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	fread(*initial_solution, sizeof(double), *n * *no_dims, h);				// the initial solution

	if (!feof(h)) { //this should kick in if there was no initial solution (or not even a rand-seed) provided
		// if there is an initial solution provided, set skip random init to true
		printf("Initial solution Y is provided!\n");
		*skip_random_init = true;		
	}																		
	
	fclose(h);
	printf("Read the %i x %i data matrix successfully!\n", *n, *d);
	return true;
}

// Function that saves map to a t-SNE file
void TSNE::save_data(double * data, int * landmarks, double * costs, int n, int d, int iter) {
	// Open file, write first 3 integers and then the data
	FILE *h;
	string filename = "result-" + to_string(iter) + ".dat";
	if ((h = fopen(filename.c_str(), "w+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	//save number of iterations in addition
	fwrite(&iter, sizeof(int), 1, h);
	//number of observations
	fwrite(&n, sizeof(int), 1, h);
	//number of target dimensions
	fwrite(&d, sizeof(int), 1, h);
	//actual data
	fwrite(data, sizeof(double), n * d, h);
	//landmarks to assure the order of points can be restored
	fwrite(landmarks, sizeof(int), n, h);
	//costs for each sample
	fwrite(costs, sizeof(double), n, h);
	fclose(h);
	//printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}
