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

//const unsigned int POP_SIZE = 100;

static double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

static void zeroMean(double* X, int N, int D);
// Exact input similarities
static void computeExactGaussianInputSimilarity(double* X, int N, int D, double* P, double perplexity);
static void computeExactLaplacianInputSimilarity(double* X, int N, int D, double* P, double perplexity);
static void computeExactStudentInputSimilarity(double* X, int no_dims, int N, int D, double* P);
// BH input similarities
static void computeGaussianInputSimilarity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K);
static void computeLaplacianInputSimilarity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K);
static void computeStudentInputSimilarity(double* X, int no_dims, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int K);
static double randn();
static double randn(double mean, double stdDev);

// Exact gradients
static void computeExactGradientKL(double* P, double* Y, int N, int D, int freeze_index, double* dC);
// using ChiSq distributed output similarities Q
static void computeExactGradientKLChiSq(double* P, double* Y, int N, int D, double* dC);
// using Student0.5 distributed output similarities Q
static void computeExactGradientKLStudentHalf(double* P, double* Y, int N, int D, double* dC);
// using StudentAlpha distributed output similarities Q
static void computeExactGradientKLStudentAlpha(double* P, double* Y, int N, int D, double alpha, double* dC);
static void computeExactGradientRKL(double* P, double* Y, int N, int D, double* dC);
static void computeExactGradientJS(double* P, double* Y, int N, int D, double* dC);

// approximated exact gradients
static void approximateExactGradient(double* P, double* Y, int N, int D, double* dC, double costFunc(double*, int, double*));

// BH approximated gradients
static void computeApproxGradientKL(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, int freeze_index, double* dC, double theta);
// using ChiSq distributed output similarities Q
static void computeApproxGradientKLChiSq(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
// using Student0.5 distributed output similarities Q
static void computeApproxGradientKLStudentHalf(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
// using StudentAlpha distributed output similarities Q
static void computeApproxGradientKLStudentAlpha(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double alpha, double* dC, double theta);
static void computeApproxGradientRKL(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
static void computeApproxGradientJS(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);

static void approximateApproxGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D,
									  double* dC, double theta, double costFunc(unsigned int*, unsigned int*, double*, double*, int, int, double, double*));

// Exact cost functions
static double evaluateExactErrorKL(double* P, double* Y, int N, int D, double* costs);
static double evaluateExactErrorKL(double* P, int N, double* Q);
// using ChiSq distributed output similarities Q
static double evaluateExactErrorKLChiSq(double* P, double* Y, int N, int D, double* costs);
// using Student0.5 distributed output similarities Q
static double evaluateExactErrorKLStudentHalf(double* P, double* Y, int N, int D, double* costs);
// using StudentAlpha distributed output similarities Q
static double evaluateExactErrorKLStudentAlpha(double* P, double* Y, int N, int D, double alpha, double* costs);
static double evaluateExactErrorRKL(double* P, double* Y, int N, int D, double* costs);
static double evaluateExactErrorRKL(double* P, int N, double* Q);
static double evaluateExactErrorJS(double* P, double* Y, int N, int D, double* costs);
static double evaluateExactErrorJS(double* P, int N, double* Q);
// helper to compute KL(Q||M)
static double evaluateExactErrorJSQM(double* P, int N, double* Q);

// BH approximated cost functions
static double evaluateApproxErrorKL(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
// using ChiSq distributed output similarities Q
static double evaluateApproxErrorKLChiSq(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
// using Student0.5 distributed output similarities Q
static double evaluateApproxErrorKLStudentHalf(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
// using StudentAlpha distributed output similarities Q
static double evaluateApproxErrorKLStudentAlpha(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double alpha, double theta, double* costs);
static double evaluateApproxErrorRKL(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
static double evaluateApproxErrorJS(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
// helper to compute KL(P||M)
//static double evaluateApproxErrorJSPM(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
// helper to compute KL(Q||M)
static double evaluateApproxErrorJSQM(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta);

static void computeEuclideanDistance(double* X, int N, int D, double* DD, bool squared);
static void updateEuclideanDistance(double* X, int N, int update_index, int D, double* DD, bool squared);
static void symmetrizeMatrix(unsigned int** row_P, unsigned int** col_P, double** val_P, int N);

// genetic algorithm optimizer
//static void computeExactGeneticOptimization(double* P, double* Y, double* Y_genomes, int N, int D, double costFunc(double*, int, double*));
static void computeExactGeneticOptimization(double* P, double* Y, int N, int iter, int D, double costFunc(double*, double*, int, int, double*));
static void computeApproxGeneticOptimization(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int iter, int D, double theta,
	double costFunc(unsigned int*, unsigned int*, double*, double*, int, int, double, double*));
static vector<pair<double, int>> sortArr(vector<double> arr, int n);

// Perform t-SNE
void TSNE::run(double* X, int N, int D, double* Y, double* costs, int* landmarks, int no_dims, double perplexity, double eta,
			   double momentum, double final_momentum, double theta, int rand_seed, bool skip_random_init, int max_iter, 
			   int lying_factor, int stop_lying_iter, int start_lying_iter, int mom_switch_iter, int input_similarities, 
			   int output_similarities, int cost_function, int optimization, int freeze_index) {
	double alpha = 0.1;
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
	double* dY = (double*)malloc(N * no_dims * sizeof(double));
	double* uY = (double*)malloc(N * no_dims * sizeof(double));
	double* gains = (double*)malloc(N * no_dims * sizeof(double));
	if (dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int i = 0; i < N * no_dims; i++)    uY[i] = .0;
	for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

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
		switch (input_similarities) {
		case 1:
			computeExactLaplacianInputSimilarity(X, N, D, P, perplexity);
			break;
		case 2:
			computeExactStudentInputSimilarity(X, no_dims, N, D, P);
			break;
		default:
			computeExactGaussianInputSimilarity(X, N, D, P, perplexity);
		}
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

	//always randomly init Y for genetic optimizer
	if (optimization == 1) {
		printf("Initializing Y (Chromosomes) at random!\n");
		for (int i = 0; i < N * no_dims * N; i++) Y[i] = randn() * .0001;
	}
	// Initialize solution (randomly)
	else {
		if (skip_random_init != true) {
			printf("Initializing Y at random!\n");
			for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
		}
		else {
			printf("Skip random initialization of Y!\n");
			// check whether a freeze index is passed that is > 0
			// which would indicate the remainder freeze_index < n < N to be initialized at random
			if (freeze_index != 0) {
				printf("Still, test observations are initialized at random!\n");
				for (int i = freeze_index * no_dims; i < N * no_dims; i++) Y[i] = randn() * .0001;
			}
		}
	}

	string cost_function_name;

	// Save results of iteration 0
	double C = .0;
	if (exact) {
		switch (cost_function) {
		case 1:
			C = evaluateExactErrorRKL(P, Y, N, no_dims, costs);
			cost_function_name = "RKL";
			break;
		case 2:
			C = evaluateExactErrorJS(P, Y, N, no_dims, costs);
			cost_function_name = "JS";
			break;
		default:
			cost_function_name = "KL";
			switch (output_similarities) {
			case 1:
				C = evaluateExactErrorKLChiSq(P, Y, N, no_dims, costs);
				break;
			case 2:
				C = evaluateExactErrorKLStudentHalf(P, Y, N, no_dims, costs);
				break;
			case 3:
				C = evaluateExactErrorKLStudentAlpha(P, Y, N, no_dims, alpha, costs);
				break;
			default:
				C = evaluateExactErrorKL(P, Y, N, no_dims, costs);
			}
		}
	}
	else { // doing approximate computation here!
		switch (cost_function) {
		case 1:
			cost_function_name = "RKL";
			C = evaluateApproxErrorRKL(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
			break;
		case 2:
			cost_function_name = "JS";
			C = evaluateApproxErrorJS(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
			break;
		default:
			cost_function_name = "KL";
			switch (output_similarities) {
			case 1:
				C = evaluateApproxErrorKLChiSq(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
				break;
			case 2:
				C = evaluateApproxErrorKLStudentHalf(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
				break;
			case 3:
				C = evaluateApproxErrorKLStudentAlpha(row_P, col_P, val_P, Y, N, no_dims, alpha, theta, costs);
				break;
			default:
				C = evaluateApproxErrorKL(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
				//C_compare = evaluateApproxErrorJS(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
			}
		}
	}
	printf("Initial Solution: %s error is %f\n", cost_function_name.c_str(), C);
	save_data(Y, landmarks, costs, N, no_dims, 0);

	// Perform main training loop
	start = clock();
	for(int iter = 0; iter < max_iter; iter++) {

		// set costs to 0
		for (int i = 0; i < N; i++) costs[i] = 0.0;

		if (optimization == 0) {

			// Compute gradient
			if (exact) {
				switch (cost_function) {
				case 1:
					computeExactGradientRKL(P, Y, N, no_dims, dY);
					//approximateExactGradient(P, Y, N, no_dims, dY, evaluateExactErrorRKL);
					break;
				case 2:
					computeExactGradientJS(P, Y, N, no_dims, dY);
					//approximateExactGradient(P, Y, N, no_dims, dY, evaluateExactErrorJS);
					break;
				default:
					switch (output_similarities) {
					case 1:
						computeExactGradientKLChiSq(P, Y, N, no_dims, dY);
						break;
					case 2:
						computeExactGradientKLStudentHalf(P, Y, N, no_dims, dY);
						break;
					case 3:
						computeExactGradientKLStudentAlpha(P, Y, N, no_dims, alpha, dY);
						break;
					default:
						computeExactGradientKL(P, Y, N, no_dims, freeze_index, dY);
						//approximateExactGradient(P, Y, N, no_dims, dY, evaluateExactErrorKL);
						break;
					}
				}
			}
			// Compute approximate gradient
			else {
				switch (cost_function) {
				case 1:
					//computeApproxGradientRKL(row_P, col_P, val_P, Y, N, no_dims, dY, theta);
					approximateApproxGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta, evaluateApproxErrorRKL);
					break;
				case 2:
					computeApproxGradientJS(row_P, col_P, val_P, Y, N, no_dims, dY, theta);
					//approximateApproxGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta, evaluateApproxErrorJS);
					break;
				default:
					switch (output_similarities) {
					case 1:
						computeApproxGradientKLChiSq(row_P, col_P, val_P, Y, N, no_dims, dY, theta);
						break;
					case 2:
						computeApproxGradientKLStudentHalf(row_P, col_P, val_P, Y, N, no_dims, dY, theta);
						break;
					case 3:
						computeApproxGradientKLStudentAlpha(row_P, col_P, val_P, Y, N, no_dims, alpha, dY, theta);
						break;
					default:
						computeApproxGradientKL(row_P, col_P, val_P, Y, N, no_dims, freeze_index, dY, theta);
						//approximateApproxGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta, evaluateApproxErrorKL);
						break;
					}
				}
			}

			// Update gains
			for (int i = freeze_index * no_dims; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
			for (int i = freeze_index * no_dims; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;

			// Perform gradient update (with momentum and gains)
			for (int i = freeze_index * no_dims; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
			for (int i = freeze_index * no_dims; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
		}
		else {
			if (exact) {
				computeExactGeneticOptimization(P, Y, N, iter, no_dims, evaluateExactErrorKL);
			}
			else {
				computeApproxGeneticOptimization(row_P, col_P, val_P, Y, N, iter, no_dims, theta, evaluateApproxErrorKL);
			}
		}

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
            C = .0;
			if (exact) {
				switch (cost_function) {
				case 1:
					C = evaluateExactErrorRKL(P, Y, N, no_dims, costs);
					break;
				case 2:
					C = evaluateExactErrorJS(P, Y, N, no_dims, costs);
					break;
				default:
					switch (output_similarities) {
					case 1:
						C = evaluateExactErrorKLChiSq(P, Y, N, no_dims, costs);
						break;
					case 2:
						C = evaluateExactErrorKLStudentHalf(P, Y, N, no_dims, costs);
						break;
					case 3:
						C = evaluateExactErrorKLStudentAlpha(P, Y, N, no_dims, alpha, costs);
						break;
					default:
						C = evaluateExactErrorKL(P, Y, N, no_dims, costs);
					}
				}
			}
			else { // doing approximate computation here!
				switch (cost_function) {
				case 1:
					C = evaluateApproxErrorRKL(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
					break;
				case 2:
					C = evaluateApproxErrorJS(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
					break;
				default:
					switch (output_similarities) {
					case 1:
						C = evaluateApproxErrorKLChiSq(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
						break;
					case 2:
						C = evaluateApproxErrorKLStudentHalf(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
						break;
					case 3:
						C = evaluateApproxErrorKLStudentAlpha(row_P, col_P, val_P, Y, N, no_dims, alpha, theta, costs);
						break;
					default:
						C = evaluateApproxErrorKL(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
						//C_compare = evaluateApproxErrorJS(row_P, col_P, val_P, Y, N, no_dims, theta, costs);
					}
				}
			}

			if (iter == 0) {
				printf("Iteration %d: %s error is %f\n", iter + 1, cost_function_name.c_str(), C);
			}
            else {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                printf("Iteration %d: %s error is %f (50 iterations in %4.2f seconds)\n", iter + 1, cost_function_name.c_str(), C, (float) (end - start) / CLOCKS_PER_SEC);
				//printf("Iteration %d: JS error would be %f (50 iterations in %4.2f seconds)\n", iter + 1, C_compare, (float)(end - start) / CLOCKS_PER_SEC);
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


// Compute gradient of the t-SNE cost function (KL - using Barnes-Hut algorithm)
static void computeApproxGradientKL(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, int freeze_index, double* dC, double theta)
{

    // Construct space-partitioning tree on current map
    SPTree* tree = new SPTree(D, Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForcesKL(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
    for(int n = 0; n < N; n++) tree->computeNonEdgeForcesKL(n, theta, neg_f + n * D, &sum_Q);

    // Compute final t-SNE gradient
    for(int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Compute gradient of the t-SNE cost function (KL - ChiSq - using Barnes-Hut algorithm)
void computeApproxGradientKLChiSq(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{
	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;
	double* pos_f = (double*)calloc(N * D, sizeof(double));
	double* neg_f = (double*)calloc(N * D, sizeof(double));
	if (pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	
	// compute Edge forces
	unsigned int ind1 = 0;
	unsigned int ind2 = 0;
	for (unsigned int n = 0; n < N; n++) {
		for (unsigned int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {

			// Compute pairwise distance and Q-value
			double dist = 0.0;
			ind2 = inp_col_P[i] * D;
			for (unsigned int d = 0; d < D; d++) {
				double diff = Y[ind1 + d] - Y[ind2 + d];
				dist += diff * diff;
			}

			dist = sqrt(dist);

			double mult = inp_val_P[i] / dist;

			// Sum positive force
			for (unsigned int d = 0; d < D; d++) {
				double diff = Y[ind1 + d] - Y[ind2 + d];
				pos_f[ind1 + d] += mult * diff;
			}
		}
		ind1 += D;
	}

	for (int n = 0; n < N; n++) tree->computeNonEdgeForcesKLChiSq(n, theta, neg_f + n * D, &sum_Q);

	// Compute final t-SNE gradient
	for (int i = 0; i < N * D; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);
	delete tree;
}


void computeApproxGradientKLStudentHalf(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{
	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;
	double* pos_f = (double*)calloc(N * D, sizeof(double));
	double* neg_f = (double*)calloc(N * D, sizeof(double));
	if (pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	

	// Loop over all edges in the graph
	unsigned int ind1 = 0;
	unsigned int ind2 = 0;
	
	for (unsigned int n = 0; n < N; n++) {
		for (unsigned int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {

			// Compute pairwise distance and Q-value
			double dist = 0.0;
			ind2 = inp_col_P[i] * D;
			for (unsigned int d = 0; d < D; d++) {
				double diff = Y[ind1 + d] - Y[ind2 + d];
				dist += diff * diff;
			}
			double mult = inp_val_P[i] / (1 + 2 * dist);

			// Sum positive force
			for (unsigned int d = 0; d < D; d++) {
				double diff = Y[ind1 + d] - Y[ind2 + d];
				pos_f[ind1 + d] += mult * diff;
			}
		}
		ind1 += D;
	}

	for (int n = 0; n < N; n++) tree->computeNonEdgeForcesKLStudentHalf(n, theta, neg_f + n * D, &sum_Q);

	// Compute final t-SNE gradient
	for (int i = 0; i < N * D; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);
	delete tree;

}


void computeApproxGradientKLStudentAlpha(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double alpha, double* dC, double theta)
{
	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;
	double* pos_f = (double*)calloc(N * D, sizeof(double));
	double* neg_f = (double*)calloc(N * D, sizeof(double));
	if (pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }


	// Loop over all edges in the graph
	unsigned int ind1 = 0;
	unsigned int ind2 = 0;

	for (unsigned int n = 0; n < N; n++) {
		for (unsigned int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {

			// Compute pairwise distance and Q-value
			double dist = 0.0;
			ind2 = inp_col_P[i] * D;
			for (unsigned int d = 0; d < D; d++) {
				double diff = Y[ind1 + d] - Y[ind2 + d];
				dist += diff * diff;
			}
			double mult = inp_val_P[i] / (1 + dist/alpha);

			// Sum positive force
			for (unsigned int d = 0; d < D; d++) {
				double diff = Y[ind1 + d] - Y[ind2 + d];
				pos_f[ind1 + d] += mult * diff;
			}
		}
		ind1 += D;
	}

	for (int n = 0; n < N; n++) tree->computeNonEdgeForcesKLStudentAlpha(n, alpha, theta, neg_f + n * D, &sum_Q);

	// Compute final t-SNE gradient
	for (int i = 0; i < N * D; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);
	delete tree;

}

// Compute gradient of the t-SNE cost function (RKL - using Barnes-Hut algorithm)
static void computeApproxGradientRKL(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{

	// First, compute cost CRKL (inefficient in the current way)
	double* costs = (double*)calloc(N, sizeof(double));
	double cRKL = evaluateApproxErrorRKL(inp_row_P, inp_col_P, inp_val_P, Y, N, D, theta, costs);
	free(costs); costs = NULL;

	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;

	// depends on dimension d
	double* term_1 = (double*)calloc(N * D, sizeof(double));
	double* term_2 = (double*)calloc(N * D, sizeof(double));
	double* term_3 = (double*)calloc(N * D, sizeof(double));

	if (term_1 == NULL || term_2 == NULL || term_3 == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	for (int n = 0; n < N; n++) tree->computeNonEdgeForcesRKLGradient(n, theta, term_1 + n * D, 
																			    term_2 + n * D,
																				term_3 + n * D,
																				&sum_Q,
																			    inp_row_P, inp_col_P);

	// Compute final t-SNE gradient nonedge forces
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			dC[n * D + d] = term_1[n * D + d];
			dC[n * D + d] -= term_2[n * D + d];
			dC[n * D + d] += log(sum_Q) * term_3[n * D + d];
			dC[n * D + d] += cRKL * term_3[n * D + d];
			dC[n * D + d] /= sum_Q;
		}
	}

	// Compute final t-SNE gradient Edge forces
	double* buff = (double*)calloc(D, sizeof(double));
	int ind1, ind2;
	double C = .0, E;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {
			E = .0;
			ind2 = inp_col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) E += buff[d] * buff[d];
			E = (1.0 / (1.0 + E));

			// finally, add to gradient
			for (int d = 0; d < D; d++) {
				dC[n * D + d] += log(inp_val_P[n * D]) * E * E / sum_Q * buff[d];
			}
		}
	}

	// Cleanup memory
	free(buff);
	free(term_1); term_1 = NULL;
	free(term_2); term_2 = NULL;
	free(term_3); term_3 = NULL;
	delete tree;
}

// Compute gradient of the t-SNE cost function (JS - using Barnes-Hut algorithm)
void computeApproxGradientJS(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{

	// First, compute cost CJSQM (inefficient in the current way)
	//double* costs = (double*)calloc(N, sizeof(double));
	double cJSQM = evaluateApproxErrorJSQM(inp_row_P, inp_col_P, inp_val_P, Y, N, D, theta);
	//free(costs); costs = NULL;

	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;

	// depends on dimension d
	double* term_1 = (double*)calloc(N * D, sizeof(double));
	double* term_2 = (double*)calloc(N * D, sizeof(double));

	if (term_1 == NULL || term_2 == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	for (int n = 0; n < N; n++) tree->computeNonEdgeForcesJSGradient(n, theta, term_1 + n * D, term_2 + n * D, &sum_Q, 
																	 inp_row_P, inp_col_P);

	// Compute final t-SNE gradient nonedge forces
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			dC[n * D + d] = log(0.5) * term_1[n * D + d];
			dC[n * D + d] += cJSQM * term_2[n * D + d];
			dC[n * D + d] /= sum_Q;
		}
	}

	// Compute final t-SNE gradient Edge forces
	double* buff = (double*)calloc(D, sizeof(double));
	int ind1, ind2;
	double C = .0, E;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {
			E = .0;
			ind2 = inp_col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) E += buff[d] * buff[d];
			E = (1.0 / (1.0 + E));

			// finally, add to gradient
			for (int d = 0; d < D; d++) {
				dC[n * D + d] += (log((0.5 * inp_val_P[n * D] + 0.5 * E / sum_Q) / (E / sum_Q)) + cJSQM) * E * E / sum_Q * buff[d];
			}
		}
	}

	// Cleanup memory
	free(buff);
	free(term_1); term_1 = NULL;
	free(term_2); term_2 = NULL;
	delete tree;

}

static void approximateApproxGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D,
									 double* dC, double theta, double costFunc(unsigned int*, unsigned int*, double*, double*, int, int, double, double*))
{

	// interval range
	double h = sqrt(DBL_EPSILON);

	double* costs = (double*)calloc(N, sizeof(double));

	// Compute final t-SNE gradient
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			//dC[i] = (f(x_i + h) - (f(x_i - h)) / 2h

			// increment Y[nD + d] by h
			Y[n * D + d] += h;
			dC[n * D + d] = costFunc(inp_row_P, inp_col_P, inp_val_P, Y, N, D, theta, costs);

			// subtract Y[nD + d] with h
			Y[n * D + d] -= (2 * h);
			dC[n * D + d] -= costFunc(inp_row_P, inp_col_P, inp_val_P, Y, N, D, theta, costs);

			//correct Y[nD + d] to original value
			Y[n * D + d] += h;

			//divide by 2h to obtain final gradient
			dC[n * D + d] /= (2 * h);
		}
	}
	free(costs); costs = NULL;
}

// Compute gradient of the t-SNE cost function (KL - exact)
static void computeExactGradientKL(double* P, double* Y, int N, int D, int freeze_index, double* dC) {

	// Make sure the current gradient contains zeros
	for(int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeEuclideanDistance(Y, N, D, DD, true);

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

// Compute gradient of the t-SNE cost function (KL - ChiSq - exact)
void computeExactGradientKLChiSq(double* P, double* Y, int N, int D, double* dC)
{
	// Make sure the current gradient contains zeros
	for (int i = 0; i < N * D; i++) dC[i] = 0.0;

	// Compute the Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, false);

	// Compute Q-matrix and normalization sum
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = exp(-0.5 * DD[nN + m]);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	// Perform the computation of the gradient
	nN = 0;
	int nD = 0;
	for (int n = 0; n < N; n++) {
		int mD = 0;
		for (int m = 0; m < N; m++) {
			if (n != m) { 
				double mult = (P[nN + m] - Q[nN + m]) / DD[nN + m];
				for (int d = 0; d < D; d++) {
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
	free(Q);  Q = NULL;
}

void computeExactGradientKLStudentHalf(double* P, double* Y, int N, int D, double* dC)
{
	// Make sure the current gradient contains zeros
	for (int i = 0; i < N * D; i++) dC[i] = 0.0;

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	// Compute Q-matrix and normalization sum
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = pow(1 + 2 * DD[nN + m], -3.0/4.0);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	// Perform the computation of the gradient
	nN = 0;
	int nD = 0;
	for (int n = 0; n < N; n++) {
		int mD = 0;
		for (int m = 0; m < N; m++) {
			if (n != m) {
				double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * pow(Q[nN + m], 4.0/3.0);
				for (int d = 0; d < D; d++) {
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
	free(Q);  Q = NULL;
}

void computeExactGradientKLStudentAlpha(double* P, double* Y, int N, int D, double alpha, double* dC)
{
	// Make sure the current gradient contains zeros
	for (int i = 0; i < N * D; i++) dC[i] = 0.0;

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	// Compute Q-matrix and normalization sum
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = pow(1 + DD[nN + m] / alpha, -(1.0 + alpha) / 2.0);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	// Perform the computation of the gradient
	nN = 0;
	int nD = 0;
	for (int n = 0; n < N; n++) {
		int mD = 0;
		for (int m = 0; m < N; m++) {
			if (n != m) {
				double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * pow(Q[nN + m], 2.0 / (1.0 + alpha));
				for (int d = 0; d < D; d++) {
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
	free(Q);  Q = NULL;
}


// Compute gradient of the t-SNE cost function (RKL - exact)
void computeExactGradientRKL(double* P, double* Y, int N, int D, double* dC)
{
	// Make sure the current gradient contains zeros
	for (int i = 0; i < N * D; i++) dC[i] = 0.0;

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	// Compute Q-matrix and normalization sum
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	double cRKL = evaluateExactErrorRKL(P, N, Q);

	// Perform the computation of the gradient
	nN = 0;
	int nD = 0;
	for (int n = 0; n < N; n++) {
		int mD = 0;
		for (int m = 0; m < N; m++) {
			if (n != m) {
				double mult = (log((P[nN + m] + FLT_MIN) / (Q[nN + m] + FLT_MIN)) + cRKL) * Q[nN + m] * Q[nN + m] * sum_Q;
				for (int d = 0; d < D; d++) {
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
	free(Q);  Q = NULL;
}

// Compute gradient of the t-SNE cost function (JS - exact)
void computeExactGradientJS(double* P, double* Y, int N, int D, double* dC)
{
	// Make sure the current gradient contains zeros
	for (int i = 0; i < N * D; i++) dC[i] = 0.0;

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	// Compute Q-matrix and normalization sum
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	double cKLQM = evaluateExactErrorJSQM(P, N, Q);

	// Perform the computation of the gradient
	nN = 0;
	int nD = 0;
	for (int n = 0; n < N; n++) {
		int mD = 0;
		for (int m = 0; m < N; m++) {
			if (n != m) {
				double mult = (log((.5 * P[nN + m] + .5 * Q[nN + m] + FLT_MIN) / (Q[nN + m] + FLT_MIN)) + cKLQM) * Q[nN + m] * Q[nN + m] * sum_Q;
				for (int d = 0; d < D; d++) {
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
	free(Q);  Q = NULL;
}

void approximateExactGradient(double* P, double* Y, int N, int D, double* dC, double costFunc(double*, int, double*))
{

	// interval range
	double h = sqrt(DBL_EPSILON);

	// Make sure the current gradient contains zeros
	for (int i = 0; i < N * D; i++) dC[i] = 0.0;// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;

	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;
	// point i
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			// increment Y[nD + d] by h
			Y[nD + d] += h;

			updateEuclideanDistance(Y, N, n, D, DD, true);
			// Compute Q-matrix and normalization sum
			nN = 0;
			sum_Q = .0;
			for (int n = 0; n < N; n++) {
				for (int m = 0; m < N; m++) {
					if (n != m) {
						Q[nN + m] = 1 / (1 + DD[nN + m]);
						sum_Q += Q[nN + m];
					}
				}
				nN += N;
			}
			for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

			dC[nD + d] += costFunc(P, N, Q);
			// subtract Y[nD + d] with h
			Y[nD + d] -= (2 * h);

			updateEuclideanDistance(Y, N, n, D, DD, true);
			// Compute Q-matrix and normalization sum
			nN = 0;
			sum_Q = .0;
			for (int n = 0; n < N; n++) {
				for (int m = 0; m < N; m++) {
					if (n != m) {
						Q[nN + m] = 1 / (1 + DD[nN + m]);
						sum_Q += Q[nN + m];
					}
				}
				nN += N;
			}
			for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;
			dC[nD + d] -= costFunc(P, N, Q);

			//correct Y[nD + d] to original value
			Y[nD + d] += h;

			//divide by 2h to obtain final gradient
			dC[nD + d] /= (2 * h);
		}
		nD += D;
	}
	// Clean up memory
	free(DD);
	free(Q);

}



// Evaluate t-SNE cost function (KL - exactly)
static double evaluateExactErrorKL(double* P, double* Y, int N, int D, double* costs) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeEuclideanDistance(Y, N, D, DD, true);

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
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			//write to costs
			costs[i] += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (Q[i * N + j] + FLT_MIN));
		}
		C += costs[i];
	}

    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

static double evaluateExactErrorKL(double* P, int N, double* Q)
{
	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			C += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (Q[i * N + j] + FLT_MIN));
		}
	}
	return C;
}

// Evaluate t-SNE cost function (KL - ChiSq Q - exactly)
static double evaluateExactErrorKLChiSq(double* P, double* Y, int N, int D, double* costs)
{
	// Compute the Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, false);

	// Compute Q-matrix and normalization sum
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = exp(-0.5 * DD[nN + m]);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			//write to costs
			costs[i] += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (Q[i * N + j] + FLT_MIN));
		}
		C += costs[i];
	}

	// Clean up memory
	free(DD);
	free(Q);
	return C;
}

// Evaluate t-SNE cost function (KL - Student (0.5) Q - exactly)
double evaluateExactErrorKLStudentHalf(double* P, double* Y, int N, int D, double* costs)
{
	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	// Compute Q-matrix and normalization sum
	int nN = 0;
	double sum_Q = DBL_MIN;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = pow(1 + 2 * DD[nN + m], -3.0/4.0);
				sum_Q += Q[nN + m];
			}
			else Q[nN + m] = DBL_MIN;
		}
		nN += N;
	}
	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			//write to costs
			costs[i] += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (Q[i * N + j] + FLT_MIN));
		}
		C += costs[i];
	}

	// Clean up memory
	free(DD);
	free(Q);
	return C;
}


// Evaluate t-SNE cost function (KL - Student Alpha Q - exactly)
double evaluateExactErrorKLStudentAlpha(double* P, double* Y, int N, int D, double alpha, double* costs)
{
	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	// Compute Q-matrix and normalization sum
	int nN = 0;
	double sum_Q = DBL_MIN;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = pow(1 + DD[nN + m] / alpha, -(1.0 + alpha) / 2.0);
				sum_Q += Q[nN + m];
			}
			else Q[nN + m] = DBL_MIN;
		}
		nN += N;
	}
	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			//write to costs
			costs[i] += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (Q[i * N + j] + FLT_MIN));
		}
		C += costs[i];
	}

	// Clean up memory
	free(DD);
	free(Q);
	return C;
}


// Evaluate t-SNE cost function (RKL - exactly)
static double evaluateExactErrorRKL(double* P, double* Y, int N, int D, double* costs)
{
	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	// Compute Q-matrix and normalization sum
	int nN = 0;
	double sum_Q = DBL_MIN;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
			else Q[nN + m] = DBL_MIN;
		}
		nN += N;
	}
	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			
			//write to costs
			costs[i] += Q[i * N + j] * log((Q[i * N + j] + FLT_MIN) / (P[i * N + j] + FLT_MIN));
		}
		C += costs[i];
	}

	// Clean up memory
	free(DD);
	free(Q);
	return C;
}

// Evaluate t-SNE cost function (RKL - exactly)
// this version gets a precomputed Q array
static double evaluateExactErrorRKL(double* P, int N, double* Q)
{
	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			//write to costs
			C += Q[i * N + j] * log((Q[i * N + j] + FLT_MIN) / (P[i * N + j] + FLT_MIN));
		}
	}
	return C;
}

// Evaluate t-SNE cost function (JS - exactly)
double evaluateExactErrorJS(double* P, double* Y, int N, int D, double* costs)
{

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	// Compute Q-matrix and normalization sum
	int nN = 0;
	double sum_Q = DBL_MIN;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
			else Q[nN + m] = DBL_MIN;
		}
		nN += N;
	}
	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {

			// dropped the 2 before each cost computation as it just liearly scales the costs
			// KL(P||M)
			// write to costs
			costs[i] += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (0.5 * P[i * N + j] + 0.5 * Q[i * N + j] + FLT_MIN));

			// KL(Q||M)
			// write to costs
			costs[i] += Q[i * N + j] * log((Q[i * N + j] + FLT_MIN) / (0.5 * P[i * N + j] + 0.5 * Q[i * N + j] + FLT_MIN));
		}
		C += costs[i];
	}

	// Clean up memory
	free(DD);
	free(Q);
	return C;
}

// Evaluate t-SNE cost function (JS - exactly)
// this version gets a precomputed Q array
double evaluateExactErrorJS(double* P, int N, double* Q)
{
	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {
			// dropped the 2 before each cost computation as it just liearly scales the costs
			// KL(P||M)
			// write to costs
			C += P[i * N + j] * log((P[i * N + j] + FLT_MIN) / (0.5 * P[i * N + j] + 0.5 * Q[i * N + j] + FLT_MIN));

			// KL(Q||M)
			// write to costs
			C += Q[i * N + j] * log((Q[i * N + j] + FLT_MIN) / (0.5 * P[i * N + j] + 0.5 * Q[i * N + j] + FLT_MIN));
		}
	}
	return C;
}

// Evaluate t-SNE cost function (JS - exactly)
// this version is a helper to compute KL(Q||M)
double evaluateExactErrorJSQM(double* P, int N, double* Q)
{
	// Sum t-SNE error
	double C = .0;
	// i = Data Point i
	for (int i = 0; i < N; i++) {
		// j = neighboring point index
		for (int j = 0; j < N; j++) {

			// dropped the 2 before each cost computation as it just liearly scales the costs
			// KL(Q||M)
			C += Q[i * N + j] * log((Q[i * N + j] + FLT_MIN) / (0.5 * P[i * N + j] + 0.5 * Q[i * N + j] + FLT_MIN));
		}
	}

	// Clean up memory
	return C;
}

// Evaluate t-SNE cost function (KL - approximately)
static double evaluateApproxErrorKL(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs)
{

	// KL = P * log (P / Q)

    // Get estimate of normalization term
    SPTree* tree = new SPTree(D, Y, N);
    double* buff = (double*) calloc(D, sizeof(double));
    double sum_Q = .0;
    for(int n = 0; n < N; n++) tree->computeNonEdgeForcesKL(n, theta, buff, &sum_Q);

    // Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			costs[n] += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
		C += costs[n];
	}

	// Clean up memory
    free(buff);
    delete tree;
    return C;
}

// Evaluate t-SNE cost function (KL- ChiSq - approximately)
double evaluateApproxErrorKLChiSq(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs)
{

	// KL = P * log (P / Q)

	// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double* buff = (double*)calloc(D, sizeof(double));
	double sum_Q = .0;
	for (int n = 0; n < N; n++) tree->computeNonEdgeForcesKLChiSq(n, theta, buff, &sum_Q);

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = exp(-0.5 * sqrt(D)) / sum_Q;
			costs[n] += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
		C += costs[n];
	}

	// Clean up memory
	free(buff);
	delete tree;
	return C;
}

// Evaluate t-SNE cost function (KL- StudentHalf - approximately)
double evaluateApproxErrorKLStudentHalf(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs)
{
	// KL = P * log (P / Q)

// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double* buff = (double*)calloc(D, sizeof(double));
	double sum_Q = .0;
	for (int n = 0; n < N; n++) tree->computeNonEdgeForcesKLStudentHalf(n, theta, buff, &sum_Q);

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = pow(1 + 2 * D, -3.0 / 4.0) / sum_Q;
			costs[n] += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
		C += costs[n];
	}

	// Clean up memory
	free(buff);
	delete tree;
	return C;
}

// Evaluate t-SNE cost function (KL- StudentHalf - approximately)
double evaluateApproxErrorKLStudentAlpha(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double alpha, double theta, double* costs)
{
	// KL = P * log (P / Q)

// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double* buff = (double*)calloc(D, sizeof(double));
	double sum_Q = .0;
	for (int n = 0; n < N; n++) tree->computeNonEdgeForcesKLStudentAlpha(n, alpha, theta, buff, &sum_Q);

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = pow(1 + D / alpha, -(1.0 + alpha) / 2) / sum_Q;
			costs[n] += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
		C += costs[n];
	}

	// Clean up memory
	free(buff);
	delete tree;
	return C;
}

// Evaluate t-SNE cost function (RKL - approximately)
static double evaluateApproxErrorRKL(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs)
{

	// RKL = Q * log (Q / P)

	// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double sum_Q = .0;
	double* term_1 = (double*)calloc(N, sizeof(double));
	double* term_2 = (double*)calloc(N, sizeof(double));
	double* term_3 = (double*)calloc(N, sizeof(double));
	if (term_1 == NULL || term_2 == NULL || term_3 == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	// Part 1: loop over all non-edges to compute term_1 and part of term_2
	for (int n = 0; n < N; n++) {
		tree->computeNonEdgeForcesRKL(n, theta, term_1 + n, term_2 + n, term_3 + n, &sum_Q, row_P, col_P);
	}

	for (int n = 0; n < N; n++) {
		costs[n] += term_1[n];
		costs[n] -= term_2[n] * log(sum_Q);
		// multiply with approx. log(0)
		costs[n] -= term_3[n] * log(FLT_MIN);
		costs[n] /= sum_Q;
	}

	// Part 2: compute term 3 for available val_P by looping over all edges
	double* buff = (double*)calloc(D, sizeof(double));
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D; 
		// variable to store cost of point n
		double Cn = .0;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			costs[n] -= Q * log(val_P[i]);
		}
		C += costs[n];
	}

	// Clean up memory
	free(term_1); term_1 = NULL;
	free(term_2); term_2 = NULL;
	free(term_3); term_3 = NULL;
	delete tree;
	return C;
}

// Evaluate t-SNE cost function (JS - approximately)
double evaluateApproxErrorJS(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs)
{
	// JS = P * log (P / (.5P + .5Q)) +  Q * log (Q / (.5P + .5Q))
	

	// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double sum_Q = .0;
	double* term_1 = (double*)calloc(N, sizeof(double));

	if (term_1 == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	// Part 1: loop over all non-edges to compute term_1 
	for (int n = 0; n < N; n++) {
		tree->computeNonEdgeForcesJS(n, theta, term_1 + n, &sum_Q, row_P, col_P);
	}

	for (int n = 0; n < N; n++) {
		//term_1[n] *= log(2) / sum_Q;
		costs[n] += term_1[n] * log(2) / sum_Q;
	}

	// Part 2: compute all edge forces
	double* buff = (double*)calloc(D, sizeof(double));
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		// variable to store cost of point n
		double Cn = .0;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			costs[n] += val_P[i] * log(val_P[i] / (0.5 * val_P[i] + 0.5 * Q)) + Q * log(Q / (0.5 * val_P[i] + 0.5 * Q));
		}
		C += costs[n];
	}

	// Clean up memory
	free(term_1); term_1 = NULL;
	delete tree;
	return C;
}

double evaluateApproxErrorJSQM(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)
{
	// JSQM = Q * log (Q / (.5P + .5Q))

	// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double sum_Q = .0;
	double C = .0;
	double* term_1 = (double*)calloc(N, sizeof(double));

	if (term_1 == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	// Part 1: loop over all non-edges to compute term_1 
	for (int n = 0; n < N; n++) {
		tree->computeNonEdgeForcesJS(n, theta, term_1 + n, &sum_Q, row_P, col_P);
	}

	for (int n = 0; n < N; n++) {
		//term_1[n] *= log(2) / sum_Q;
		C += term_1[n] * log(2) / sum_Q;
	}

	// Part 2: compute all edge forces
	double* buff = (double*)calloc(D, sizeof(double));
	int ind1, ind2;
	double  Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			C += Q * log(Q / (0.5 * val_P[i] + 0.5 * Q));
		}
		//C += costs[n];
	}

	// Clean up memory
	free(term_1); term_1 = NULL;
	delete tree;
	return C;
}


// Compute gaussian input similarities with a fixed perplexity
static void computeExactGaussianInputSimilarity(double* X, int N, int D, double* P, double perplexity) {

	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(X, N, D, DD, true);

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

void computeExactLaplacianInputSimilarity(double* X, int N, int D, double* P, double perplexity)
{
	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(X, N, D, DD, false);

	// Compute the Laplacian kernel row by row
	int nN = 0;
	for (int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX;
		double tol = 1e-5;
		double sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		while (!found && iter < 200) {

			// Compute Laplacian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			double H = 0.0;
			for (int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
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

		// Row normalize P
		for (int m = 0; m < N; m++) P[nN + m] /= sum_P;
		nN += N;
	}

	// Clean up memory
	free(DD); DD = NULL;
}

void computeExactStudentInputSimilarity(double* X, int no_dims, int N, int D, double* P)
{

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(X, N, D, DD, true);

	// Compute the Student kernel row by row
	int nN = 0;
	for (int n = 0; n < N; n++) {

		// Compute Student kernel row
		for (int m = 0; m < N; m++) P[nN + m] = pow(1 + DD[nN + m] / no_dims, -(no_dims + 1) / 2.0);
		P[nN + n] = DBL_MIN;

		double sum_P = DBL_MIN;
		for (int m = 0; m < N; m++) sum_P += P[nN + m];

		// Row normalize P
		for (int m = 0; m < N; m++) P[nN + m] /= sum_P;
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
		for (int m = 0; m < K; m++) cur_P[m] = pow(1 + distances[m + 1] / no_dims, -(no_dims + 1) / 2.0);

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
/*
void computeExactGeneticOptimization(double* P, double* Y, double* Y_genomes, int N, int D, double costFunc(double*, int, double*))
{
	unsigned int pop_size = 10;
	unsigned int num_offspring = 10;
	//std::vector<double> offspring;
	std::vector<double> fitness;
	std::vector<double> fitness_inverted;

	double x_min = Y[0];
	double x_max = Y[0];
	double y_min = Y[1];
	double y_max = Y[1];

	for (int n = 1; n < N; n++) {
		for (int d = 0; d < D; d++) {
			if (d == 0) {
				if (x_min > Y[n * D + d]) x_min = Y[n * D + d];
				if (x_max < Y[n * D + d]) x_max = Y[n * D + d];
			}
			else {
				if (y_min > Y[n * D + d]) y_min = Y[n * D + d];
				if (y_max < Y[n * D + d]) y_max = Y[n * D + d];
			}
		}
	}

	printf("dimensions of the map:\nx_min= %f\nx_max= %f\ny_min= %f\ny_max= %f\n", x_min, x_max, y_min, y_max);

	double x_range = x_max - x_min;
	double y_range = y_max - y_min;

	double* offspring = (double*) malloc((num_offspring+pop_size) * D * sizeof(double));
	//double* fitness = (*double) calloc(popsize + num_offspring, sizeof(double));

	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeEuclideanDistance(Y, N, D, DD, true);

	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;

	// optimize all given Y values
	for (int n = 0; n < N; n++) {

		// compute fitness of all population elements
		for (int i = 0; i < pop_size; i++) {
			//copy genome to current Y
			for (int d = 0; d < D; d++) {
				Y[n * D + d] = Y_genomes[n * D * pop_size + i * D + d];
			}

			updateEuclideanDistance(Y, N, n, D, DD, true);
			// Compute Q-matrix and normalization sum
			nN = 0;
			sum_Q = .0;
			for (int n = 0; n < N; n++) {
				for (int m = 0; m < N; m++) {
					if (n != m) {
						Q[nN + m] = 1 / (1 + DD[nN + m]);
						sum_Q += Q[nN + m];
					}
				}
				nN += N;
			}
			for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

			// minus to convert maximization into minimization
			double fitness_i = costFunc(P, N, Q);
			fitness.push_back(fitness_i);
			fitness_inverted.push_back(1 / fitness_i);
		}

		double total_inverted_fitness = .0;
		for (unsigned int i = 0; i < fitness.size(); ++i) {
			total_inverted_fitness += 1/fitness[i];
		}

		int parent_in_offspring_buffer = 0;
		//copy parents to offspring malloc
		for (int i = 0; i < pop_size * D; i++) {
			offspring[i] = Y_genomes[n * D * pop_size + i];
			parent_in_offspring_buffer++;
		}


		// create offspring
		for (int o = 0; o < num_offspring; o++) {
			// Roulette wheel selection of parent objects

			unsigned int parent_one = 0;
			unsigned int parent_two = 0;

			double rndNumber = rand() / (double)RAND_MAX;
			double offset = 0.0;

			// parent 1
			for (int i = 0; i < pop_size; i++) {
				offset += 1/fitness[i]/total_inverted_fitness;
				if (rndNumber < offset) {
					parent_one = i;
					break;
				}
			}

			parent_two = parent_one;
			while (parent_two == parent_one) {
				rndNumber = rand() / (double)RAND_MAX;
				offset = 0.0;

				// parent 2
				for (int i = 0; i < pop_size; i++) {
					offset += 1/fitness[i] / total_inverted_fitness;
					if (rndNumber < offset) {
						parent_two = i;
						break;
					}
				}
			}

			// Uniform Crossover
			for (int d = 0; d < D; d++) {
				offspring[parent_in_offspring_buffer + o * D + d] = Y_genomes[n * D * pop_size + parent_one * D + d];
				offspring[parent_in_offspring_buffer + o * D + d] += Y_genomes[n * D * pop_size + parent_two * D + d];
				offspring[parent_in_offspring_buffer + o * D + d] /= 2;
			}

			// gaussian mutation 

			rndNumber = rand() / (double)RAND_MAX;
			if (rndNumber < .05) {
				for (int d = 0; d < D; d++) {
					if (d == 0) {
						offspring[parent_in_offspring_buffer + o * D + d] =
							min(max(randn(offspring[parent_in_offspring_buffer + o * D + d], x_range / 10), x_min), x_max);
					}
					else {
						offspring[parent_in_offspring_buffer + o * D + d] =
							min(max(randn(offspring[parent_in_offspring_buffer + o * D + d], y_range / 10), y_min), y_max);
					}
				}
			}			
		}

		// empty fitness as we do no longer care for parents, they die
		//fitness.clear();

		// compute fitness of individuals
		for (int o = 0; o < num_offspring; o++) {
			for (int d = 0; d < D; d++) {
				Y[n * D + d] = offspring[parent_in_offspring_buffer + o * D + d];
			}

			updateEuclideanDistance(Y, N, n, D, DD, true);
			// Compute Q-matrix and normalization sum
			nN = 0;
			sum_Q = .0;
			for (int n = 0; n < N; n++) {
				for (int m = 0; m < N; m++) {
					if (n != m) {
						Q[nN + m] = 1 / (1 + DD[nN + m]);
						sum_Q += Q[nN + m];
					}
				}
				nN += N;
			}
			for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

			// for now, replace entire population by new one
			// push back fitness to fitness vector
			double fitness_i = costFunc(P, N, Q);
			fitness.push_back(fitness_i);
		}

		// override population to survivors
		// order fitness to find best offspring
		vector<pair<double, int>> fitness_ordered = sortArr(fitness, num_offspring + pop_size);

		for (int o = 0; o < pop_size; o++) {
			int survivor_index = fitness_ordered[o].second;
			for (int d = 0; d < D; d++) {
				Y_genomes[n * D * pop_size + o * D + d] = offspring[survivor_index * D + d];
			}
		}

		// recover best survivor to Y

		int best_index = fitness_ordered[0].second;

		for (int d = 0; d < D; d++) {
			Y[n * D + d] = offspring[best_index * D + d];
		}




	}


	// Clean up memory
	free(DD);
	free(Q);
	free(offspring);
	//free(fitness);
}
*/
void computeExactGeneticOptimization(double* P, double* Y, int N, int iter, int D, double costFunc(double*, double*, int, int, double*))
{
	// offspring will contain parents as well
	double* offspring = (double*)malloc(2 * N * N * D * sizeof(double));
	if (offspring == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	std::vector<double> fitness_vector;
	std::vector<double> fitness_inverted;

	//copy parents to offspring malloc
	for (int i = 0; i < N * D * N; i++) {
		offspring[i] = Y[i];
	}

	double fitness = .0;

	double* costs = (double*)calloc(N, sizeof(double));
	if (costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	double x_min = Y[0];
	double x_max = Y[0];
	double y_min = Y[1];
	double y_max = Y[1];

	for (int n = 1; n < N; n++) {
		for (int d = 0; d < D; d++) {
			if (d == 0) {
				if (x_min > Y[n * D + d]) x_min = Y[n * D + d];
				if (x_max < Y[n * D + d]) x_max = Y[n * D + d];
			}
			else {
				if (y_min > Y[n * D + d]) y_min = Y[n * D + d];
				if (y_max < Y[n * D + d]) y_max = Y[n * D + d];
			}
		}
	}

	//printf("dimensions of the map:\nx_min= %f\nx_max= %f\ny_min= %f\ny_max= %f\n", x_min, x_max, y_min, y_max);

	double x_range = x_max - x_min;
	double y_range = y_max - y_min;

	// compute fitness of individuals 
	for (int i = 0; i < N; i++) {
		// set costs to 0
		for (int j = 0; j < N; j++) costs[j] = 0.0;
		fitness = costFunc(P, Y + (i * D * N), N, D, costs);
		fitness_vector.push_back(fitness);
	}

	// compute total inverted fitness
	double total_inverted_fitness = .0;
	for (unsigned int i = 0; i < fitness_vector.size(); ++i) {
		total_inverted_fitness += 1 / fitness_vector[i];
	}

	// create offspring
	for (int o = 0; o < N; o++) {
		// roulette wheel selection of parents
		unsigned int parent_one = 0;
		unsigned int parent_two = 0;

		double rndNumber = rand() / (double)RAND_MAX;
		double offset = 0.0;

		// parent 1
		for (int i = 0; i < N; i++) {
			offset += 1 / fitness_vector[i] / total_inverted_fitness;
			if (rndNumber < offset) {
				parent_one = i;
				break;
			}
		}

		parent_two = parent_one;
		while (parent_two == parent_one) {
			rndNumber = rand() / (double)RAND_MAX;
			offset = 0.0;

			// parent 2
			for (int i = 0; i < N; i++) {
				offset += 1 / fitness_vector[i] / total_inverted_fitness;
				if (rndNumber < offset) {
					parent_two = i;
					break;
				}
			}
		}

		// average crossover
		/*
		for (int i = 0; i < D * N; i++) {
			offspring[N * D * N + o * D * N + i] = offspring[parent_one * D * N + i];
			offspring[N * D * N + o * D * N + i] += offspring[parent_two * D * N + i];
			offspring[N * D * N + o * D * N + i] /= 2;
		}
		*/
		
		// random uniform crossover
		for (int i = 0; i < D * N; i++) {
			rndNumber = rand() / (double)RAND_MAX;
			if (rndNumber > .5) {
				offspring[N * D * N + o * D * N + i] = offspring[parent_one * D * N + i];
			}
			else {
				offspring[N * D * N + o * D * N + i] = offspring[parent_two * D * N + i];
			}
		}

		// gaussian mutation 

		double scale = 2.0 + iter / N * 8;

		for (int i = 0; i < D * N; i++) {
			rndNumber = rand() / (double)RAND_MAX;
			if (rndNumber < .05) {
				for (int d = 0; d < D; d++) {
					if (d == 0) {
						offspring[N * D * N + o * D * N + i] =
							//min(max(randn(offspring[N * D * N + o * D * N + i], x_range / 10), x_min), x_max);
							randn(offspring[N * D * N + o * D * N + i], x_range / scale);
					}
					else {
						offspring[N * D * N + o * D * N + i] =
							//min(max(randn(offspring[N * D * N + o * D * N + i], y_range / 10), y_min), y_max);
							randn(offspring[N * D * N + o * D * N + i], y_range / scale);
					}
				}
			}
		}
	}
	// compute fitness of newly generated offspring
	for (int i = 0; i < N; i++) {
		// set costs to 0
		for (int j = 0; j < N; j++) costs[j] = 0.0;
		fitness = costFunc(P, offspring + (N * D * N + i * D * N), N, D, costs);
		fitness_vector.push_back(fitness);
	}

	// order fitness to find best offspring
	vector<pair<double, int>> fitness_ordered = sortArr(fitness_vector, 2 * N);

	// debug
	//for (int p = 0; p < fitness_ordered.size(); p++) {
	//	printf("cost of individual %i is %f\n", fitness_ordered[p].second, fitness_ordered[p].first);
	//}

	// recover best survivors to Y

	for (int i = 0; i < N; i++) {
		int offspring_index = fitness_ordered[i].second;

		for (int nd = 0; nd < N * D; nd++) {
			Y[i * N * D + nd] = offspring[offspring_index * N * D + nd];
		}
	}

	// clear memory
	free(offspring); offspring = NULL;
}


//static double evaluateApproxErrorKL(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
void computeApproxGeneticOptimization(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int iter, int D, double theta,
	double costFunc(unsigned int*, unsigned int*, double*, double*, int, int, double, double*))
{
	// offspring will contain parents as well
	double* offspring = (double*)malloc(2 * N * N * D * sizeof(double));
	if (offspring == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	std::vector<double> fitness_vector;
	std::vector<double> fitness_inverted;

	//copy parents to offspring malloc
	for (int i = 0; i < N * D * N; i++) {
		offspring[i] = Y[i];
	}

	double fitness = .0;

	double* costs = (double*)calloc(N, sizeof(double));
	if (costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	double x_min = Y[0];
	double x_max = Y[0];
	double y_min = Y[1];
	double y_max = Y[1];

	for (int n = 1; n < N; n++) {
		for (int d = 0; d < D; d++) {
			if (d == 0) {
				if (x_min > Y[n * D + d]) x_min = Y[n * D + d];
				if (x_max < Y[n * D + d]) x_max = Y[n * D + d];
			}
			else {
				if (y_min > Y[n * D + d]) y_min = Y[n * D + d];
				if (y_max < Y[n * D + d]) y_max = Y[n * D + d];
			}
		}
	}

	//printf("dimensions of the map:\nx_min= %f\nx_max= %f\ny_min= %f\ny_max= %f\n", x_min, x_max, y_min, y_max);

	double x_range = x_max - x_min;
	double y_range = y_max - y_min;

	// compute fitness of individuals 
	for (int i = 0; i < N; i++) {
		// set costs to 0
		for (int j = 0; j < N; j++) costs[j] = 0.0;
		fitness = costFunc(row_P, col_P, val_P, Y + (i * D * N), N, D, theta, costs);
		fitness_vector.push_back(fitness);
	}

	// compute total inverted fitness
	double total_inverted_fitness = .0;
	for (unsigned int i = 0; i < fitness_vector.size(); ++i) {
		total_inverted_fitness += 1 / fitness_vector[i];
	}

	// create offspring
	for (int o = 0; o < N; o++) {
		// roulette wheel selection of parents
		unsigned int parent_one = 0;
		unsigned int parent_two = 0;

		double rndNumber = rand() / (double)RAND_MAX;
		double offset = 0.0;

		// parent 1
		for (int i = 0; i < N; i++) {
			offset += 1 / fitness_vector[i] / total_inverted_fitness;
			if (rndNumber < offset) {
				parent_one = i;
				break;
			}
		}

		parent_two = parent_one;
		while (parent_two == parent_one) {
			rndNumber = rand() / (double)RAND_MAX;
			offset = 0.0;

			// parent 2
			for (int i = 0; i < N; i++) {
				offset += 1 / fitness_vector[i] / total_inverted_fitness;
				if (rndNumber < offset) {
					parent_two = i;
					break;
				}
			}
		}

		// average crossover
		/*
		for (int i = 0; i < D * N; i++) {
			offspring[N * D * N + o * D * N + i] = offspring[parent_one * D * N + i];
			offspring[N * D * N + o * D * N + i] += offspring[parent_two * D * N + i];
			offspring[N * D * N + o * D * N + i] /= 2;
		}
		*/

		// random uniform crossover
		for (int i = 0; i < D * N; i++) {
			rndNumber = rand() / (double)RAND_MAX;
			if (rndNumber > .5) {
				offspring[N * D * N + o * D * N + i] = offspring[parent_one * D * N + i];
			}
			else {
				offspring[N * D * N + o * D * N + i] = offspring[parent_two * D * N + i];
			}
		}

		// gaussian mutation 

		for (int i = 0; i < D * N; i++) {
			rndNumber = rand() / (double)RAND_MAX;
			if (rndNumber < .05) {
				for (int d = 0; d < D; d++) {
					if (d == 0) {
						offspring[N * D * N + o * D * N + i] =
							//min(max(randn(offspring[N * D * N + o * D * N + i], x_range / 10), x_min), x_max);
							randn(offspring[N * D * N + o * D * N + i], x_range / 10);
					}
					else {
						offspring[N * D * N + o * D * N + i] =
							//min(max(randn(offspring[N * D * N + o * D * N + i], y_range / 10), y_min), y_max);
							randn(offspring[N * D * N + o * D * N + i], y_range / 10);
					}
				}
			}
		}
	}
	// compute fitness of newly generated offspring
	for (int i = 0; i < N; i++) {
		// set costs to 0
		for (int j = 0; j < N; j++) costs[j] = 0.0;
		
		fitness = costFunc(row_P, col_P, val_P, offspring + (N * D * N + i * D * N), N, D, theta, costs);
		fitness_vector.push_back(fitness);
	}

	// order fitness to find best offspring
	vector<pair<double, int>> fitness_ordered = sortArr(fitness_vector, 2 * N);

	// debug
	//for (int p = 0; p < fitness_ordered.size(); p++) {
	//	printf("cost of individual %i is %f\n", fitness_ordered[p].second, fitness_ordered[p].first);
	//}

	// recover best survivors to Y

	for (int i = 0; i < N; i++) {
		int offspring_index = fitness_ordered[i].second;

		for (int nd = 0; nd < N * D; nd++) {
			Y[i * N * D + nd] = offspring[offspring_index * N * D + nd];
		}
	}

	// clear memory
	free(offspring); offspring = NULL;
}

static vector<pair<double, int>> sortArr(vector<double> arr, int n)
{
	// Vector to store element 
	// with respective present index 
	vector<pair<double, int>> vp;

	// Inserting element in pair vector 
	// to keep track of previous indexes 
	for (int i = 0; i < n; ++i) {
		vp.push_back(make_pair(arr[i], i));
	}

	// Sorting pair vector 
	sort(vp.begin(), vp.end());

	return vp;
}



// Compute squared Euclidean distance matrix
static void computeEuclideanDistance(double* X, int N, int D, double* DD, bool squared) {
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
			if (!squared) *curr_elem = sqrt(*curr_elem);
			*curr_elem_sym = *curr_elem;
        }
    }
}

static void updateEuclideanDistance(double* X, int N, int update_index, int D, double* DD, bool squared) {
	const double* XnD = X;
	for (int n = 0; n < N; ++n, XnD += D) {
		const double* XmD = XnD + D;
		double* curr_elem = &DD[n * N + n];
		*curr_elem = 0.0;
		double* curr_elem_sym = curr_elem + N;
		for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N) {
			if (update_index == n || update_index == m) {
				*(++curr_elem) = 0.0;
				for (int d = 0; d < D; ++d) {
					*curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
				}
				if (!squared)* curr_elem = sqrt(*curr_elem);
				*curr_elem_sym = *curr_elem;
			}
			else ++curr_elem;
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

// Generates a Gaussian random number (with mean and sd)
static double randn(double mean, double stdDev) {
	static double spare;
	static bool hasSpare = false;

	if (hasSpare) {
		hasSpare = false;
		return spare * stdDev + mean;
	}
	else {
		double u, v, s;
		do {
			u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
			v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
			s = u * u + v * v;
		} while (s >= 1.0 || s == 0.0);
		s = sqrt(-2.0 * log(s) / s);
		spare = v * s;
		hasSpare = true;
		return mean + stdDev * u * s;
	}
}



// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(double** data, int* n, int* d, double** initial_solution, int* no_dims, double* theta, double* perplexity, double* eta, double* momentum, double* final_momentum,
					 int* rand_seed, int* max_iter, int* stop_lying_iter, int* restart_lying_iter, int* momentum_switch_iter, int* lying_factor, bool* skip_random_init,
					 int* input_similarities, int* output_similarities, int* cost_function, int* optimization, int* freeze_index) {
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

	// Freeze index

	fread(freeze_index, sizeof(int), 1, h);									// freeze_index
	printf("freeze index = %d\n", *freeze_index);

	// Data and optional randseed/initial solution

	*data = (double*) malloc(*n * *d * sizeof(double));
    if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    fread(*data, sizeof(double), *n * *d, h);                               // the data

	fread(rand_seed, sizeof(int), 1, h);
	printf("Random seed set to %i\n", *rand_seed);							// random seed (even if no seed is passed, it is set to -1 (thanks to fread default passing amount of successfully read elements)
	
	// create initial solution malloc according to whether genetic optimizer is used or not
	if (*optimization == 1) {
		*initial_solution = (double*)malloc(*n * *no_dims * *n * sizeof(double));
	}
	else {
		*initial_solution = (double*)malloc(*n * *no_dims  * sizeof(double));
	}

	if (*initial_solution == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	fread(*initial_solution, sizeof(double), *n * *no_dims, h);				// the initial solution

	if (!feof(h)) { //this should kick in if there was no initial solution (or not even a rand-seed) provided
		// if there is an initial solution provided, set skip random init to true
		printf("Initial solution Y is provided!\n");
		*skip_random_init = true;		
	}		

	if (*freeze_index != 0) {
		printf("Initial solution Y is provided for train dataset!\n");
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
