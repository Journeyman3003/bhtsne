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


#ifndef TSNE_H
#define TSNE_H

#ifdef __cplusplus
extern "C" {
namespace TSNE {
#endif
    void run(double* X, int N, int D, double* Y, double* costs, int* landmarks, int no_dims, double perplexity, double eta,
		     double momentum, double final_momentum, double theta, int rand_seed, bool skip_random_init, int max_iter, int lying_factor,
			 int stop_lying_iter, int restart_lying_iter, int mom_switch_iter, int input_similarities, int output_similarities, 
		     int cost_function, int optimization, int freeze_index);
	bool load_data(double** data, int* n, int* d, double** initial_solution, int* no_dims, double* theta, double* perplexity, double* eta, double* momentum, double* final_momentum,
		           int* rand_seed, int* max_iter, int* stop_lying_iter, int* restart_lying_iter, int* momentum_switch_iter, int* lying_factor, bool* skip_random_init,
				   int* input_similarities, int* output_similarities, int* cost_function, int* optimization, int* freeze_index);
	void save_data(double* data, int* landmarks, double* costs, int n, int d, int iter);
#ifdef __cplusplus
};
}
#endif

#endif
