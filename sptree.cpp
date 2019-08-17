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

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include "sptree.h"

// Default constructor for SPTree -- build tree, too!
SPTree::SPTree(unsigned int D, double* inp_data, unsigned int N)
	: dimension(D),
	data(inp_data)
{
	// Compute mean, width, and height of current map (boundaries of SPTree)
	double* point = data;
	std::vector<double> mean_Y(D, 0.0);
	std::vector<double> min_Y(D, DBL_MAX);
	std::vector<double> max_Y(D, -DBL_MAX);
	for (unsigned int n = 0; n < N; n++) {
		for (unsigned int d = 0; d < D; d++) {
			double value = point[d];
			mean_Y[d] += value;
			min_Y[d] = std::min(min_Y[d], value);
			max_Y[d] = std::max(max_Y[d], value);
		}
		point += D;
	}
	for (int d = 0; d < D; d++) mean_Y[d] /= N;

	// Construct the tree
	std::vector<double> width(D);
	max_width = 0.0;
	for (int d = 0; d < D; d++) {
		width[d] = std::max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
		max_width = std::max(max_width, width[d]);
	}
	widths.push_back(std::move(width));

	point = data;
	// root is always the first in Y, that is point index = 0
	root = new_node(point, 0, std::move(mean_Y), widths[0].data());

	for (unsigned int i = 1; i < N; i++) {
		point += D;
		insert(root, point, i);
	}

	for (Node& node : nodes) {
		double scale = 1.0 / node.size;
		for (unsigned int d = 0; d < D; ++d) {
			node.center_of_mass[d] *= scale;
		}
	}
}

// Destructor for SPTree
SPTree::~SPTree() = default;

// Create a new leaf node
SPTree::Node* SPTree::new_node(const double* point, unsigned int point_index, std::vector<double> center, const double* width)
{
	nodes.emplace_back();
	Node* node = &nodes.back();
	node->point = point;
	node->size = 1;
	node->indices.push_back(point_index);
	node->center = std::move(center);
	node->width = width;
	node->center_of_mass.assign(point, point + dimension);
	return node;
}

// Insert a point into the SPTree
void SPTree::insert(Node* node, const double* point, unsigned int point_index)
{
	unsigned int depth = 0;

	while (node->point != point) {
		++node->size;
		node->indices.push_back(point_index);

		for (unsigned int d = 0; d < dimension; ++d) {
			node->center_of_mass[d] += point[d];
		}

		++depth;

		if (node->point) {
			// If this is a leaf note, split it into an internal node
			// pass the point index of leaf node to to-be child nodes
			insertChild(node, node->point, node->indices[0], depth);
			node->point = nullptr;
		}

		node = insertChild(node, point, point_index, depth);
	}
}

// Find the right child node for a point, creating it if necessary
SPTree::Node* SPTree::insertChild(Node* node, const double* point, unsigned int point_index, unsigned int depth) {
	// Find which child to insert into
	unsigned int i = 0;
	for (unsigned int d = 0; d < dimension; ++d) {
		if (point[d] > node->center[d]) {
			i |= 1 << d;
		}
	}

	if (i >= node->children.size()) {
		node->children.resize(1 << dimension, nullptr);
	}

	Node* child = node->children[i];

	if (!child) {
		if (depth >= widths.size()) {
			std::vector<double> width(dimension);
			for (unsigned int d = 0; d < dimension; ++d) {
				width[d] = 0.5 * node->width[d];
			}
			widths.push_back(std::move(width));
		}
		const double* width = widths[depth].data();

		std::vector<double> center(dimension);
		for (unsigned int d = 0; d < dimension; ++d) {
			if (i & (1 << d)) {
				center[d] = node->center[d] + width[d];
			}
			else {
				center[d] = node->center[d] - width[d];
			}
		}

		child = new_node(point, point_index, std::move(center), width);
		node->children[i] = child;
	}

	return child;
}

// Compute non-edge forces using Barnes-Hut algorithm
void SPTree::computeNonEdgeForcesKL(unsigned int point_index, double theta, double neg_f[], double* sum_Q)
{
	double* point = data + point_index * dimension;
	computeNonEdgeForcesKL(root, max_width * max_width, point, theta * theta, neg_f, sum_Q);
}

// Compute non-edge forces using Barnes-Hut algorithm (ChiSq Output Similarities)
void SPTree::computeNonEdgeForcesKLChiSq(unsigned int point_index, double theta, double neg_f[], double* sum_Q)
{
	double* point = data + point_index * dimension;
	computeNonEdgeForcesKLChiSq(root, max_width * max_width, point, theta * theta, neg_f, sum_Q);
}

// Compute non-edge forces using Barnes-Hut algorithm (Student0.5 Output Similarities)
void SPTree::computeNonEdgeForcesKLStudentHalf(unsigned int point_index, double theta, double neg_f[], double* sum_Q)
{
	double* point = data + point_index * dimension;
	computeNonEdgeForcesKLStudentHalf(root, max_width * max_width, point, theta * theta, neg_f, sum_Q);
}

void SPTree::computeNonEdgeForcesRKL(unsigned int point_index, double theta, double* term_1, double* term_2, double* term_3, double* sum_Q,
									 unsigned int* row_P, unsigned int* col_P) //row_p and col_p for blacklisted values
{
	double* point = data + point_index * dimension;
	// point index still required to determine blacklisted values
	computeNonEdgeForcesRKL(root, max_width * max_width, point, point_index, theta * theta, term_1, term_2, term_3, sum_Q, row_P, col_P);
}

void SPTree::computeNonEdgeForcesRKLGradient(unsigned int point_index, double theta, double* term_1, double* term_2, double* term_3, double* sum_Q, unsigned int* row_P, unsigned int* col_P)
{
	double* point = data + point_index * dimension;
	// point index still required to determine blacklisted values
	computeNonEdgeForcesRKLGradient(root, max_width * max_width, point, point_index, theta * theta, 
								    term_1, term_2, term_3, sum_Q, row_P, col_P);
}

void SPTree::computeNonEdgeForcesJS(unsigned int point_index, double theta, double* term_1, double* sum_Q, unsigned int* row_P, unsigned int* col_P)
{
	double* point = data + point_index * dimension;
	// point index still required to determine blacklisted values
	computeNonEdgeForcesJS(root, max_width * max_width, point, point_index, theta * theta, term_1, sum_Q, row_P, col_P);

}

void SPTree::computeNonEdgeForcesJSGradient(unsigned int point_index, double theta, double* term_1, double* term_2, double* sum_Q, unsigned int* row_P, unsigned int* col_P)
{
	double* point = data + point_index * dimension;
	// point index still required to determine blacklisted values
	computeNonEdgeForcesJSGradient(root, max_width * max_width, point, point_index, theta * theta,
								   term_1, term_2, sum_Q, row_P, col_P);
}

// Compute non-edge forces using Barnes-Hut algorithm
void SPTree::computeNonEdgeForcesKL(Node* node, double max_width_sq, double* point, double theta_sq, double neg_f[], double* sum_Q)
{
	// Make sure that we spend no time on self-interactions
	if (node->point == point) return;

	// Compute distance between point and center-of-mass
	double D = 0.0;
	for (unsigned int d = 0; d < dimension; d++) {
		double diff = point[d] - node->center_of_mass[d];
		D += diff * diff; // || y_i - y_j ||^2
	}

	// Optimize (max_width / sqrt(D) < theta) by squaring and multiplying through by D
	if (node->point || max_width_sq < theta_sq * D) {
		// Compute and add t-SNE force between point and current node
		D = 1.0 / (1.0 + D); // || E_ij^-1
		double mult = node->size * D; // || node_size * E_ij^-1
		*sum_Q += mult; // add to Z
		mult *= D; // E_ij^2 --> q_ij^2 * Z^2 (note the actual term is q_ij^2 * Z!)
		for (unsigned int d = 0; d < dimension; d++) { // split computation of SUM(q_ij^2 * Z * (y_i - y_j)) dimension-wise
			double diff = point[d] - node->center_of_mass[d];
			neg_f[d] += mult * diff;
		}
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (Node* child : node->children) {
			if (child) {
				computeNonEdgeForcesKL(child, max_width_sq / 4.0, point, theta_sq, neg_f, sum_Q);
			}
		}
	}
}

void SPTree::computeNonEdgeForcesKLChiSq(Node* node, double max_width_sq, double* point, double theta_sq, double neg_f[], double* sum_Q)
{
	// Make sure that we spend no time on self-interactions
	if (node->point == point) return;

	// Compute distance between point and center-of-mass
	double D = 0.0;
	for (unsigned int d = 0; d < dimension; d++) {
		double diff = point[d] - node->center_of_mass[d];
		D += diff * diff; // || y_i - y_j ||^2
	}

	// Optimize (max_width / sqrt(D) < theta) by squaring and multiplying through by D
	if (node->point || max_width_sq < theta_sq * D) {
		// Compute and add t-SNE force between point and current node
		double E = exp(-0.5 * sqrt(D)); // || E_ij
		double mult = node->size * E; // || node_size * E_ij^-1
		*sum_Q += mult; // add to Z
		mult /= sqrt(D); // E_ij * 1/d_ij 
		for (unsigned int d = 0; d < dimension; d++) { // split computation of SUM(q_ij * 1/d_ij * (y_i - y_j)) dimension-wise
			double diff = point[d] - node->center_of_mass[d];
			neg_f[d] += mult * diff;
		}
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (Node* child : node->children) {
			if (child) {
				computeNonEdgeForcesKLChiSq(child, max_width_sq / 4.0, point, theta_sq, neg_f, sum_Q);
			}
		}
	}
}

void SPTree::computeNonEdgeForcesKLStudentHalf(Node* node, double max_width_sq, double* point, double theta_sq, double neg_f[], double* sum_Q)
{
	// Make sure that we spend no time on self-interactions
	if (node->point == point) return;

	// Compute distance between point and center-of-mass
	double D = 0.0;
	for (unsigned int d = 0; d < dimension; d++) {
		double diff = point[d] - node->center_of_mass[d];
		D += diff * diff; // || y_i - y_j ||^2
	}

	// Optimize (max_width / sqrt(D) < theta) by squaring and multiplying through by D
	if (node->point || max_width_sq < theta_sq * D) {
		// Compute and add t-SNE force between point and current node
		double E = pow(1 + 2 * D, -3.0 / 4.0); // || E_ij
		double mult = node->size * E; // || node_size * E_ij
		*sum_Q += mult; // add to Z
		mult *= 1 / (1 + 2 * D); // E_ij * e_ij^(4/3) 
		for (unsigned int d = 0; d < dimension; d++) { // split computation of SUM(e_ij * e_ij^(4/3) * (y_i - y_j)) dimension-wise
			double diff = point[d] - node->center_of_mass[d];
			neg_f[d] += mult * diff;
		}
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (Node* child : node->children) {
			if (child) {
				computeNonEdgeForcesKLStudentHalf(child, max_width_sq / 4.0, point, theta_sq, neg_f, sum_Q);
			}
		}
	}
}

// Compute non-edge forces using Barnes-Hut algorithm for RKL objective
void SPTree::computeNonEdgeForcesRKL(Node* node, double max_width_sq, double* point, unsigned int point_index, double theta_sq, 
									 double* term_1, double* term_2, double* term_3, double* sum_Q,
									 unsigned int* row_P, unsigned int* col_P) //row_p and col_p for blacklisted values)
{
	// Make sure that we spend no time on self-interactions
	if (node->point == point) return;

	// Compute distance between point and center-of-mass
	double D = 0.0;
	for (unsigned int d = 0; d < dimension; d++) {
		double diff = point[d] - node->center_of_mass[d];
		D += diff * diff; // || y_i - y_j ||^2
	}

	// Optimize (max_width / sqrt(D) < theta) by squaring and multiplying through by D
	if (node->point || max_width_sq < theta_sq * D) {
		// Compute and add t-SNE force between point and current node

		unsigned int blacklist_count = 0;
		for (unsigned int i = row_P[point_index]; i < row_P[point_index + 1]; i++) {
			for (unsigned int j = 0; j < node->indices.size(); j++) {
				if (col_P[i] == node->indices[j]) blacklist_count++;
			}
		}

		// non_blacklisted sum
		D = 1.0 / (1.0 + D); // || E_ij^-1
		double mult = node->size * D; // || node_size * E_ij^-1
		*sum_Q += mult; // add to Z

		// blacklisted sum
		double mult_blacklisted = (node->size - blacklist_count) * D; // || node_size - blacklist_count * E_ij^-1
		
		*term_1 += mult * log(mult); // add to sum_j e_ij * log e_ij

		*term_2 += mult; // add to sum_j e_ij 

		*term_3 += mult_blacklisted; // add to sum_j e_ij (accounting for blacklisted values of j)
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (Node* child : node->children) {
			if (child) {
				computeNonEdgeForcesRKL(child, max_width_sq / 4.0, point, point_index, theta_sq, term_1, term_2, term_3, sum_Q, row_P, col_P);
			}
		}
	}
}

void SPTree::computeNonEdgeForcesRKLGradient(Node* node, double max_width_sq, double* point, unsigned int point_index, double theta_sq, double* term_1, double* term_2, double* term_3, double* sum_Q, unsigned int* row_P, unsigned int* col_P)
{
	// Make sure that we spend no time on self-interactions
	if (node->point == point) return;

	// Compute distance between point and center-of-mass
	double D = 0.0;
	for (unsigned int d = 0; d < dimension; d++) {
		double diff = point[d] - node->center_of_mass[d];
		D += diff * diff; // || y_i - y_j ||^2
	}

	// Optimize (max_width / sqrt(D) < theta) by squaring and multiplying through by D
	if (node->point || max_width_sq < theta_sq * D) {
		// Compute and add t-SNE force between point and current node
		D = 1.0 / (1.0 + D); // || E_ij^-1
		// compute sum_Q
		*sum_Q += node->size * D; // node_size * E_ij^-1 add to Z
		
		unsigned int blacklist_count = 0;
		for (unsigned int i = row_P[point_index]; i < row_P[point_index + 1]; i++) {
			for (unsigned int j = 0; j < node->indices.size(); j++) {
				if (col_P[i] == node->indices[j]) blacklist_count++;
			}
		}

		//compute dimension-wise terms (i.e. all terms including (y_i-y_j))

		for (unsigned int d = 0; d < dimension; d++) { 

			double diff = point[d] - node->center_of_mass[d];

			term_1[d] += (node->size - blacklist_count) * (log(FLT_MIN) * D * D * diff);
			term_2[d] += (node->size) * log(D) * D * D * diff;
			term_3[d] += (node->size) * D * D * diff; 
		}
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (Node* child : node->children) {
			if (child) {
				computeNonEdgeForcesRKLGradient(child, max_width_sq / 4.0, point, point_index, theta_sq, term_1, term_2, term_3, sum_Q, row_P, col_P);
			}
		}
	}
}

void SPTree::computeNonEdgeForcesJS(Node* node, double max_width_sq, double* point, unsigned int point_index, double theta_sq, double* term_1, double* sum_Q, unsigned int* row_P, unsigned int* col_P)
{
	// Make sure that we spend no time on self-interactions
	if (node->point == point) return;

	// Compute distance between point and center-of-mass
	double D = 0.0;
	for (unsigned int d = 0; d < dimension; d++) {
		double diff = point[d] - node->center_of_mass[d];
		D += diff * diff; // || y_i - y_j ||^2
	}

	// Optimize (max_width / sqrt(D) < theta) by squaring and multiplying through by D
	if (node->point || max_width_sq < theta_sq * D) {
		// Compute and add t-SNE force between point and current node

		unsigned int blacklist_count = 0;
		for (unsigned int i = row_P[point_index]; i < row_P[point_index + 1]; i++) {
			for (unsigned int j = 0; j < node->indices.size(); j++) {
				if (col_P[i] == node->indices[j]) blacklist_count++;
			}
		}

		// non_blacklisted sum
		D = 1.0 / (1.0 + D); // || E_ij^-1
		double mult = node->size * D; // || node_size * E_ij^-1
		*sum_Q += mult; // add to Z

		// blacklisted sum
		double mult_blacklisted = (node->size - blacklist_count) * D; // || node_size - blacklist_count * E_ij^-1

		*term_1 += mult_blacklisted; // add to sum_j e_ij
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (Node* child : node->children) {
			if (child) {
				computeNonEdgeForcesJS(child, max_width_sq / 4.0, point, point_index, theta_sq, term_1, sum_Q, row_P, col_P);
			}
		}
	}
}

void SPTree::computeNonEdgeForcesJSGradient(Node* node, double max_width_sq, double* point, unsigned int point_index, double theta_sq, double* term_1, double* term_2, double* sum_Q, unsigned int* row_P, unsigned int* col_P)
{
	// Make sure that we spend no time on self-interactions
	if (node->point == point) return;

	// Compute distance between point and center-of-mass
	double D = 0.0;
	for (unsigned int d = 0; d < dimension; d++) {
		double diff = point[d] - node->center_of_mass[d];
		D += diff * diff; // || y_i - y_j ||^2
	}

	// Optimize (max_width / sqrt(D) < theta) by squaring and multiplying through by D
	if (node->point || max_width_sq < theta_sq * D) {
		// Compute and add t-SNE force between point and current node
		D = 1.0 / (1.0 + D); // || E_ij^-1
		// compute sum_Q
		*sum_Q += node->size * D; // node_size * E_ij^-1 add to Z

		unsigned int blacklist_count = 0;
		for (unsigned int i = row_P[point_index]; i < row_P[point_index + 1]; i++) {
			for (unsigned int j = 0; j < node->indices.size(); j++) {
				if (col_P[i] == node->indices[j]) blacklist_count++;
			}
		}

		//compute dimension-wise terms (i.e. all terms including (y_i-y_j))

		for (unsigned int d = 0; d < dimension; d++) {

			double diff = point[d] - node->center_of_mass[d];

			term_1[d] += (node->size - blacklist_count) * D * D * diff;
			term_2[d] += (node->size) * D * D * diff;
		}
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (Node* child : node->children) {
			if (child) {
				computeNonEdgeForcesJSGradient(child, max_width_sq / 4.0, point, point_index, theta_sq, term_1, term_2, sum_Q, row_P, col_P);
			}
		}
	}
}

// Computes edge forces
void SPTree::computeEdgeForcesKL(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f)
{

	// Loop over all edges in the graph
	unsigned int ind1 = 0;
	unsigned int ind2 = 0;
	double D;
	for (unsigned int n = 0; n < N; n++) {
		for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {

			// Compute pairwise distance and Q-value
			D = 1.0;
			ind2 = col_P[i] * dimension;
			for (unsigned int d = 0; d < dimension; d++) {
				double diff = data[ind1 + d] - data[ind2 + d];
				D += diff * diff;
			}
			D = val_P[i] / D;

			// Sum positive force
			for (unsigned int d = 0; d < dimension; d++) {
				double diff = data[ind1 + d] - data[ind2 + d];
				pos_f[ind1 + d] += D * diff;
			}
		}
		ind1 += dimension;
	}
}

// Print out the tree
void SPTree::print()
{
	print(root);
}

void SPTree::print(Node* node)
{
	if (node->point) {
		printf("Leaf node; data = [");
		for (int d = 0; d < dimension; d++) {
			if (d > 0) {
				printf(", ");
			}
			printf("%f", node->point[d]);
		}
		printf("]\n");
	}
	else {
		printf("Intersection node with center-of-mass = [");
		for (int d = 0; d < dimension; d++) {
			if (d > 0) {
				printf(", ");
			}
			printf("%f", node->center_of_mass[d]);
		}
		printf("]; children are:\n");
		for (Node* child : node->children) {
			if (child) {
				print(child);
			}
		}
	}
}