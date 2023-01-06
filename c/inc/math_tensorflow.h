#ifndef _MATH_TENSORFLOW_H_
#define _MATH_TENSORFLOW_H_

#include <math.h>
#include <stdlib.h> //For rand()
//
void vecadd(int r, float *A, float *B, float *result);
void vecsub(int r, float *A, float *B, float *result);
void vecmul(int r, float *A, float *B, float *result);
void addvecsub(int r, float *A, float *B, float *result);

// Add the matrix "A" with the matrix "B"
void matadd(int r, int c, float A[][c], float B[][c], float result[][c]);

// Subtract the matrix "A" from the matrix "B"
void matsub(int r, int c, float A[][c], float B[][c], float result[][c]);

void matdiv(int r, int c, float A[][c], float B[][c], float result[][c]);

//Multiply the matrix "A" by the matrix "B"
void matmul(int r1, int c1, int r2, int c2, float A[][c1], float B[][c2], float result[][c2]);
void matsqrt(int r, int c, float A[][c], float result[][c]);
void vecmatmul(int c1, int r2, int c2, float* A, float B[][c2], float* result);
void matmul_short_float(int r1, int c1, int r2, int c2, unsigned short A[][c1], float B[][c2], float result[][c2]);

//Add matrix "A" with the Scalar "K"
void scalar_matadd(int r, int c, float K, float A[][c], float result[][c]);

//Multiply matrix "A" by the Scalar "K"
void scalar_matmul(int r, int c, float K, float A[][c], float result[][c]);

//Divide matrix "A" by the Scalar "K"
void scalar_matdiv(int r, int c, float K, float A[][c], float result[][c]);

//Add vector "A" with the Scalar "K"
void scalar_vecadd(int r, float K, float *A, float *result);

//Multiply 1D vector "A" by the Scalar "K"
void scalar_vecmul(int r, float K, float *A, float *result);

//Divide 1D vector "A" by the Scalar "K"
void scalar_vecdiv(int r, float K, float *A, float *result);

//Multiply matrix "A" by the Scalar "K" and add it to matrix "A"
void scalar_matmul_add(int r, int c, float K, float A[][c], float result[][c]);
void scalar_matmul_add_doble(int r, int c, double K, float A[][c], float result[][c]);

//result = A.*B multiplies arrays A and B by multiplying corresponding elements.
void times_vec2vec(int r, float* A, float* B, float* result);

//result = A.*B multiplies 1D vector A by 2D vector B.
void times_vec2mat(int r, int c, float A[], float B[][c], float result[][c]);

//
void times_mat(int r, int c, float A[][c], float B[][c], float result[][c]);

// result[1:r, 1:c, 1:d] = A_2D[1:r, None, 1:d] * B_3D[1:r, 1:c, 1:d]
void times_mat2(int r, int c, int d, float A[][d], float B[][c][d], float result[][c][d]);


//Tensor contraction over specified indices and outer product.
void einsum_bk_jk_bj(int r1, int d1, int r2, int c2, float A[][d1], float B[][c2], float result[][r2]);
void einsum_j_jk_k(int r1, int r2, int c2, float A[], float B[][c2], float result[]);
void einsum_k_jk_j(int r1, int r2, int c2, float* A, float B[][c2], float* result);

//Computes the sum of elements across dimensions of a tensor.
void reduce_sum(int r, float *A, float result);
void reduce_sum3D(int r, int c, int d, float A[][c][d], float result[][d]);
void reduce_sum4D(int r, int c, int d, int w, float A[][c][d][w], float result[][w]);

// Computes the sum of elements across dimensions of a tensor.
void  reduce_dim(int r, int c, float A[][c], float *result);
void  reduce_dim3D(int r, int c, int d, float A[][c][d], float result[][d]);

//Computes the mean of elements across dimensions of a tensor.
//equal to tf.reduce_mean(A, axis=(0))
void reduce_mean(int r, int c, float A[][c], float *result);
//equal to tf.reduce_mean(A, axis=(0, 1))
void reduce_mean3D(int r, int c, int d, float A[][c][d], float result[][d]);

// Calculate loss function
void Floss_1D(int r, float A[], float *loss);
void Floss_2D(int r, int c, float A[][c], float *loss);
float Floss_3D(int r, int c, int d, float A[][c][d]);

void one_hot(int indices, int depth, float* result);
void softmax(int r, float* logits, float* result);
void logsoftmax(int input_len, float *input, float* result);
void sparse_softmax_cross_entropy_with_logits(int input_len, int labels, float* logits, float* loss);
void argmax(int r, float* input, int* result);
void equal(int x, int y, int* result);

void changeValues(int *a, int *b);
void rd_permutation_arr(int n, int* arr1);
void rd_permutation(int n, int* result);


#endif

