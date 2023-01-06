#include "math_tensorflow.h"

// Add the vector "A" with the vector "B"
void vecadd(int r, float *A, float *B, float *result)
{
   int i;
   for (i = 0; i < r; ++i)
      result[i] = A[i] + B[i];

}

// subtract the vector "A" with the vector "B"
void vecsub(int r, float *A, float *B, float *result)
{  
   int i;
   for (i = 0; i < r; ++i)
      result[i] = A[i] - B[i];

}

// Multiply the vector "A" with the vector "B"
void vecmul(int r, float *A, float *B, float *result)
{  
   int i;
   for (i = 0; i < r; ++i)
      result[i] = A[i] * B[i];

}

// subtract the vector "A" with the vector "B"
void addvecsub(int r, float *A, float *B, float *result)
{
   int i;
   for (i = 0; i < r; ++i)
      result[i] += A[i] - B[i];

}

// Add the matrix "A" with the matrix "B"
void matadd(int r, int c, float A[][c], float B[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) {
         result[i][j] = A[i][j] + B[i][j];
      }

}

// Subtract the matrix "A" from the matrix "B"
void matsub(int r, int c, float A[][c], float B[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) {
         result[i][j] = A[i][j] - B[i][j];
        }

}

// Divide the matrix "A" by the matrix "B"
void matdiv(int r, int c, float A[][c], float B[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) {
         result[i][j] = A[i][j] / B[i][j];
      }

}

//Multiply the matrix "A" by the matrix "B"
void matmul(int r1, int c1, int r2, int c2, float A[][c1], float B[][c2], float result[][c2])
{
   int i,j,k;
   // Initializing elements of matrix mult to 0.
   for (i = 0; i < r1; ++i) {
      for (j = 0; j < c2; ++j) {
         result[i][j] = 0;
      }
   }

   // Multiplying first and second matrices and storing it in result
   for (i = 0; i < r1; ++i) {
      for (j = 0; j < c2; ++j) {
         for (k = 0; k < c1; ++k) {
            result[i][j] += A[i][k] * B[k][j];//(double)
         }
      }
   }
}



void matsqrt(int r, int c, float A[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) {
         result[i][j] = sqrt(A[i][j]);
      }

}

//Multiply the matrix "A" by the matrix "B"
void vecmatmul(int c1, int r2, int c2, float* A, float B[][c2], float* result)
{
   int j,k;
   // Initializing elements of matrix mult to 0.
   for (j = 0; j < c2; ++j) 
   {
      result[j] = 0;
   }
   

   // Multiplying first and second matrices and storing it in result
   for (j = 0; j < c2; ++j) 
   {
      for (k = 0; k < c1; ++k) 
      {
         result[j] += A[k] * B[k][j];//(double)
      }
   }
   
}

//Multiply the matrix "A" by the matrix "B"
void matmul_short_float(int r1, int c1, int r2, int c2, unsigned short A[][c1], float B[][c2], float result[][c2])
{
   int i,j,k;
   // Initializing elements of matrix mult to 0.
   for (i = 0; i < r1; ++i) {
      for (j = 0; j < c2; ++j) {
         result[i][j] = 0;
      }
   }

   // Multiplying first and second matrices and storing it in result
   for (i = 0; i < r1; ++i) {
      for (j = 0; j < c2; ++j) {
         for (k = 0; k < c1; ++k) {
            result[i][j] += A[i][k] * B[k][j];//(double)
         }
      }
   }
}



//Add matrix "A" with the Scalar "K"
void scalar_matadd(int r, int c, float K, float A[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) 
         result[i][j] = K + A[i][j];
}

//Multiply matrix "A" by the Scalar "K"
void scalar_matmul(int r, int c, float K, float A[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) 
         result[i][j] = K * A[i][j];
}

//Divide matrix "A" by the Scalar "K"
void scalar_matdiv(int r, int c, float K, float A[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) 
         result[i][j] = A[i][j] / (float)K;
}

//Add vector "A" with the Scalar "K"
void scalar_vecadd(int r, float K, float *A, float *result)
{
   int i;
   for (i = 0; i < r; ++i)
      result[i] = K + A[i];
}

//Multiply 1D vector "A" by the Scalar "K"
void scalar_vecmul(int r, float K, float *A, float *result)
{
   int i;
   for (i = 0; i < r; ++i)
      result[i] = A[i] * (float)K;
}

//Divide 1D vector "A" by the Scalar "K"
void scalar_vecdiv(int r, float K, float *A, float *result)
{
   int i;
   for (i = 0; i < r; ++i)
      result[i] = A[i] / (float)K;
}

//Multiply matrix "A" by the Scalar "K" and add it to matrix "A"
// result SHOULD NOT be init with zero here.
void scalar_matmul_add(int r, int c, float K, float A[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) 
         result[i][j] = K *A[i][j] + result[i][j];// (double)
}

void scalar_matmul_add_doble(int r, int c, double K, float A[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) 
         result[i][j] = K * (double)A[i][j] + result[i][j];
}

//result = A.*B multiplies arrays A and B by multiplying corresponding elements.
void times_vec2vec(int r, float *A, float *B, float *result)
{
   int i;
   for (i = 0; i < r; ++i)
         result[i] = A[i] * B[i];
}

//result = A.*B multiplies 1D vector A by 2D vector B (Broadcasting).
void times_vec2mat(int r, int c, float A[], float B[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) 
         result[i][j] = A[j] * B[i][j];
}

//
void times_mat(int r, int c, float A[][c], float B[][c], float result[][c])
{
   int i,j;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j)
            result[i][j] = A[i][j] * B[i][j];
}
//result = A.*B multiplies 2D matrix A by 3D matrix B of different size.
// result[1:r, 1:c, 1:d] = A_2D[1:r, None, 1:d] * B_3D[1:r, 1:c, 1:d]
void times_mat2(int r, int c, int d, float A[][d], float B[][c][d], float result[][c][d])
{
   int i,j,k;
   for (i = 0; i < r; ++i)
      for (j = 0; j < c; ++j) 
         for (k = 0; k < d; ++k) 
            result[i][j][k] = A[i][k] * B[i][j][k];
}


//Tensor contraction over specified indices and outer product.
void einsum_bk_jk_bj(int r1, int d1, int r2, int c2, float A[][d1], float B[][c2], float result[][r2])
{
   int i,j,k,b;
   // Initializing elements of matrix mult to 0.
   for (i = 0; i < r1; ++i) {
      for (k = 0; k < r2; ++k) {
         result[i][k] = 0;
      }
   }

   for (b = 0; b < r1; ++b)
      for (j = 0; j < r2; ++j) 
         for (k = 0; k < d1; ++k) 
            result[b][j] += A[b][k] * B[j][k];//(double)
}

// Init the result vector before calling
void einsum_j_jk_k(int r1, int r2, int c2, float* A, float B[][c2], float* result)
{  
   int j,k;
   // Initializing to 0.
   for (k = 0; k < c2; ++k) 
   {
      result[k] = 0;
   }

   for (k = 0; k < c2; ++k)
      for (j = 0; j < r2; ++j) 
         result[k] += A[j] * B[j][k];//(double)
}

// Init the result vector before calling
void einsum_k_jk_j(int r1, int r2, int c2, float* A, float B[][c2], float* result)
{
   // Initializing to 0.
   // for (int k = 0; k < c2; ++k) 
   // {
   //    result[k] = 0;
   // }
   int j,k;
   
   for (j = 0; j < r2; ++j)
      for (k = 0; k < c2; ++k)
         result[j] += A[k] * B[j][k];//(double)
}

//Computes the sum of elements across dimensions of a tensor.
void reduce_sum(int r, float *A, float result)
{
   int i;
   result = 0;
   for (i = 0; i < r; ++i)
      result += A[i];
}

// output should be initialized before calling the function
void  reduce_sum3D(int r, int c, int d, float A[][c][d], float result[][d])
{
   int i,j,k;
   for (i = 0; i < r; ++i) 
      for (j = 0; j < c; ++j)
         for (k = 0; k < d; ++k)
            result[j][k] += A[i][j][k];
}

void  reduce_sum3D_double(int r, int c, int d, double A[][c][d], float result[][d])
{
   int i,j,k;
   for (i = 0; i < r; ++i) 
      for (j = 0; j < c; ++j)
         for (k = 0; k < d; ++k)
            result[j][k] = (float)(result[j][k] + A[i][j][k]);
}

void  reduce_sum4D(int r, int c, int d, int w, float A[][c][d][w], float result[][w])
{
   int i,j,k,t;
   // Initializing elements of matrix to 0.
   for (i = 0; i < d; ++i) {
      for (j = 0; j < w; ++j) {
         result[i][j] = 0;
      }
   }

   for (i = 0; i < r; ++i) 
      for (j = 0; j < c; ++j)
         for (k = 0; k < d; ++k)
            for (t = 0; t < w; ++t)
               result[k][t] += A[i][j][k][t];
}

// Computes the sum of elements across dimensions of a tensor.
// First step to compute tf.reduce_mean(A, axis=(0))
// result SHOULD NOT be init with zero here.
void  reduce_dim(int r, int c, float A[r][c], float *result)
{
   int i,j;
   for (j = 0; j < c; ++j)
      {
         //sum
         for (i = 0; i < r; ++i)
            result[j] += A[i][j];

         //avg
         // result[j] = result[j] / (float)(r);   
      }
}

// Computes the sum of elements across dimensions of a tensor.
// First step to compute tf.reduce_mean(A, axis=(0,1))
// result SHOULD NOT be init with zero here.
void  reduce_dim3D(int r, int c, int d, float A[][c][d], float result[][d])
{
   int i,j,k;
   for (j = 0; j < c; ++j)
      for (k = 0; k < d; ++k)
      {
         //sum
         for (i = 0; i < r; ++i)
            result[j][k] += A[i][j][k];
      }
}

// Computes the mean of elements across dimensions of a tensor.
// equal to tf.reduce_mean(A, axis=(0))
// result SHOULD NOT be init with zero here.
void  reduce_mean(int r, int c, float A[r][c], float *result)
{
   int i,j;
   for (j = 0; j < c; ++j)
      {
         //sum
         for (i = 0; i < r; ++i)
            result[j] += A[i][j];

         //avg
         result[j] = result[j] / (float)(r);   
      }
}

// Computes the mean of elements across dimensions of a tensor.
// equal to tf.reduce_mean(A, axis=(0, 1))
// result SHOULD NOT be init with zero here.
void  reduce_mean3D(int r, int c, int d, float A[][c][d], float result[][d])
{
   int i,j,k;
   for (j = 0; j < c; ++j)
      for (k = 0; k < d; ++k)
      {
         //sum
         for (i = 0; i < r; ++i)
            result[j][k] += A[i][j][k];

         //avg
         result[j][k] = result[j][k] / (float)(r);   
      }
}

// Calculate loss function
void Floss_1D(int r, float A[], float *loss)
{
   float result = 0;
   int i;
   for (i = 0; i < r; ++i) 
      result += (A[i] * A[i]);

   result = 0.5 * result;
   *loss = result;
}

void Floss_2D(int r, int c, float A[][c], float *loss)
{
   float result = 0;
   int i,j;
   for (i = 0; i < r; ++i) 
      for (j = 0; j < c; ++j)
         result += (A[i][j] * A[i][j]);//(double)

   result = 0.5 * result;
   *loss += result;
}

float Floss_3D(int r, int c, int d, float A[][c][d])
{
   float result = 0;
   int i,j,k;
   for (i = 0; i < r; ++i) 
      for (j = 0; j < c; ++j)
         for (k = 0; k < d; ++k)
               result += (A[i][j][k] * A[i][j][k]);//(double)

   result = 0.5 * result;
   return result;
}

void one_hot(int indices, int depth, float* result)
{
   int i;
   for (i = 0; i < depth; ++i)
      if (indices == i)
         result[i] = 1;
      else
         result[i] = 0;
}

// Computes softmax activations.
// https://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
// https://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
// https://codereview.stackexchange.com/questions/180467/implementing-softmax-in-c
void softmax(int input_len, float *input, float* result) 
{
   //   assert(input);
   // assert(input_len >= 0);  Not needed
   int i;
   float m = input[0];
   for (i = 0; i < input_len; i++) 
   {
      if (input[i] > m) 
      {
         m = input[i];
      }
   }

   float sum = 0.0;
   for (i = 0; i < input_len; i++) 
   {
      sum += expf(input[i] - m);
   }

   float offset = m + logf(sum);
   for (i = 0; i < input_len; i++) 
   {
      result[i] = expf(input[i] - offset);
   }
}

void logsoftmax(int input_len, float *input, float* result) 
{
   //   assert(input);
   // assert(input_len >= 0);  Not needed
   int i;
   float m = input[0];
   for (i = 0; i < input_len; i++) 
   {
      if (input[i] > m) 
      {
         m = input[i];
      }
   }

   float sum = 0.0;
   for (i = 0; i < input_len; i++) 
   {
      sum += expf(input[i] - m);
   }

   float offset = m + logf(sum);
   for (i = 0; i < input_len; i++) 
   {
      result[i] = (input[i] - offset);
   }
}

// Computes sparse softmax cross entropy between logits and labels.
// labels = Target
void sparse_softmax_cross_entropy_with_logits(int input_len, int labels, float* logits, float* loss)
{
   float result[input_len];
   logsoftmax(input_len, logits, result);
   *loss = - result[labels];
}

// Returns the index with the largest value across axes of a tensor.
void argmax(int r, float* input, int* result)
{
   int index = 0;
   int i;
   float max = input[index];

   for (i = 1; i < r; ++i)
   {
      if (input[i] > max)
      {
         max = input[i];
         index = i;
      }
   }
   *result  = index;
}

// Returns the truth value of (x == y) element-wise.
void equal(int x, int y, int* result)
{
   int temp;
   if(x==y)
      temp = 1;
   else
      temp = 0;

   *result = temp;
}

// Generate a random permutation of array elements
// https://www.w3resource.com/c-programming-exercises/array/c-array-exercise-77.php
void changeValues(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Generate a random permutation of array elements
void rd_permutation_arr(int n, int* v)
{
   //  srand ( time(NULL) );
   int i,j;
   for (i = n-1; i > 0; i--)
   {
      j = rand() % (i+1);
      changeValues(&v[i], &v[j]);
   }
}

// Generate a random permutation of n dim
void rd_permutation(int n, int* v)
{
   int i;
   // Fill the vector with the values
   // 1, 2, 3, ..., n
   for (i = 0; i < n; i++)
      v[i] = i + 1;

   rd_permutation_arr(n, v);
}