#ifndef _EPROP_LSNN_H_
#define _EPROP_LSNN_H_

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "models.h"
#include "math_tensorflow.h"
#include "main.h"

int initialize_eprop_parameters(uint32_t n_in, uint32_t n_regular, uint32_t n_adaptive, uint32_t n_out, int seq_len, Cell *cell, State *state);
void init_adam_param(uint32_t n_in, uint32_t n_regular, uint32_t n_adaptive, uint32_t n_out, Adam *adam);
void eprop(Cell *cell, State *state, int n_batch, int n_in, int n_rec, int n_out, int t, float* xt, 
           int yt, float* filtered_z, float* output, float *output_error, 
           float epsilon_v_in[][n_rec], float epsilon_a_in[][n_rec], float epsilon_v_rec[][n_rec], float epsilon_a_rec[][n_rec],
           float e_trace_in[][n_rec], float e_trace_in_tmp1[][n_rec], float e_trace_in_tmp2[][n_rec], float e_trace_rec[][n_rec],
           float e_trace_rec_tmp1[][n_rec], float e_trace_rec_tmp2[][n_rec],
           float *random_tensor, bool is_train,
           int n_iter, uint16_t print_every);
void AdamOptimizer(Cell *cell, Adam *adam, int n_in, int n_rec, int n_out);
void optimize_eprop(Cell *cell, State *state, Adam *adam, int n_batch, int n_in, int n_rec, int n_out, int seq_len, 
                    float X_in[][n_in], int Y_target, int* correct_num, float* loss, bool is_train, int n_iter, uint16_t print_every);

#endif