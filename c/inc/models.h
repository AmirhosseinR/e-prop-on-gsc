#ifndef _MODELS_H_
#define _MODELS_H_

// #define DOUBLE_PRECISION

#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h> // rand()
#include <stdio.h> 

#define N_batch              1                       // batch size of the testing set
#define N_out                12                      // number of output neurons (number of target curves)
#define N_in                 40                      // number of input units
#define N_ch                 2                       // number of input channel
#define N_regular            0                       // number of regular spiking units in the recurrent layer
#define N_adaptive           120                     // number of adaptive spiking units in the recurrent layer
#define N_rec                N_regular + N_adaptive  // total number of recurrent units
#define N_epoch              33                      // number of iterations
#define Seq_len              100                     // number of time steps per sequence
#define N_train              584                     // number of data in the training dataset:  55948, 622
#define N_valid              141                     // number of data in the validation dataset: 7058, 141
#define N_test               525                     // number of data in the test dataset:      3081, 61

#define delta_t              1.0f                    // (ms) simulation step
#define vth                  1.6f
#define tau_a                150.0f                  // Adaptation time constant
#define tau_m                5.0f                    // Membrane time constant of recurrent neurons
#define tau_out              10.0f                   // Mikolov: tau for PSP decay at output
#define BETA                 0.184f                  // Scaling constant of the adaptive threshold
#define reg                  0.0f                    // regularization coefficient
#define reg_rate             10                      // target rate for regularization

#define drop_out_probability -0.85f
#define adam_epsilon         1.0e-5f
#define momentum             0.9f
#define l2                   1.0e-5                   // l2 regularization
#define lr_init              0.01f
#define lr_decay             0.3f
#define lr_decay_every       2                       // Decay learning rate every

#define shuffle              false

// Adam optimization
#define Beta_m               momentum
#define Beta_v               0.999f
#define Epsilon              adam_epsilon

// Multi-Gaussian surrogate gradient
#define gamma                0.5f                     // dampening factor to stabilize learning in RNNs
#define lens                 0.5f
#define scale                6.0f
#define hight                0.15f

#define PI_F                 3.14159265f
#define TWO_PI_F             6.28318531f
#define SQRT_2PI_F           2.5066282746f


struct State 
{
    float z[N_rec];
    float v[N_rec];
    float b[N_rec];
};
typedef struct State State;
extern    State  state;

struct Cell 
{
    float dt;
    float b_j0_value;
    float alpha;
    float one_alpha;
    float rho;
    float one_rho;
    float one_alpha_x_one_rho;
    float decay_out;
    float beta[N_rec];
    float w_in[N_in*N_ch][N_rec];
    float w_rec[N_rec][N_rec];
    float w_out[N_rec][N_out];
    float grad_in[N_in*N_ch][N_rec];
    float grad_rec[N_rec][N_rec];
    float grad_out[N_rec][N_out];
};
typedef struct Cell Cell;
extern    Cell  cell;

struct Adam
{
    float learning_rate;
    float beta_m;
    float beta_v;
    float epsilon;
    float m_in[N_in*N_ch][N_rec];
    float v_in[N_in*N_ch][N_rec];
    float m_rec[N_rec][N_rec];
    float v_rec[N_rec][N_rec];
    float m_out[N_rec][N_out];
    float v_out[N_rec][N_out];
    float beta_m_power;
    float beta_v_power;

};
typedef struct Adam Adam;
extern    Adam adam;

#include "math_tensorflow.h"
//#include "tutorial_pattern_generation.h"



//The membrane potential. Define the pseudo derivative used to derive through spikes.
void pseudo_derivative(int r, int c, float v_scaled[][c], float dampening_factor, float result[][c]);

//Heaviside function (to compute the spikes)
void SpikeFunction(int n_rec, float* v_scaled, float* z, int systicks);

//RNN cell model to simulate Learky Integrate and Fire (LIF) neurons
void CustomALIF(Cell *cell, State *state, int n_batch, int n_in, int n_rec, int t, float* inputs, float* v_scaled, bool is_train, int n_iter, uint16_t print_every);

void update_epsilon_v(uint32_t n_pre, uint32_t n_post, float* z_pre, float alpha, float epsilon_v[][n_post]);
void update_epsilon_a(uint32_t n_pre, uint32_t n_post, Cell *cell, float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post]);
void update_e_trace(uint32_t n_pre, uint32_t n_post, float one_alpha, float* beta, float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post], float e_trace[][n_post]);
void update_e_trace_optzd(uint32_t n_pre, uint32_t n_post, float one_alpha, float* beta, float decay_out, float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post], float e_trace[][n_post]);
void compute_spike_grad(uint32_t n_pre, float* v_scaled, float* psi);
void compute_eligibility_traces(Cell *cell, uint32_t n_pre, uint32_t n_post, float* z_pre, bool is_rec, 
                                float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post], float e_trace[][n_post]);
void compute_eligibility_traces_optzd(Cell *cell, uint32_t n_pre, uint32_t n_post, float* z_pre, bool is_rec, 
                                float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post], float e_trace[][n_post]);
void compute_loss_gradient_out(uint32_t n_out, uint32_t n_rec, uint32_t seq_len, float* output_error,  float* filtered_z, float grad_out[][n_out]);
//Shift the input on the time dimension by one.
void shift_by_one_time_step(int r, int c, float A[][c], float shifted_A[][c]);

//Filters a tensor with an exponential filter.
void exp_convolve(int r, float tensor[], float decay, float *result);
void exp_convolve2D(int r, int c, float tensor[][c], float decay, float result[][c]);
void exp_convolve3D(int r, int c, int d, float tensor[][c][d], float decay, float result[][c][d]);

void get_stats(int r, float *A, float *minA, float *maxA, float *meanA, float *stdA);

void Loss_L2(Cell *cell, uint32_t n_in, uint32_t n_rec, uint32_t n_out, float* losses_l2);
void grad_L2(Cell *cell, uint32_t n_in, uint32_t n_rec, uint32_t n_out);

void dropout(int r, float* x, float keep_prob, float* result);
void rand_array(int r, float* random_tensor);
void dropout_v2(int r, float* x, float* random_tensor, float keep_prob, float* result);
void dropout2D(int ts, int c, float x[][c], float keep_prob, float result[][c]);
void dropout2D_v2(int ts, int c, float x[][c], float* random_tensor, float keep_prob, float result[][c]);

int read_features(int r, int c, int d, char *file_path, float result[][c][d]);
int read_nth_features(char *file_path, int line, int r, int c, float result[][c]);
int read_target(int r, char *file_path, int* result);
int read_nth_target(char* file_path, int line, int* result);
int get_nth_line(char *file_path, int line, int r, int c, float result[][c]);


#endif