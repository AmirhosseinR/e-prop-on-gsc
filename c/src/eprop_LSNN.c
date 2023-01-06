#include "eprop_LSNN.h"

//Initialize parameters with small random values
//Equivalent to
//class LightLIF(Cell):
//   def __init__ ():
//       ...
int initialize_eprop_parameters(uint32_t n_in, uint32_t n_regular, uint32_t n_adaptive, uint32_t n_out, int seq_len, Cell *cell, State *state)
{
    FILE *fp;
    int i, j;
    uint32_t n_rec = n_regular + n_adaptive;
    //------------------------------------------
    for (j = 0; j < n_rec; ++j) 
    {
        state->z[j] = 0;
        state->v[j] = 0;
        state->b[j] = 0;
    }
    //------------------------------------------
    cell->dt         = delta_t;
    cell->b_j0_value = vth;
    cell->alpha      = exp(-delta_t / (float) tau_m);
    cell->one_alpha  = 1 - cell->alpha;
    cell->rho        = exp(-delta_t / (float) tau_a); // decay_b -> rho
    cell->one_rho    = 1 - cell->rho;
    cell->one_alpha_x_one_rho = (1 - cell->alpha) * (1 - cell->rho);
    cell->decay_out  = exp(-delta_t / (float) tau_out);
    for (j = 0; j < n_rec; ++j)
        if (j < n_regular)
            cell->beta[j] = 0;
        else
            cell->beta[j] = BETA;
    //------------------------------------------
    #ifndef __LINUX__
    fp = fopen("..\\files\\w_in.dat", "r");
    #else
    fp = fopen("../files/w_in.dat", "r");
    #endif

    if (!fp) { 
        printf("Could not open w_in.dat\n");
        return -1;
    }
    for(i=0; i<n_in; ++i) 
    {
        for(j=0; j<n_rec; ++j) 
        {
            fscanf(fp,"%f",&cell->w_in[i][j]);
        }
    }
    fclose(fp);
    //------------------------------------------
    #ifndef __LINUX__
    fp = fopen("..\\files\\w_rec.dat", "r");
    #else
    fp = fopen("../files/w_rec.dat", "r");
    #endif

    if (!fp) 
    {
        printf("Could not open w_rec.dat\n");
        return -1;
    }
    for(i=0; i<n_rec; ++i) 
    {
        for(j=0; j<n_rec; ++j) 
        {
            fscanf(fp,"%f",&cell->w_rec[i][j]);
        }
    }
    fclose(fp);
    //------------------------------------------
    #ifndef __LINUX__
    fp = fopen("..\\files\\w_out.dat", "r");
    #else
    fp = fopen("../files/w_out.dat", "r");
    #endif

    if (!fp) 
    {
        printf("Could not open w_out.dat\n");
        return -1;
    }
    for(i=0; i<n_rec; ++i) 
    {
        for(j=0; j<n_out; ++j) 
        {
            fscanf(fp,"%f",&cell->w_out[i][j]);
        }
    }
    fclose(fp);
    //------------------------------------------
    return 0;
}

void init_adam_param(uint32_t n_in, uint32_t n_regular, uint32_t n_adaptive, uint32_t n_out, Adam *adam)
{   
    uint32_t n_rec = n_regular + n_adaptive;
    int i, j;

    adam->learning_rate = lr_init;
    adam->beta_m        = Beta_m;
    adam->beta_v        = Beta_v;
    adam->epsilon       = Epsilon;

    for (i = 0; i < n_in; ++i)
        for (j = 0; j < n_rec; ++j)
        {
            adam->m_in[i][j]  = 0;
            adam->v_in[i][j]  = 0;
        }

    for (i = 0; i < n_rec; ++i)
        for (j = 0; j < n_rec; ++j)
        {
            adam->m_rec[i][j]  = 0;
            adam->v_rec[i][j]  = 0;
        }

    for (i = 0; i < n_rec; ++i)
        for (j = 0; j < n_out; ++j)
        {
            adam->m_out[i][j]  = 0;
            adam->v_out[i][j]  = 0;
        }

    adam->beta_m_power  = adam->beta_m;
    adam->beta_v_power  = adam->beta_v;
}

void AdamOptimizer(Cell *cell, Adam *adam, int n_in, int n_rec, int n_out)
{
    int   i,j;
    float lr;
    float weight_update;

    lr = adam->learning_rate * sqrt(1-adam->beta_v_power) / (1-adam->beta_m_power);

    for (i = 0; i < n_in; ++i)
        for (j = 0; j < n_rec; ++j)
        {
            adam->m_in[i][j] = adam->beta_m * adam->m_in[i][j] + (1 - adam->beta_m) * cell->grad_in[i][j];
            adam->v_in[i][j] = adam->beta_v * adam->v_in[i][j] + (1 - adam->beta_v) * cell->grad_in[i][j] * cell->grad_in[i][j];
            weight_update = (lr * adam->m_in[i][j]) / (sqrt(adam->v_in[i][j])+adam->epsilon);
            cell->w_in[i][j] = cell->w_in[i][j] - weight_update;
        }

    for (i = 0; i < n_rec; ++i)
        for (j = 0; j < n_rec; ++j)
        {
            adam->m_rec[i][j] = adam->beta_m * adam->m_rec[i][j] + (1 - adam->beta_m) * cell->grad_rec[i][j];
            adam->v_rec[i][j] = adam->beta_v * adam->v_rec[i][j] + (1 - adam->beta_v) * cell->grad_rec[i][j] * cell->grad_rec[i][j];
            weight_update = (lr * adam->m_rec[i][j]) / (sqrt(adam->v_rec[i][j]) + adam->epsilon);
            cell->w_rec[i][j] = cell->w_rec[i][j] - weight_update;
        }

    for (i = 0; i < n_rec; ++i)
        for (j = 0; j < n_out; ++j)
        {
            adam->m_out[i][j] = adam->beta_m * adam->m_out[i][j] + (1 - adam->beta_m) * cell->grad_out[i][j];
            adam->v_out[i][j] = adam->beta_v * adam->v_out[i][j] + (1 - adam->beta_v) * cell->grad_out[i][j] * cell->grad_out[i][j];
            weight_update = (lr * adam->m_out[i][j]) / (sqrt(adam->v_out[i][j]) + adam->epsilon);
            cell->w_out[i][j] = cell->w_out[i][j] - weight_update;
        }

    adam->beta_m_power *= adam->beta_m;
    adam->beta_v_power *= adam->beta_v;

}

//Implement the forward propagation of the recurrent neural network
void eprop(Cell *cell, State *state, int n_batch, int n_in, int n_rec, int n_out, int t, float* xt, 
           int yt, float* filtered_z, float* output, float *output_error, 
           float epsilon_v_in[][n_rec], float epsilon_a_in[][n_rec], float epsilon_v_rec[][n_rec], float epsilon_a_rec[][n_rec],
           float e_trace_in[][n_rec], float e_trace_in_tmp1[][n_rec], float e_trace_in_tmp2[][n_rec], float e_trace_rec[][n_rec],
           float e_trace_rec_tmp1[][n_rec], float e_trace_rec_tmp2[][n_rec],
           float *random_tensor, bool is_train,
           int n_iter, uint16_t print_every)
{   
    uint16_t k;

    uint16_t i,j;

    float v_scaled[n_rec];
    float psi[n_rec];
    
    float z_pre_step[n_rec];
    float output_tmp[n_out];

    float e_trace_in_optzd[n_in][n_rec];
    float epsilon_v_in_optzd[n_in][n_rec];
    float epsilon_a_in_optzd[n_in][n_rec];

    float e_trace_rec_optzd[n_rec][n_rec];
    float epsilon_v_rec_optzd[n_rec][n_rec];
    float epsilon_a_rec_optzd[n_rec][n_rec];

    for (i = 0; i < n_in; ++i) 
        for (j = 0; j < n_rec; ++j) 
        {
            e_trace_in_optzd[i][j]   = e_trace_in_tmp2[i][j];
            epsilon_v_in_optzd[i][j] = epsilon_v_in[i][j];
            epsilon_a_in_optzd[i][j] = epsilon_a_in[i][j];
        }
            

    for (i = 0; i < n_rec; ++i) 
        for (j = 0; j < n_rec; ++j) 
        {
            e_trace_rec_optzd[i][j]   = e_trace_rec_tmp2[i][j];
            epsilon_v_rec_optzd[i][j] = epsilon_v_rec[i][j];
            epsilon_a_rec_optzd[i][j] = epsilon_a_rec[i][j];

        }

    // eq. (6) , (7)
    for (k = 0; k < n_rec; ++k)
        z_pre_step[k] = state->z[k];

    CustomALIF(cell, state, n_batch, n_in, n_rec, t, xt, v_scaled, is_train, n_iter, print_every);//update z & v & b

    //aplly dropout to z
    if (is_train)
        dropout_v2(n_rec, state->z, random_tensor, drop_out_probability, state->z);

    // Accumulate filtered_z value from previous time step.
    exp_convolve(n_rec, state->z, cell->decay_out, filtered_z);

    einsum_j_jk_k(n_rec, n_rec, n_out, filtered_z, cell->w_out, output_tmp);// output_tmp: output in each time step
    vecadd(n_out, output_tmp, output, output); // Accumulate output value from previous time step.
    
    if (is_train)
    {
        //'''''''''''''''''''''''//||
        //Back propagation path // ||
        //_____________________//__||
        compute_spike_grad(n_rec, v_scaled, psi);

        // Accumulate e_trace_in value from previous time step.
        compute_eligibility_traces(cell, n_in, n_rec, xt, false, psi, epsilon_v_in, epsilon_a_in, e_trace_in_tmp1);
        exp_convolve2D(n_in, n_rec, e_trace_in_tmp1, cell->decay_out, e_trace_in_tmp2); // filtered_e_trace
        matadd(n_in, n_rec, e_trace_in, e_trace_in_tmp2, e_trace_in);

        compute_eligibility_traces_optzd(cell, n_in, n_rec, xt, false, psi, epsilon_v_in_optzd, epsilon_a_in_optzd, e_trace_in_optzd);

        for (i = 0; i < n_in; ++i)
            for (j = 0; j < n_rec; ++j)
                if(e_trace_in_optzd[i][j] != e_trace_in_tmp2[i][j])
                    printf("\ne_trace_in_optzd[%i][%i]: %f, e_trace_in_tmp2[%i][%i]: %f", i,j, e_trace_in_optzd[i][j], i,j, e_trace_in_tmp2[i][j]);
        
        // Accumulate e_trace_rec value from previous time step.
        compute_eligibility_traces(cell, n_rec, n_rec, z_pre_step, true, psi, epsilon_v_rec, epsilon_a_rec, e_trace_rec_tmp1);
        exp_convolve2D(n_rec, n_rec, e_trace_rec_tmp1, cell->decay_out, e_trace_rec_tmp2);
        matadd(n_rec, n_rec, e_trace_rec, e_trace_rec_tmp2, e_trace_rec);

        compute_eligibility_traces_optzd(cell, n_rec, n_rec, z_pre_step, true, psi, epsilon_v_rec_optzd, epsilon_a_rec_optzd, e_trace_rec_optzd);

        for (i = 0; i < n_rec; ++i) 
            for (j = 0; j < n_rec; ++j) 
                if(e_trace_rec_optzd[i][j] != e_trace_rec_tmp2[i][j])
                    printf("\ne_trace_rec_optzd[%i][%i]: %f, e_trace_rec_tmp2[%i][%i]: %f", i,j, e_trace_rec_optzd[i][j], i,j, e_trace_rec_tmp2[i][j]);
    }
}

//Execute one step of the optimization to train the model.
void optimize_eprop(Cell *cell, State *state, Adam *adam, int n_batch, int n_in, int n_rec, int n_out, int seq_len, 
                    float X_in[][n_in], int Y_target, int* correct_num, float* loss, bool is_train, int n_iter, uint16_t print_every)
{
    int   i, j, t;
    float epsilon_v_in[n_in][n_rec];
    float epsilon_a_in[n_in][n_rec];
    float epsilon_v_rec[n_rec][n_rec];
    float epsilon_a_rec[n_rec][n_rec];
    float xt[n_in];
    float output[n_out];
    float output_softmax[n_out];
    float output_error[n_out];
    float output_mean[n_out];
    float target_outputs[n_out];
    float learning_signals[n_rec];
    float e_trace_in[n_in][n_rec];
    float e_trace_in_tmp1[n_in][n_rec];
    float e_trace_in_tmp2[n_in][n_rec];
    float e_trace_rec[n_rec][n_rec];
    float e_trace_rec_tmp1[n_rec][n_rec];
    float e_trace_rec_tmp2[n_rec][n_rec];
    float filtered_z[n_rec];
    float random_tensor_in[n_in];
    float random_tensor_rec[n_rec];
    int   output_prediction;
    int   is_correct;
    float loss_batch, losses_l2;

    //------------------------------------------  
    for ( j = 0; j < n_rec; ++j) 
    {
        state->z[j] = 0;
        state->v[j] = 0;
        state->b[j] = 0;

        learning_signals[j] = 0;
        filtered_z[j] = 0;
    }
    //------------------------------------------
    for ( j = 0; j < n_out; ++j)
    {
        output_error[j] = 0;
        output[j] = 0;
    }
    //------------------------------------------
    if (is_train)
    {
        for (i = 0; i < n_in; ++i) 
            for (j = 0; j < n_rec; ++j) 
            {
                e_trace_in[i][j] = 0;
                e_trace_in_tmp1[i][j] = 0;
                e_trace_in_tmp2[i][j] = 0;
                epsilon_v_in[i][j] = 0;
                epsilon_a_in[i][j] = 0;
            }

        for (i = 0; i < n_rec; ++i) 
            for (j = 0; j < n_rec; ++j)
            {
                e_trace_rec[i][j] = 0;
                e_trace_rec_tmp1[i][j] = 0;
                e_trace_rec_tmp2[i][j] = 0;
                epsilon_v_rec[i][j] = 0;
                epsilon_a_rec[i][j] = 0;
            }
        //------------------------------------------
        for(i = 0; i < n_rec; ++i)
            for(j = 0; j < n_out; ++j)
                cell->grad_out[i][j] = 0;
    }
    //------------------------------------------
    // generate random array for dropout to use for all time steps
    if (is_train)
    {
        rand_array(n_in, random_tensor_in); // input
        rand_array(n_rec, random_tensor_rec); // z
    }
    //------------------------------------------
    //loop over all time-steps
    for (t=0; t < seq_len; t++)
    {
        for (j = 0; j < n_in; ++j) 
            xt[j] = X_in[t][j];
        
        if (is_train)
            dropout_v2(n_in, xt, random_tensor_in, drop_out_probability, xt); // input

        // Forward propagate through time
        eprop(cell, state, n_batch, n_in, n_rec, n_out, t, xt, 
              Y_target, filtered_z, output, output_error, 
              epsilon_v_in, epsilon_a_in, epsilon_v_rec, epsilon_a_rec,
              e_trace_in, e_trace_in_tmp1, e_trace_in_tmp2, e_trace_rec, 
              e_trace_rec_tmp1, e_trace_rec_tmp2,
              random_tensor_rec, is_train,
              n_iter, print_every);

        for(i = 0; i < n_rec; ++i)
            for(j = 0; j < n_out; ++j)
                cell->grad_out[i][j] += (1 * filtered_z[i]);
    }
    //------------------------------------------
    scalar_vecdiv(n_out, seq_len, output, output_mean);
    sparse_softmax_cross_entropy_with_logits(n_out, Y_target, output_mean, &loss_batch);
    
    Loss_L2(cell, n_in, n_rec, n_out, &losses_l2);
    *loss += loss_batch + (l2 * losses_l2);

    argmax(n_out, output_mean, &output_prediction);
    equal(Y_target, output_prediction, &is_correct);
    *correct_num += is_correct;

    if (is_train)
        if (n_iter % 1 == 0)
        {
            printf("\nline_cnt: %d", n_iter);
            printf("\nprediction: [%d], classes: [%d]", output_prediction, Y_target);
            printf("\nloss_batch: %0.7f\n", loss_batch);
        }

    if (is_train)
    {
        softmax(n_out, output_mean, output_softmax);
        one_hot(Y_target, n_out, target_outputs);
        vecsub(n_out, output_softmax, target_outputs, output_error);

        // eq. (4) w_out
        einsum_k_jk_j(n_out, n_rec, n_out, output_error, cell->w_out, learning_signals); 
        scalar_vecdiv(n_rec, seq_len, learning_signals, learning_signals);

        times_vec2mat(n_in,  n_rec, learning_signals, e_trace_in,  cell->grad_in);
        times_vec2mat(n_rec, n_rec, learning_signals, e_trace_rec, cell->grad_rec);
        // compute_loss_gradient_out(n_out, n_rec, seq_len, output_error,  filtered_z, cell->grad_out);

        for(i = 0; i < n_rec; ++i)
            for(j = 0; j < n_out; ++j)
                cell->grad_out[i][j] = (cell->grad_out[i][j] * output_error[j]) / seq_len;

        // Make effect of L2 regularization on bach path
        grad_L2(cell, n_in, n_rec, n_out);

        // Update weights
        AdamOptimizer(cell, adam, n_in, n_rec, n_out);
    }
}
