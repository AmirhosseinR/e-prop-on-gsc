#include "models.h"


/*
BTW eq. (10) & (11)
The membrane potential.
Define the pseudo derivative used to derive through spikes.
:param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
:param dampening_factor: parameter that stabilizes learning
:return:
*/
void pseudo_derivative(int r, int c, float v_scaled[][c], float dampening_factor, float result[][c])
{
    //return tf.maximum(1 - tf.abs(v_scaled), 0) * dampening_factor
    float temp;
    int i,j;
    for (i = 0; i < r; ++i) {
        for (j = 0; j < c; ++j) {
            temp = 1-fabs(v_scaled[i][j]);
            if (temp > 0)
                result[i][j] = (1-fabs(v_scaled[i][j])) * dampening_factor;
            else
                result[i][j] = 0;
        }
    }
}


/*
The tensorflow function which is defined as a Heaviside function (to compute the spikes),
but with a gradient defined with the pseudo derivative.
:param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
:param dampening_factor: parameter to stabilize learning
:return: the spike tensor
*/
void SpikeFunction(int n_rec, float* v_scaled, float* z, int systicks)
{
    int j;
    for (j = 0; j < n_rec; ++j) {
        if (v_scaled[j] > 0)
            {
                // printf("\nSND spk %d at %d, DMA: %d", j, systicks, 0);
                z[j] = 1;
            }
        else
            z[j] = 0;
    }
}


/*
CustomALIF provides the recurrent tensorflow cell model for implementing LSNNs in combination with eligibility propagation (e-prop).
This model uses v^{t+1} = alpha * v^t + (1-alpha)*i_t - z * Aj
*/
void CustomALIF(Cell *cell, State *state, int n_batch, int n_in, int n_rec, int t, float* inputs, float* v_scaled, bool is_train, int n_iter, uint16_t print_every)
{
    uint32_t j, k;
    // float v_scaled[n_rec];
    float B[n_rec];
    float i_in[n_rec];
    float i_rec[n_rec];
    float I_reset[n_rec];
    float it[n_rec];

    // eq. (10), decay_b -> rho, b -> a_j 
    for (k = 0; k < n_rec; ++k) // (1-cell->rho) 
        state->b[k] = cell->rho * state->b[k] +  cell->one_rho * state->z[k]; //new_b: state->b, old_z: state->z

    // i_in = matmul(Win_ij.x_i)
    vecmatmul(n_in, n_in, n_rec, inputs, cell->w_in, i_in);

    // i_rec = matmul(Wrec_ij.z_i)
    vecmatmul(n_rec, n_rec, n_rec, state->z, cell->w_rec, i_rec);

    // it = i_in + i_rec
    vecadd(n_rec, i_in, i_rec, it);
    
    //B = new_b * cell->beta + cell->b_j0_value , eq. (8), adaptive_thr -> A_j
    vecmul(n_rec, cell->beta, state->b, B);
    scalar_vecadd(n_rec, cell->b_j0_value, B, B);
    
    times_vec2vec(n_rec, state->z, B, I_reset);
    //The neurons membrane potential is reduced by a constant value after an output spike, 
    //which relates our model to the spike response model
    //new_v = alpha * v + (1 - alpha) * i_t - I_reset;  eq. (6)
    for (j = 0; j < n_rec; ++j)
    {
        // state->v[j] = cell->alpha * state->v[j] + (1-cell->alpha) * it[j] - I_reset[j];//new_v
        state->v[j] = cell->alpha * state->v[j] + (cell->one_alpha) * it[j] - I_reset[j];//new_v
    }

    vecsub(n_rec, state->v, B, v_scaled);

    SpikeFunction(n_rec, v_scaled, state->z, t);//new_z

    //-------------------------------------------------------------
}

void gaussian(uint32_t n_rec, float* x, float mu, float sigma, float* result)
{   
    int i;
    for (i = 0; i < n_rec; ++i)
        result[i] = exp(-((x[i] - mu) * (x[i] - mu)) / (2 * sigma *sigma)) / (SQRT_2PI_F) / sigma;
}



void update_epsilon_v(uint32_t n_pre, uint32_t n_post, float* z_pre, float alpha, float epsilon_v[][n_post])
{
    int i,j;
    for (i = 0; i < n_pre; ++i)
        for (j = 0; j < n_post; ++j)
            epsilon_v[i][j] = alpha * epsilon_v[i][j] + z_pre[i];
}

void update_epsilon_a(uint32_t n_pre, uint32_t n_post, Cell *cell, float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post])
{
    int i,j;
    for (i = 0; i < n_pre; ++i)
        for (j = 0; j < n_post; ++j)
            {
             // epsilon_a[i][j] = (rho       - (1-rho)       * beta[j]       * psi[j]) * epsilon_a[i][j] + psi[j] * epsilon_v[i][j] * (1-rho) * (1-alpha);
                epsilon_a[i][j] = (cell->rho - cell->one_rho * cell->beta[j] * psi[j]) * epsilon_a[i][j] + psi[j] * epsilon_v[i][j] * cell->one_alpha_x_one_rho;
            }

}

void update_e_trace(uint32_t n_pre, uint32_t n_post, float one_alpha, float* beta, float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post], float e_trace[][n_post])
{
    int i,j;
    for (i = 0; i < n_pre; ++i)
        for (j = 0; j < n_post; ++j)
        {
         // e_trace[i][j] = psi[j] * ((1-alpha) * epsilon_v[i][j] - beta[j] * epsilon_a[i][j]);  // eq. (25)
            e_trace[i][j] = psi[j] * (one_alpha * epsilon_v[i][j] - beta[j] * epsilon_a[i][j]);  // eq. (25)
        }
}

void update_e_trace_optzd(uint32_t n_pre, uint32_t n_post, float one_alpha, float* beta, float decay_out, float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post], float e_trace[][n_post])
{
    int i,j;
    float temp;
    for (i = 0; i < n_pre; ++i)
        for (j = 0; j < n_post; ++j)
        {
         // e_trace[i][j] = psi[j] * ((1-alpha) * epsilon_v[i][j] - beta[j] * epsilon_a[i][j]);  // eq. (25)
            e_trace[i][j] = (e_trace[i][j] * decay_out) + (psi[j] * (one_alpha * epsilon_v[i][j] - beta[j] * epsilon_a[i][j]));  // eq. (25) + exp_convolve2D
        }
}


void compute_spike_grad(uint32_t n_pre, float* v_scaled, float* psi)
{    
    float temp [n_pre];
    float temp1[n_pre];
    float temp2[n_pre];
    float temp3[n_pre];
    int i;
    //------------------psi--------------------------------
    gaussian(n_pre, v_scaled,   0.0,   1   * lens, temp1);
    scalar_vecmul(n_pre, (1.0 + hight), temp1, temp1);

    gaussian(n_pre, v_scaled,  lens, scale * lens, temp2);
    scalar_vecmul(n_pre, (0.0 + hight), temp2, temp2);

    gaussian(n_pre, v_scaled, -lens, scale * lens, temp3);
    scalar_vecmul(n_pre, (0.0 + hight), temp3, temp3);

    for (i = 0; i < n_pre; ++i)
        temp[i] = temp1[i] - temp2[i] - temp3[i];

    scalar_vecmul(n_pre, gamma, temp, psi);
    //-----------------------------------------------------
}

void compute_eligibility_traces(Cell *cell, uint32_t n_pre, uint32_t n_post, float* z_pre, bool is_rec, 
                                float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post], float e_trace[][n_post])
{   
    int i,j;
    update_epsilon_v(n_pre, n_post, z_pre, cell->alpha, epsilon_v);
    // scalar_matmul(n_pre, n_post, (1-cell->alpha), epsilon_v, epsilon_v);

    update_epsilon_a(n_pre, n_post, cell, psi, epsilon_v, epsilon_a);

    update_e_trace(n_pre, n_post, cell->one_alpha, cell->beta, psi, epsilon_v, epsilon_a, e_trace);

    if (is_rec == true)
    {
        for(i = 0; i < n_pre; ++i)
            for(j = 0; j < n_post; ++j)
                if(i==j)
                {
                    e_trace[i][j]   = 0;
                    epsilon_v[i][j] = 0;
                    epsilon_a[i][j] = 0;
                }
    }

}

void compute_eligibility_traces_optzd(Cell *cell, uint32_t n_pre, uint32_t n_post, float* z_pre, bool is_rec, 
                                float* psi, float epsilon_v[][n_post], float epsilon_a[][n_post], float e_trace[][n_post])
{   
    int i,j;
    update_epsilon_v(n_pre, n_post, z_pre, cell->alpha, epsilon_v);
    // scalar_matmul(n_pre, n_post, (1-cell->alpha), epsilon_v, epsilon_v);

    update_epsilon_a(n_pre, n_post, cell, psi, epsilon_v, epsilon_a);

    update_e_trace_optzd(n_pre, n_post, cell->one_alpha, cell->beta, cell->decay_out, psi, epsilon_v, epsilon_a, e_trace);

    if (is_rec == true)
    {
        for(i = 0; i < n_pre; ++i)
            for(j = 0; j < n_post; ++j)
                if(i==j)
                {
                    e_trace[i][j]   = 0;
                    epsilon_v[i][j] = 0;
                    epsilon_a[i][j] = 0;
                }
    }

}

// grad_out = tf.reduce_mean(output_error[:, None, None, :] * filtered_z[:, :, :, None], axis=(0, 1))
void compute_loss_gradient_out(uint32_t n_out, uint32_t n_rec, uint32_t seq_len, float* output_error,  float* filtered_z, float grad_out[][n_out])
{
    int i,j;
    for(i = 0; i < n_rec; ++i)
        for(j = 0; j < n_out; ++j)
            grad_out[i][j] = (output_error[j] * filtered_z[i]) / seq_len;
}

/*
Temporal filters: eq. (12)
Filters a tensor with an exponential filter.
:param tensor: a tensor of shape (trial, time, neuron)
:param decay: a decay constant of the form exp(-dt/tau) with tau the time constant
:return: the filtered tensor of shape (trial, time, neuron)
*/
void exp_convolve(int r, float* tensor, float decay, float *result)
{
    int j;
    for (j = 0; j < r; ++j)
    {
        result[j] = result[j] * decay + tensor[j] ;
    }
}

void exp_convolve2D(int r, int c, float tensor[][c], float decay, float result[][c])
{
    int i,j;
    for (i = 0; i< r; ++i)
        for (j = 0; j < c; ++j)
    {
        result[i][j] = result[i][j] * decay + tensor[i][j];
    }
}

void exp_convolve3D(int r, int c, int d, float tensor[][c][d], float decay, float result[][c][d])
{
    // float temp = (1 - decay);
    int i,j,k;
    for (i = 0; i < r; ++i)
        for (j = 0; j < c; ++j)
            for (k = 0; k < d; ++k) 
            {
                // result[i][j][k] = result[i][j][k] * (float)decay + (float)temp * tensor[i][j][k];
                result[i][j][k] = (result[i][j][k] - tensor[i][j][k]) * decay + tensor[i][j][k];
            }
}

void get_stats(int r, float *A, float *minA, float *maxA, float *meanA, float *stdA)
{
    int i;
    float temp = 0;

    *minA = A[0];
    *maxA = A[0];
    *meanA = A[0];
    *stdA = 0;

    for (i = 1; i < r; ++i)
    {
        if (*minA > A[i])
            *minA = A[i];

        if (*maxA < A[i])
            *maxA = A[i];

        *meanA = *meanA + A[i];
    }

    *meanA = *meanA / (float)r;


    for (i = 0; i < r; ++i)
    {
        temp = (A[i] - *meanA);
        *stdA += (temp * temp);
    }
    
    *stdA = *stdA / (float)r;
    *stdA = sqrtf(*stdA);

}

// Compute L2 regularization Loss
void Loss_L2(Cell *cell, uint32_t n_in, uint32_t n_rec, uint32_t n_out, float* losses_l2)
{
    int i,j;
    float sum2 = 0;

    for (i = 0; i < n_in; ++i)
        for (j = 0; j < n_rec; ++j)
            sum2 += cell->w_in[i][j] * cell->w_in[i][j];

    for (i = 0; i < n_rec; ++i)
        for (j = 0; j < n_rec; ++j)
            sum2 += cell->w_rec[i][j] * cell->w_rec[i][j];

    for (i = 0; i < n_rec; ++i)
        for (j = 0; j < n_out; ++j)
            sum2 += cell->w_out[i][j] * cell->w_out[i][j];

    *losses_l2 = sum2;
}

// Compute L2 regularization effect on weights
void grad_L2(Cell *cell, uint32_t n_in, uint32_t n_rec, uint32_t n_out)
{
    int i,j;

    for (i = 0; i < n_in; ++i)
        for (j = 0; j < n_rec; ++j)
            cell->grad_in[i][j] +=  2 * l2 * cell->w_in[i][j];

    for (i = 0; i < n_rec; ++i)
        for (j = 0; j < n_rec; ++j)
            cell->grad_rec[i][j] += 2 * l2 * cell->w_rec[i][j];

    for (i = 0; i < n_rec; ++i)
        for (j = 0; j < n_out; ++j)
            cell->grad_out[i][j] += 2 * l2 * cell->w_out[i][j];    
}

// Computes dropout
void dropout(int r, float* x, float keep_prob, float* result)
{
    /*
    For each element of `x`, with probability `rate=1- keep_prob`, outputs `0`, and otherwise
    scales up the input by `1 / (1-rate)`. The scaling is such that the expected sum is unchanged.
    By default, each element is kept or dropped independently.
    */
    int i;
    float random_tensor[r];
    float rate;
    float scale_in;

    for(i = 0; i < r; ++i)
        random_tensor[i] = rand() / (float)(RAND_MAX);

    rate = 1- keep_prob;
    scale_in = 1 / keep_prob;

    for(i = 0; i < r; ++i)
        if (keep_prob == 1)
            result[i] = x[i];
        else if (keep_prob < 0)
            result[i] = x[i];
        else if (random_tensor[i] >= rate)
            result[i] = x[i] * scale_in;
        else
            result[i] = 0;
}

// Generate random array
void rand_array(int r, float* random_tensor)
{
    int i;

    for(i = 0; i < r; ++i)
        random_tensor[i] = rand() / (float)(RAND_MAX);
}

// Computes dropout
void dropout_v2(int r, float* x, float* random_tensor, float keep_prob, float* result)
{
    /*
    For each element of `x`, with probability `rate=1- keep_prob`, outputs `0`, and otherwise
    scales up the input by `1 / (1-rate)`. The scaling is such that the expected sum is unchanged.
    By default, each element is kept or dropped independently.
    */
    int i;
    float rate;
    float scale_in;

    rate = 1- keep_prob;
    scale_in = 1 / keep_prob;

    if (keep_prob == 1 || keep_prob < 0)    // No effect
    {
        for(i = 0; i < r; ++i)
            result[i] = x[i];
    }
    else
    {   for(i = 0; i < r; ++i)
            if (random_tensor[i] >= rate)
                result[i] = x[i] * scale_in;
            else
                result[i] = 0;
    }
}

// Computes dropout: All time steps share the dame discountinuity (drop)
void dropout2D(int ts, int c, float x[][c], float keep_prob, float result[][c])
{
    /*
    For each element of `x`, with probability `rate=1- keep_prob`, outputs `0`, and otherwise
    scales up the input by `1 / (1-rate)`. The scaling is such that the expected sum is unchanged.
    By default, each element is kept or dropped independently.
    */
    int i, j;
    // float r;
    float random_tensor[c];
    float rate;
    float scale_in;

    for(i = 0; i < c; ++i)
        random_tensor[i] = rand() / (float)(RAND_MAX);

    rate = 1- keep_prob;
    scale_in = 1 / keep_prob;

    // for(i = 0; i < ts; ++i)
    //     for (j = 0; j < c; ++j)
    //         if (keep_prob == 1)         // No effect
    //             result[i][j] = x[i][j];
    //         else if (keep_prob < 0)     // No effect
    //             result[i][j] = x[i][j];
    //         else if (random_tensor[j] >= rate)
    //             result[i][j] = x[i][j] * scale_in;
    //         else
    //             result[i][j] = 0;

    if (keep_prob == 1 || keep_prob < 0)    // No effect
    {
        for(i = 0; i < ts; ++i)
            for (j = 0; j < c; ++j)
                result[i][j] = x[i][j];
    }
    else
    {
        for (j = 0; j < c; ++j)
            if (random_tensor[j] >= rate)
                for(i = 0; i < ts; ++i)
                    result[i][j] = x[i][j] * scale_in;
            else
                for(i = 0; i < ts; ++i)
                    result[i][j] = 0;
    }

    
}

// Computes dropout: All time steps share the dame discontinuity (drop)
void dropout2D_v2(int ts, int c, float x[][c], float* random_tensor, float keep_prob, float result[][c])
{
    /*
    For each element of `x`, with probability `rate=1- keep_prob`, outputs `0`, and otherwise
    scales up the input by `1 / (1-rate)`. The scaling is such that the expected sum is unchanged.
    By default, each element is kept or dropped independently.
    */
    int i, j;
    float rate;
    float scale_in;

    rate = 1- keep_prob;
    scale_in = 1 / keep_prob;

    if (keep_prob == 1 || keep_prob < 0)    // No effect
    {
        for(i = 0; i < ts; ++i)
            for (j = 0; j < c; ++j)
                result[i][j] = x[i][j];
    }
    else
    {
        for (j = 0; j < c; ++j)
            if (random_tensor[j] >= rate)
                for(i = 0; i < ts; ++i)
                    result[i][j] = x[i][j] * scale_in;
            else
                for(i = 0; i < ts; ++i)
                    result[i][j] = 0;
    }

    
}


int read_features(int r, int c, int d, char *file_path, float result[][c][d])
{
	int i,j,k; 
	FILE *fp;

    fp = fopen(file_path, "r");

    if (!fp)
	{
        printf("Could not open the dataset features\n");
        return -1;
    }

    for(i=0; i<r; ++i) 
    {
        for(j=0; j<c; ++j) 
        {
            for(k=0; k<d; ++k)
            {
                fscanf(fp, "%f", &result[i][j][k]);
            }
        }
    }
    fclose(fp);
    return 0;
}

#define BUFFERSIZE1 ( Seq_len * N_in * N_ch * 10 )//10 = (8-byte float) + (1-byte space) + (1-byte extra)
// Line number start from 1
int read_nth_features(char *file_path, int line, int r, int c, float result[][c])
{
    char buf[BUFFERSIZE1];
    int lines = 0;
    int i,j;
    int n = 0;
    FILE *fp;

    fp = fopen(file_path, "r");

    if (!fp) 
    {
        printf("Could not open the dataset features\n");
        return -1;
    }

    do 
    {
        if (++lines == line) 
        {
            fgets(buf, sizeof buf, fp);

            break;
        }
    }while((fscanf(fp, "%*[^\n]"), fscanf(fp, "%*c")) != EOF);

    char *p = buf;
    
    for(i=0; i<r; ++i) 
        for(j=0; j<c; ++j) 
        {
            sscanf(p, "%f %n", &result[i][j], &n);
            p += n;
        }
  
    fclose(fp);
    return 0;
}

int read_target(int r, char *file_path, int* result)
{
	int i; 
	FILE *fp;
	
    fp = fopen(file_path, "r");

    if (!fp) 
    {
        printf("Could not open the dataset target\n");
        return -1;
    }

    for(i=0; i<N_train; ++i) 
    {
        fscanf(fp, "%d", &result[i]);
    }
    fclose(fp);
    return 0;
}

#define BUFFERSIZE2 ( Seq_len * 10 )//10 = (8-byte float) + (1-byte space) + (1-byte extra)
int read_nth_target(char* file_path, int line, int* result)
{
    char buf[BUFFERSIZE2];
    int lines = 0;
    int n = 0;
    FILE *fp;

    fp = fopen(file_path, "r");

    if (!fp) 
    {
        printf("Could not open the dataset features\n");
        return -1;
    }

    do 
    {
        if (++lines == line) 
        {
            fgets(buf, sizeof buf, fp);

            break;
        }
    }while((fscanf(fp, "%*[^\n]"), fscanf(fp, "%*c")) != EOF);

    char *p = buf;
    
    sscanf(p, "%d %n", result, &n);
  
    fclose(fp);
    return 0;
}


#define BUFFERSIZE ( 101 * 120 * 10 )
int get_nth_line(char *file_path, int line, int r, int c, float result[][c])
{
    char buf[BUFFERSIZE];
    int lines = 0;
    int i,j;
    int n = 0;
    FILE *fp;

    fp = fopen(file_path, "r");

    if (!fp) 
    {
        printf("Could not open the dataset features\n");
        return -1;
    }

    do 
    {
        if (++lines == line) 
        {
            fgets(buf, sizeof buf, fp);

            break;
        }

    }while((fscanf(fp, "%*[^\n]"), fscanf(fp, "%*c")) != EOF);

    char *p = buf;
    
    for(i=0; i<r; ++i) 
        for(j=0; j<c; ++j) 
        {
            sscanf(p, "%f %n", &result[i][j], &n);
            p += n;
        }
  
    fclose(fp);
    return 0;
}

