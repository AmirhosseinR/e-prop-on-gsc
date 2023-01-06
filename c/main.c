#include "eprop_LSNN.h"


Cell  cell;
State state;
Adam adam;


#ifdef __LINUX__
float features_train[N_train][Seq_len][N_in*N_ch];
int   target_train[N_train];
float features_valid[N_valid][Seq_len][N_in*N_ch];
int   target_valid[N_valid];
#endif

float input_features[Seq_len][N_in*N_ch];
int   output_target;

float input_features1[Seq_len][N_in*N_ch];
int   output_target1;
float compare_in;
float compare_tarin;

char char_name[100];
char dataset_path_trn_ftr[100];
char dataset_path_trn_trg[100];
char dataset_path_val_ftr[100];
char dataset_path_val_trg[100];

int main()
{
    int i,j,k;
    int n_epoch;
    float lr;
    uint16_t print_every = 100;    //print statistics every K iterations
    int correct_num = 0;
    float loss_train = 0;
    float loss_valid = 0;
    float loss_test = 0;
    float epoch_loss, epoch_error;
    bool is_train = true;
    int line=0;
    int mini_batch_indices[N_train];

    srand ( time(NULL) ); // to prevent sequence repetition between runs  
    rd_permutation(N_train, mini_batch_indices);
    //------------------------------------------
    // Train dataset
    #ifndef __LINUX__
	strcpy(dataset_path_trn_ftr, "..\\files\\train\\features.dat");
    strcpy(dataset_path_trn_trg, "..\\files\\train\\target_outputs.dat");
    strcpy(dataset_path_val_ftr, "..\\files\\dev\\features.dat");
    strcpy(dataset_path_val_trg, "..\\files\\dev\\target_outputs.dat");
	#else
	strcpy(dataset_path_trn_ftr, "../files/train/features.dat");
    strcpy(dataset_path_trn_trg, "../files/train/target_outputs.dat");
    strcpy(dataset_path_val_ftr, "../files/dev/features.dat");
    strcpy(dataset_path_val_trg, "../files/dev/target_outputs.dat");
	#endif
    //------------------------------------------
    #ifdef __LINUX__
    read_features(N_train, Seq_len, N_in*N_ch, dataset_path_trn_ftr, features_train);
    read_target(N_train, dataset_path_trn_trg, target_train);
    read_features(N_valid, Seq_len, N_in*N_ch, dataset_path_val_ftr, features_valid);
    read_target(N_valid, dataset_path_val_trg, target_valid);
    #endif
    //------------------------------------------
    initialize_eprop_parameters(N_in*N_ch, N_regular, N_adaptive, N_out, Seq_len, &cell, &state);
    init_adam_param(N_in*N_ch, N_regular, N_adaptive, N_out, &adam);
    //------------------------------------------
    printf("\n***********************");
    for (n_epoch=0; n_epoch < N_epoch; ++n_epoch)
    {
        printf("\nn_epoch: %d", n_epoch);
        rd_permutation_arr(N_train, mini_batch_indices);
        loss_train = 0;
        if(n_epoch>0 && (n_epoch % lr_decay_every ==0))
        {
            printf("\n-------------------------");
            adam.learning_rate = lr_decay * adam.learning_rate;
            printf("\nNew learning rate: %f", adam.learning_rate);
            printf("\n-------------------------");
        }
        //------------------------------------------
        //           Trian the network
        //------------------------------------------
        correct_num = 0;
        is_train = true;
        // Read one batch at a time
        for (i = 0; i < N_train; ++i) 
        {
            // printf("\nn_iter: %d", i);
            if (shuffle)
                line = mini_batch_indices[i];
            else
                line = i+1;

            #ifndef __LINUX__
            read_nth_features(dataset_path_trn_ftr, line, Seq_len, N_in*N_ch, input_features);
            read_nth_target(dataset_path_trn_trg, line, &output_target);
            #else
            for(j=0; j<Seq_len; ++j) 
                for(k=0; k<N_in*N_ch; ++k) 
                    input_features[j][k] = features_train[line-1][j][k];
            output_target = target_train[line-1];
            #endif
        
            optimize_eprop(&cell, &state, &adam, N_batch, N_in*N_ch, N_rec, N_out, Seq_len, input_features, output_target, &correct_num, &loss_train, is_train, i+1, print_every);
        }
        epoch_loss = loss_train / (float)N_train;
        printf("\nloss train: %0.7f", epoch_loss);

        epoch_error = 1 - (correct_num / (float)N_train);
        printf("\nError train: %0.7f", epoch_error);
        //------------------------------------------
        //             Validate
        //------------------------------------------
        correct_num = 0;
        line = 0;
        is_train = false;
        loss_valid = 0;
        for (i = 0; i < N_valid; ++i) 
        {
            #ifndef __LINUX__
            line += 1;
            read_nth_features(dataset_path_val_ftr, line, Seq_len, N_in*N_ch, input_features);
            read_nth_target(dataset_path_val_trg, line, &output_target);
            #else
            for(j=0; j<Seq_len; ++j) 
                for(k=0; k<N_in*N_ch; ++k) 
                    input_features[j][k] = features_valid[i][j][k];
            
            output_target = target_valid[i];
            #endif

            optimize_eprop(&cell, &state, &adam, N_batch, N_in*N_ch, N_rec, N_out, Seq_len, input_features, output_target, &correct_num, &loss_valid, is_train, i, print_every);
            // printf("\nloss: %0.7f\n", loss/(i+1));
        }
        epoch_loss = loss_valid / (float)N_valid;
        printf("\nloss validate: %0.7f", epoch_loss);

        epoch_error = 1 - (correct_num / (float)N_valid);
        printf("\nError validate: %0.7f", epoch_error);
        printf("\n***********************");
    }
    return 0;

}