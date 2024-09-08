#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/**************************************************
 * STRUCTURE TO CONTAIN THE NETWORK ARCHITECHTURE *
 **************************************************/
typedef struct {
    // Epochs
    int epochs;

    // Hyper-parameters
    double learning_rate;
    double annealing_rate;

    // Sizes
    int i_size;
    int h_size;
    int hh_size;
    int hhh_size;
    int o_size;

    // Values
    double *input;
    double *hidden;
    double *hidden2;
    double *hidden3;
    double *output;
    double *targets;

    // Weights
    double **weights_ih;
    double **weights_hh;
    double **weights_hhh;
    double **weights_ho;

    // Biases
    double *bias_h;
    double *bias_hh;
    double *bias_hhh;
    double *bias_o;

    // Name of the model
    char *model;
} architechture;

// Function pointers
typedef double (*hiddenfunc1ptr)  (double);
typedef double (*hiddenfunc2ptr)  (double);
typedef double (*hiddenfunc3ptr)  (double);
typedef double (*outputfuncptr)   (double);

typedef double (*dhiddenfunc1ptr) (double);
typedef double (*dhiddenfunc2ptr) (double);
typedef double (*dhiddenfunc3ptr) (double);
typedef double (*doutputfuncptr)  (double);

typedef double (*errorptr)        (double, double);
typedef double (*derrorptr)       (double, double);

/*******************
 * ERROR FUNCTIONS *
 *******************/
// HUBER LOSS
double huber (double prediction, double target) {
    double delta = 1.35;
    if (fabs(prediction-target) <= delta) {
        return 0.5 * (prediction-target) * (prediction-target);
    }
    return (delta * fabs(prediction-target) - 0.5 * (delta * delta));
}
double dhuber (double prediction, double target) {
    double delta = 1.35;
    if (fabs(prediction-target) < delta) {
        return prediction-target;
    }
    // Negatives
    if (prediction-target < 0.0) return -delta;
    // Positives
    else if (prediction-target > 0.0) return delta;
    // Zero
    else return 0.0;
}

//MSE
double mse (double prediction, double target) {
    return (prediction - target) * (prediction - target);
}

double dmse (double prediction, double target) {
    return 2.0 * (prediction - target);
}

/************************
 * ACTIVATION FUNCTIONS *
 ************************/
// Sigmoid
double sigmoid ( double a ) {
    return 1.0 / (1.0 + expf(-a));
}
double dsigmoid ( double a ) {
    return sigmoid(a) * (1.0 - sigmoid(a));
}

// Tanh
double dtanh ( double a ) {
    return 1.0 - (tanh(a) * tanh(a));
}

// RELU
double relu ( double a ) {
    if (a < 0.0) return 0.0;
    else return a;
}
double drelu ( double a ) {
    if (a < 0.0) return 0.0;
    else return 1.0;
}

// Leaky RELU
double lrelu ( double a ) {
    if (a < 0.0) return 0.01 * a;
    else return a;
}
double dlrelu ( double a ) {
    if (a < 0.0) return a;
    else return 1;
}

// GELU
double gelu ( double a ) {
    return 0.5 * a * (1.0 + erf(a / sqrt(2.0)));
}
double dgelu ( double a ) {
    double phi = 0.5 * (1.0 + erf(a/sqrt(2.0)));
    double exp = expf(-a * a / 2.0);
    return phi + (a * exp) / (sqrt(2.0 * M_PI));
}

// Softplus
double softplus ( double a ) {
    return log(1.0 + expf(a));
}
double dsoftplus ( double a ) {
    return 1.0 / (1.0 + expf(-a));
}

/******************************
 * SELECT YOUR FUNCTIONS HERE *
 ******************************/
// Activation functions
hiddenfunc1ptr hiddenfunc1 = gelu;
hiddenfunc2ptr hiddenfunc2 = gelu;
hiddenfunc3ptr hiddenfunc3 = gelu;
outputfuncptr  outputfunc  = sigmoid;

// Their derivatives
dhiddenfunc1ptr dhiddenfunc1 = dgelu;
dhiddenfunc2ptr dhiddenfunc2 = dgelu;
dhiddenfunc3ptr dhiddenfunc3 = dgelu;
doutputfuncptr  doutputfunc  = dsigmoid;

// Error function and its derivative
//errorptr error = huber;
//derrorptr derror = dhuber;
errorptr error = mse;
derrorptr derror = dmse;

/***************************
 * MISCELLANEOUS FUNCTIONS *
 ***************************/
// Return a random int in a specified range
int randrange (int min, int max) {
    return rand() % (max-min+1) + min;
}

// Get a float between -0.5 and 0.5
double frand () {
    return ((double) rand() / (double) RAND_MAX) - 0.5;
}

/***********************
 * FORWARD PROPAGATION *
 ***********************/
void forwardpropagation (architechture network) {
    // Calculate hidden layer values
    for (int i = 0; i < network.h_size; i++) {
        network.hidden[i] = 0;
        for (int j = 0; j < network.i_size; j++) {
            network.hidden[i] += network.input[j] * network.weights_ih[j][i];
        }
        network.hidden[i] += network.bias_h[i];
        network.hidden[i] = hiddenfunc1 (network.hidden[i]);
    }

    // Calculate second hidden layer values
    for (int i = 0; i < network.hh_size; i++) {
        network.hidden2[i] = 0;
        for (int j = 0; j < network.h_size; j++) {
            network.hidden2[i] += network.hidden[j] * network.weights_hh[j][i];
        }
        network.hidden[i] += network.bias_hh[i];
        network.hidden2[i] = hiddenfunc2 (network.hidden2[i]);
    }

    // Calculate third hidden layer values
    for (int i = 0; i < network.hhh_size; i++) {
        network.hidden3[i] = 0;
        for (int j = 0; j < network.hh_size; j++) {
            network.hidden3[i] += network.hidden2[j] * network.weights_hhh[j][i];
        }
        network.hidden3[i] += network.bias_hhh[i];
        network.hidden3[i] = hiddenfunc3 (network.hidden3[i]);
    }

    // Calculate output layer values
    for (int i = 0; i < network.o_size; i++) {
        network.output[i] = 0;
        for (int j = 0; j < network.hhh_size; j++) {
            network.output[i] += network.hidden3[j] * network.weights_ho[j][i];
        }
        network.output[i] += network.bias_o[i];
        network.output[i] = outputfunc (network.output[i]);
    }
}

/********************
 * BACK PROPAGATION *
 ********************/
double backpropagation (architechture network) {

    // Error to be reported
    double output_error =    0;
    double output_gradients  [network.o_size];

    double hidden3_errors    [network.hhh_size];
    double hidden3_gradients [network.hhh_size];

    double hidden2_errors    [network.hh_size];
    double hidden2_gradients [network.hh_size];

    double hidden_errors     [network.h_size];
    double hidden_gradients  [network.h_size];

    for (int i = 0; i < network.o_size; i++) {
        // Store error of this iteration
        output_error += error (network.targets[i], network.output[i]);

        output_gradients[i] = derror (network.targets[i], network.output[i]) * doutputfunc (network.output[i]);
        network.bias_o[i] += network.learning_rate * output_gradients[i];
    }

    for (int i = 0; i < network.hhh_size; i++) {
        hidden3_errors[i] = 0;
        for (int j = 0; j < network.o_size; j++) {
            network.weights_ho[i][j] += network.learning_rate * output_gradients[j] * network.hidden3[i];
            hidden3_errors[i] += output_gradients[j] * network.weights_ho[i][j];
        }
        hidden3_gradients[i] = hidden3_errors[i] * dhiddenfunc3(network.hidden3[i]);
        network.bias_hhh[i] += network.learning_rate * hidden3_gradients[i];
    }

    for (int i = 0; i < network.hh_size; i++) {
        hidden2_errors[i] = 0;
        for (int j = 0; j < network.hhh_size; j++) {
            network.weights_hhh[i][j] += network.learning_rate * hidden3_gradients[j] * network.hidden2[i];
            hidden2_errors[i] += hidden3_gradients[j] * network.weights_hhh[i][j];
        }
        hidden2_gradients[i] = hidden2_errors[i] * dhiddenfunc2(network.hidden2[i]);
        network.bias_hh[i] += network.learning_rate * hidden2_gradients[i];
    }

    for (int i = 0; i < network.h_size; i++) {
        hidden_errors[i] = 0;
        for (int j = 0; j < network.hh_size; j++) {
            network.weights_hh[i][j] += network.learning_rate * hidden2_gradients[j] * network.hidden[i];
            hidden_errors[i] += hidden2_gradients[j] * network.weights_hh[i][j];
        }
        hidden_gradients[i] = hidden_errors[i] * dhiddenfunc1(network.hidden[i]);
        network.bias_h[i] += network.learning_rate * hidden_gradients[i];
    }

    for (int i = 0; i < network.i_size; i++) {
        for (int j = 0; j < network.h_size; j++) {
            network.weights_ih[i][j] += network.learning_rate * hidden_gradients[j] * network.input[i];
        }
    }

    // Return the average error of this iteration
    return output_error / network.o_size;
}

/********************************
 * RANDOMIZE WEIGHTS AND BIASES *
 ********************************/
void randomizer (architechture network) {

    for (int i = 0; i < network.i_size; i++) {
        for (int j = 0; j < network.h_size; j++) {
           network.weights_ih[i][j] = frand();
        }
    }

    for (int i = 0; i < network.h_size; i++) {
        for (int j = 0; j < network.hh_size; j++) {
            network.weights_hh[i][j] = frand();
            network.bias_h[i] = frand();
        }
    }

    for (int i = 0; i < network.hh_size; i++) {
        for (int j = 0; j < network.hhh_size; j++) {
            network.weights_hhh[i][j] = frand();
            network.bias_hh[i] = frand();
        }
    }

    for (int i = 0; i < network.hhh_size; i++) {
        for (int j = 0; j < network.o_size; j++) {
            network.weights_ho[i][j] = frand();
            network.bias_hhh[i] = frand();
        }
    }

    for (int i = 0; i < network.o_size; i++) {
        network.bias_o[i] = frand();
    }
}

// Save weights and biases
void savemodel (architechture network) {
    FILE *saved_model = fopen (network.model, "w");

    // Header of the file
    fprintf (saved_model, "%d %d %d %d %d\n", network.i_size, network.h_size, network.hh_size, network.hhh_size, network.o_size);

    for (int i = 0; i < network.i_size; i++) {
        for (int j = 0; j < network.h_size; j++)  {
            fprintf (saved_model, "%lf\n", network.weights_ih[i][j]);
        }
    }
    for (int i = 0; i < network.h_size; i++) {
        for (int j = 0; j < network.hh_size; j++) {
            fprintf (saved_model, "%lf\n", network.weights_hh[i][j]);
            fprintf (saved_model, "%lf\n", network.bias_h[i]);
        }
    }
    for (int i = 0; i < network.hh_size; i++) {
        for (int j = 0; j < network.hhh_size; j++) {
            fprintf (saved_model, "%lf\n", network.weights_hhh[i][j]);
            fprintf (saved_model, "%lf\n", network.bias_hh[i]);
        }
    }
    for (int i = 0; i < network.hhh_size; i++) {
        for (int j = 0; j < network.o_size; j++) {
            fprintf (saved_model, "%lf\n", network.weights_ho[i][j]);
            fprintf (saved_model, "%lf\n", network.bias_hhh[i]);
        }
    }
    for (int i = 0; i < network.o_size; i++) {
        fprintf (saved_model, "%lf\n", network.bias_o[i]);
    }

    fclose (saved_model);
}

/***********************
 * READ EXISTING MODEL *
 ***********************/
void readmodel (architechture network) {

    FILE *loaded_model = fopen (network.model, "r");

    if (!loaded_model) {
        printf ("Error loading model %s.\n", network.model);
    }

    else {
        //int network.i_size, network.h_size, network.hh_size, network.hhh_size, network.o_size;

        // Load header
        fscanf (loaded_model, "%d %d %d %d %d\n", &network.i_size, &network.h_size, &network.hh_size, &network.hhh_size, &network.o_size);

        // Load weights/biases
         for (int i = 0; i < network.i_size; i++) {
            for (int j = 0; j < network.h_size; j++)  {
                fscanf (loaded_model, "%lf\n", &network.weights_ih[i][j]);
            }
         }
         for (int i = 0; i < network.h_size; i++) {
             for (int j = 0; j < network.hh_size; j++) {
                 fscanf (loaded_model, "%lf\n", &network.weights_hh[i][j]);
                 fscanf (loaded_model, "%lf\n", &network.bias_h[i]);
             }
         }
         for (int i = 0; i < network.hh_size; i++) {
             for (int j = 0; j < network.hhh_size; j++) {
                 fscanf (loaded_model, "%lf\n", &network.weights_hhh[i][j]);
                 fscanf (loaded_model, "%lf\n", &network.bias_hh[i]);
             }
         }
         for (int i = 0; i < network.hhh_size; i++) {
             for (int j = 0; j < network.o_size; j++) {
                 fscanf (loaded_model, "%lf\n", &network.weights_ho[i][j]);
                 fscanf (loaded_model, "%lf\n", &network.bias_hhh[i]);
             }
         }
         for (int i = 0; i < network.o_size; i++) {
             fscanf (loaded_model, "%lf\n", &network.bias_o[i]);
         }
         fclose (loaded_model);
    }
}

/********
 * MAIN *
 ********/
int main (int argc, char **argv) {

    if (argc != 11) {
        puts ("Usage:\n./CCeptron\n\tdata.csv\n\tinput_size\n\thidden_size\n\thidden_size2\n\thidden_size3\n\toutput_size\n\tepochs\n\tlearning_rate\n\tannealing_rate\n\tsaved_model");
        return 1;
    }

    FILE *datafile = fopen (argv[1], "r");

    if (!datafile) {
        puts ("Data file error.");
        return 1;
    }

    // Seed random number generator
    srand(time(0) + getpid());

    architechture network;

    char *separator = ",";

    // Obtain hyperparameters
    network.i_size   = atoi (argv[2]);
    network.h_size   = atoi (argv[3]);
    network.hh_size  = atoi (argv[4]);
    network.hhh_size = atoi (argv[5]);
    network.o_size   = atoi (argv[6]);

    // Obtain parameters
    network.epochs         = atoi (argv[7]);
    network.learning_rate  = atof (argv[8]);
    network.annealing_rate = atof (argv[9]);

    // Name of the saved model
    network.model = argv[10];

    FILE *saved_model = fopen (network.model, "r");

    // Set this sufficient buffer limit to avoid realloc. Modify to suit needs
    int BUF_SIZE = (network.i_size + network.o_size) * 12;

    int rows = 0;

    // Allocate line depending on the input size
    char *line = malloc (sizeof(char) * BUF_SIZE);

    // Get line count and then reset position on file
    while ( fgets ( line, BUF_SIZE, datafile ) != NULL ) rows++;
    rewind (datafile);

    printf ("Rows: %d\n", rows);

    double **container = malloc (sizeof (double*) * rows);
    for (int row = 0; row<rows; row++) {
        container[row] = malloc(sizeof (double) * BUF_SIZE);
    }

    // Reset row counter
    rows -= rows;

    // Parse lines into the container
    while (fgets(line, BUF_SIZE, datafile)) {
        for (int i = 0; i < network.i_size + network.o_size; i++) {
            if (i == 0) container[rows][i] = atof( strtok (line, separator) );
            else container[rows][i] = atof( strtok ( NULL, separator ) );
        }
        rows++;
    }

    // Close file
    fclose (datafile);

    // Store classes separately from the container
    double targets[rows][network.o_size];

    for (int row = 0; row < rows; row++) {
        for (int class = 0; class < network.o_size; class++) {
            targets[row][class] = container[row][network.i_size + class];
        }
    }

    free(line);

    // Values
    double **input = malloc ( sizeof (double*) * rows );
    for (int i = 0; i < rows; i++) input[i] = malloc ( sizeof(double) * network.i_size );

    network.hidden = malloc (sizeof(double) * network.h_size);
    network.hidden2 = malloc (sizeof(double) * network.hh_size);
    network.hidden3 = malloc (sizeof(double) * network.hhh_size);
    network.output = malloc (sizeof(double) * network.o_size);

    // Weights
    network.weights_ih = malloc(sizeof(double*) * network.i_size);
    for (int i = 0; i < network.i_size; i++) network.weights_ih[i] = malloc(sizeof(double) * network.h_size);
    network.weights_hh = malloc(sizeof(double*) * network.h_size);
    for (int i = 0; i < network.h_size; i++) network.weights_hh[i] = malloc(sizeof(double) * network.hh_size);
    network.weights_hhh = malloc(sizeof(double*) * network.hh_size);
    for (int i = 0; i < network.hh_size; i++) network.weights_hhh[i] = malloc(sizeof(double) * network.hhh_size);
    network.weights_ho = malloc(sizeof(double*) * network.hhh_size);
    for (int i = 0; i < network.hhh_size; i++) network.weights_ho[i] = malloc(sizeof(double) * network.o_size);

    network.bias_h = malloc(sizeof(double) * network.h_size);
    network.bias_hh = malloc(sizeof(double) * network.hh_size);
    network.bias_hhh = malloc(sizeof(double) * network.hhh_size);
    network.bias_o = malloc(sizeof(double) * network.o_size);

    // Train and generate weights/biases if a saved model doesn't exist
    if (!saved_model) {

        /********************************
         * RANDOMIZE WEIGHTS AND BIASES *
         ********************************/
        randomizer (network);

        /************
         * TRAINING *
         ************/
        printf ("Training for %d epochs.\n", network.epochs);

        double iteration_error[network.epochs];

        FILE *saved_errors = fopen ("savederrors", "w");

        for (int epoch = 0; epoch < network.epochs; epoch++) {
            for (int row = 0; row < rows; row++) {

                // Pick random row to train on
                int selected_row = randrange(0, rows-1);

                for (int j = 0; j < network.i_size; j++) {
                    // The random row to train on
                    input[row][j] = container[selected_row][j];
                }

                network.input = input[row];

                network.targets = targets[selected_row];

                // Forward propagation
                forwardpropagation(network);

                // Store error
                iteration_error [epoch] = backpropagation(network);
            }

            fprintf (saved_errors, "%lf\n", iteration_error[epoch]);

            //Report error every 20 epochs
            if (epoch % 20 == 0) {
                printf("\rEpoch %d/%d -- Error: %.6lf, Rate: %f", epoch, network.epochs, iteration_error[epoch], network.learning_rate);
                fflush(stdout);
            }

            // Progressively lowering the learning rate
            network.learning_rate *= network.annealing_rate;
        }

        fclose (saved_errors);

        /***************************
         * SAVE WEIGHTS AND BIASES *
         ***************************/
        savemodel(network);
    }

    // Load weights and biases if a saved model exists
    else {
        printf ("Reading weights and biases from %s.\n", argv[10]);
        readmodel (network);
    }

    /***********
     * TESTING *
     ***********/
    for (int row = 0; row < rows; row++) {

        // Pick random row to test
        int selected_row = randrange (0, rows-1);

        for (int j = 0; j < network.i_size; j++) {
            input[row][j] = container[selected_row][j];
        }
        
        network.input = input[row];
        network.targets = targets[selected_row];

        // Forwardpropagation: testing
        forwardpropagation(network);

        printf("\n");
        for (int i = 0; i < network.o_size; i++) {
            printf("Output: %.4lf Target: %.4lf: %.2f%%\n", network.output[i], targets[selected_row][i], 100 - fabs(network.output[i] - targets[selected_row][i])*100);
        }
        printf ("------\n");
    }

    /***************
     * FREE MEMORY *
     ***************/
    for (int row = 0; row < rows; row++) {
        free(container[row]);
        free(input[row]);
    }
    free (container);
    free (input);

    for (int i = 0; i < network.i_size; i++) free(network.weights_ih[i]);
    free(network.weights_ih);
    for (int i = 0; i < network.h_size; i++) free(network.weights_hh[i]);
    free(network.weights_hh);
    for (int i = 0; i < network.hh_size; i++) free(network.weights_hhh[i]);
    free(network.weights_hhh);
    for (int i = 0; i < network.hhh_size; i++) free(network.weights_ho[i]);
    free(network.weights_ho);

    free(network.hidden);
    free(network.hidden2);
    free(network.hidden3);
    free(network.output);

    free(network.bias_h);
    free(network.bias_hh);
    free(network.bias_hhh);
    free(network.bias_o);

    return 0;
}
