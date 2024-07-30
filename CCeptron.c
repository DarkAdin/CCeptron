#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE, EPOCHS;
float LEARNING_RATE, ANNEALING_RATE;
char *SAVED_MODEL;

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
    //else {
    // Negatives
    if (prediction-target < 0.0) return -delta;
    // Positives
    else if (prediction-target > 0.0) return delta;
    // Zero
    else return 0.0;
    //}
}

//MSE
double mse (double prediction, double target) {
    return (prediction - target) * (prediction - target);
}

double dmse (double prediction, double target) {
    return (prediction - target);
}

/************************
 * ACTIVATION FUNCTIONS *
 ************************/
// Sigmoid
double sigmoid ( double a ) {
    return 1.0 / (1.0 + expf(-a));
}
double dsigmoid ( double a ) {
    //return a * (1 - a);
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
hiddenfunc3ptr hiddenfunc3 = relu;
outputfuncptr  outputfunc  = sigmoid;

// Their derivatives
dhiddenfunc1ptr dhiddenfunc1 = dgelu;
dhiddenfunc2ptr dhiddenfunc2 = dgelu;
dhiddenfunc3ptr dhiddenfunc3 = drelu;
doutputfuncptr  doutputfunc  = dsigmoid;

// Error function and its derivative
errorptr error = huber;
derrorptr derror = dhuber;

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
void forwardpropagation (double input[INPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double hidden[HIDDEN_SIZE], double hidden2[HIDDEN_SIZE2], double hidden3[HIDDEN_SIZE3], double output[OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    // Calculate hidden layer values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * weights_ih[j][i];
        }
        hidden[i] = hiddenfunc1 (hidden[i] + bias_h[i]);
    }

    // Calculate hidden2 layer values
    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden2[i] += hidden[j] * weights_hh[j][i];
        }
        hidden2[i] = hiddenfunc2 (hidden2[i] + bias_hh[i]);
    }

    // Calculate hidden3 layer values
    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        hidden3[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            hidden3[i] += hidden2[j] * weights_hhh[j][i];
        }
        hidden3[i] = hiddenfunc3 (hidden3[i] + bias_hhh[i]);
    }

    // Calculate output layer values
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            output[i] += hidden3[j] * weights_ho[j][i];
        }
        output[i] = outputfunc (output[i] + bias_o[i]);
    }
}

/********************
 * BACK PROPAGATION *
 ********************/
double backpropagation (double input[INPUT_SIZE], double hidden[HIDDEN_SIZE], double hidden2[HIDDEN_SIZE2], double hidden3[HIDDEN_SIZE3], double output[OUTPUT_SIZE], double target[OUTPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    double output_error =    0;			        // Error to be reported
    double output_gradients  [OUTPUT_SIZE];

    double hidden3_errors    [HIDDEN_SIZE3];
    double hidden3_gradients [HIDDEN_SIZE3];

    double hidden2_errors    [HIDDEN_SIZE2];
    double hidden2_gradients [HIDDEN_SIZE2];

    double hidden_errors     [HIDDEN_SIZE];
    double hidden_gradients  [HIDDEN_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error += error (target[i], output[i]); // Store error of this iteration

        output_gradients[i] = derror (target[i], output[i]) * doutputfunc (output[i]);
        bias_o[i] += LEARNING_RATE * output_gradients[i];
    }

    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        hidden3_errors[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] += LEARNING_RATE * output_gradients[j] * hidden3[i];
            hidden3_errors[i] += output_gradients[j] * weights_ho[i][j];
        }
        hidden3_gradients[i] = hidden3_errors[i] * dhiddenfunc3(hidden3[i]);
        bias_hhh[i] += LEARNING_RATE * hidden3_gradients[i];
    }

    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2_errors[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            weights_hhh[i][j] += LEARNING_RATE * hidden3_gradients[j] * hidden2[i];
            hidden2_errors[i] += hidden3_gradients[j] * weights_hhh[i][j];
        }
        hidden2_gradients[i] = hidden2_errors[i] * dhiddenfunc2(hidden2[i]);
        bias_hh[i] += LEARNING_RATE * hidden2_gradients[i];
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_errors[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            weights_hh[i][j] += LEARNING_RATE * hidden2_gradients[j] * hidden[i];
            hidden_errors[i] += hidden2_gradients[j] * weights_hh[i][j];
        }
        hidden_gradients[i] = hidden_errors[i] * dhiddenfunc1(hidden[i]);
        bias_h[i] += LEARNING_RATE * hidden_gradients[i];
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] += LEARNING_RATE * hidden_gradients[j] * input[i];
        }
    }

    // Return the average error of this iteration
    return output_error / OUTPUT_SIZE;
}

/********************************
 * RANDOMIZE WEIGHTS AND BIASES *
 ********************************/
void randomizer (double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] = frand();
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            weights_hh[i][j] = frand();
            bias_h[i] = 0;
        }
    }

    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            weights_hhh[i][j] = frand();
            bias_hh[i] = 0;
        }
    }

    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] = frand();
            bias_hhh[i] = 0;
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias_o[i] = 0;
    }
}

// Save weights and biases
void savemodel (char *model, double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    FILE *saved_model = fopen (model, "w");

    // Header of the file
    fprintf (saved_model, "%d %d %d %d %d\n", INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE);

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++)  {
            fprintf (saved_model, "%lf\n", weights_ih[i][j]);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            fprintf (saved_model, "%lf\n", weights_hh[i][j]);
            fprintf (saved_model, "%lf\n", bias_h[i]);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            fprintf (saved_model, "%lf\n", weights_hhh[i][j]);
            fprintf (saved_model, "%lf\n", bias_hh[i]);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            fprintf (saved_model, "%lf\n", weights_ho[i][j]);
            fprintf (saved_model, "%lf\n", bias_hhh[i]);
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        fprintf (saved_model, "%lf\n", bias_o[i]);
    }

    fclose (saved_model);
}

/***********************
 * READ EXISTING MODEL *
 ***********************/
void readmodel (char *model, double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    FILE *loaded_model = fopen (model, "r");

    if (!loaded_model) {
        printf ("Error loading model %s.\n", model);
    }

    else {
        int INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE;

        // Load header
        fscanf (loaded_model, "%d %d %d %d %d\n", &INPUT_SIZE, &HIDDEN_SIZE, &HIDDEN_SIZE2, &HIDDEN_SIZE3, &OUTPUT_SIZE);

        // Load weights/biases
         for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++)  {
                fscanf (loaded_model, "%lf\n", &weights_ih[i][j]);
            }
         }
         for (int i = 0; i < HIDDEN_SIZE; i++) {
             for (int j = 0; j < HIDDEN_SIZE2; j++) {
                 fscanf (loaded_model, "%lf\n", &weights_hh[i][j]);
                 fscanf (loaded_model, "%lf\n", &bias_h[i]);
             }
         }
         for (int i = 0; i < HIDDEN_SIZE2; i++) {
             for (int j = 0; j < HIDDEN_SIZE3; j++) {
                 fscanf (loaded_model, "%lf\n", &weights_hhh[i][j]);
                 fscanf (loaded_model, "%lf\n", &bias_hh[i]);
             }
         }
         for (int i = 0; i < HIDDEN_SIZE3; i++) {
             for (int j = 0; j < OUTPUT_SIZE; j++) {
                 fscanf (loaded_model, "%lf\n", &weights_ho[i][j]);
                 fscanf (loaded_model, "%lf\n", &bias_hhh[i]);
             }
         }
         for (int i = 0; i < OUTPUT_SIZE; i++) {
             fscanf (loaded_model, "%lf\n", &bias_o[i]);
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

    FILE *saved_model = fopen (argv[10], "r");

    // Seed random number generator
    srand(time(0) + getpid());

    // Obtain hyperparameters
    INPUT_SIZE     = atoi (argv[2]);
    HIDDEN_SIZE    = atoi (argv[3]);
    HIDDEN_SIZE2   = atoi (argv[4]);
    HIDDEN_SIZE3   = atoi (argv[5]);
    OUTPUT_SIZE    = atoi (argv[6]);
    EPOCHS         = atoi (argv[7]);
    LEARNING_RATE  = atof (argv[8]);
    ANNEALING_RATE = atof (argv[9]);
    SAVED_MODEL    = argv[10];

    // Set this sufficient buffer limit to avoid realloc. Modify to suit needs
    int BUF_SIZE = (INPUT_SIZE + OUTPUT_SIZE) * 12;

    int rows = 0;

    // Allocate line depending on the input size
    char *line = malloc (sizeof(char) * BUF_SIZE);

    // Get line count and then reset position on file
    while (fgets(line, BUF_SIZE, datafile)!=NULL) rows++;
    rewind (datafile);

    printf ("Rows: %d\n", rows);

    //float container[rows][BUF_SIZE];
    float **container = malloc (sizeof (float*) * rows);
    for (int row = 0; row<rows; row++) {
        container[row] = malloc(sizeof (float) * BUF_SIZE);
    }

    // Reset row counter
    rows -= rows;

    // Parse lines into the container
    while (fgets(line, BUF_SIZE, datafile)) {
        for (int i = 0; i<INPUT_SIZE+OUTPUT_SIZE; i++) {
            if (i == 0) container[rows][i] = atof(strtok(line,","));
            else container[rows][i] = atof(strtok(NULL,","));
        }
        rows++;
    }

    // Close file
    fclose (datafile);

    // Store classes separately from the container
    double targets[rows][OUTPUT_SIZE];
    for (int row = 0; row < rows; row++) {
        for (int class = 0; class < OUTPUT_SIZE; class++) {
            targets[row][class] = container[row][INPUT_SIZE + class];
        }
    }

    free(line);

    // Values
    //double input[rows][INPUT_SIZE];
    double **input = malloc ( sizeof (double*) * rows );
    for (int i = 0; i < rows; i++) input[i] = malloc ( sizeof(double) * INPUT_SIZE );
    double hidden[HIDDEN_SIZE];
    double hidden2[HIDDEN_SIZE2];
    double hidden3[HIDDEN_SIZE3];
    double output[OUTPUT_SIZE];

    // Weights
    double weights_ih[INPUT_SIZE][HIDDEN_SIZE];
    double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2];
    double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3];
    double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE];

    // Biases
    double bias_h[HIDDEN_SIZE];
    double bias_hh[HIDDEN_SIZE2];
    double bias_hhh[HIDDEN_SIZE3];
    double bias_o[OUTPUT_SIZE];

    // Train and generate weights/biases if a saved model doesn't exist
    if (!saved_model) {

        /********************************
         * RANDOMIZE WEIGHTS AND BIASES *
         ********************************/
        randomizer (
            weights_ih, weights_hh, weights_hhh, weights_ho,
            bias_h, bias_hh, bias_hhh, bias_o
        );

        /************
         * TRAINING *
         ************/
        printf ("Training for %d epochs.\n", EPOCHS);

        double iteration_error[EPOCHS];

        FILE *saved_errors = fopen ("savederrors", "w");

        for (int epoch = 0; epoch < EPOCHS; epoch++) {

            for (int row = 0; row < rows; row++) {

                // Pick random row to train on
                int selected_row = randrange(0, rows-1);

                for (int j = 0; j < INPUT_SIZE; j++) {
                    // The random row to train on
                    input[row][j] = container[selected_row][j];
                }

                // Forward propagation
                forwardpropagation(
                    input[row],
                    weights_ih, weights_hh, weights_hhh, weights_ho,
                    hidden, hidden2, hidden3, output,
                    bias_h, bias_hh, bias_hhh, bias_o
                );

                // Store error
                iteration_error [epoch] = backpropagation(
                    input[row],
                    hidden, hidden2, hidden3,
                    output,
                    targets[selected_row],
                    weights_ih, weights_hh, weights_hhh, weights_ho,
                    bias_h, bias_hh, bias_hhh, bias_o
                );
            }

            fprintf (saved_errors, "%lf\n", iteration_error[epoch]);

            //Report error every 20 epochs
            if (epoch % 20 == 0) {
                printf("\rEpoch %d/%d -- Error: %.6lf, Rate: %f", epoch, EPOCHS, iteration_error[epoch], LEARNING_RATE);
                fflush(stdout);
            }

            // Progressively lowering the learning rate
            LEARNING_RATE *= ANNEALING_RATE;
        }

        fclose (saved_errors);

        /***************************
         * SAVE WEIGHTS AND BIASES *
         ***************************/
        savemodel(
            SAVED_MODEL,
            weights_ih, weights_hh, weights_hhh, weights_ho,
            bias_h, bias_hh, bias_hhh, bias_o
        );
    }

    // Load weights and biases if a saved model exists
    else {
        printf ("Reading weights and biases from %s.\n", argv[10]);
        readmodel (
            argv[10],
            weights_ih, weights_hh, weights_hhh, weights_ho,
            bias_h, bias_hh,bias_hhh, bias_o
        );
    }

    /***********
     * TESTING *
     ***********/
    for (int row = 0; row < rows; row++) {

        // Pick random row to test
        int selected_row = randrange (0, rows-1);

        for (int j = 0; j < INPUT_SIZE; j++) {
            input[row][j] = container[selected_row][j];
        }

        // Forwardpropagation: testing
        forwardpropagation(
            input[row],
            weights_ih, weights_hh, weights_hhh, weights_ho,
            hidden, hidden2, hidden3, output,
            bias_h, bias_hh, bias_hhh, bias_o
        );

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            printf("Output: %.4lf Target: %.4lf: %.2f%%\n", output[i], targets[selected_row][i], 100 - fabs(output[i] - targets[selected_row][i])*100);
        }
        printf ("------\n");
    }

    // Free memory
    for (int row = 0; row < rows; row++) {
        free(container[row]);
        free(input[row]);
    }
    free (container);
    free (input);

    return 0;
}
