#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>

int INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE, EPOCHS;
float LEARNING_RATE, ANNEALING_RATE;

/* ERROR FUNCTIONS */
double huber (double prediction, double target) {
    double delta = 1.35;
    // HUBER LOSS
    if (fabs(prediction-target) < delta) return 0.5 * (prediction-target) * (prediction-target);
    else return (delta * fabs(prediction-target) - 0.5 * delta * delta);
}
double dhuber (double prediction, double target) {
    double delta = 1.35;
    // HUBER LOSS
    if (fabs(prediction-target) < delta) return prediction-target;
    else {
        // Negatives
        if (prediction-target < 0) return -delta;
        // Positives
        else if (prediction-target > 0) return delta;
        // Zero
        //else return 0.0;
        else return 0;
    }
}

/* MSE */
double mse (double prediction, double target) {
    return (prediction - target) * (prediction - target);
}

double dmse (double prediction, double target) {
    return (prediction - target);
}

/*----------------ACTIVATION FUNCTIONS---------------------*/
// Sigmoid
double sigmoid ( double a ) {
    return 1 / (1 + expf(-a));
}
double dsigmoid ( double a ) {
    return a * (1 - a);
}

// Tanh
double dtanh ( double a ) {
    return 1 - tanh(a) * tanh(a);
}

// RELU
double relu ( double a ) {
    if (a < 0) return 0;
    else return a;
}
double drelu ( double a ) {
    if (a < 0) return 0;
    else return 1;
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

// Softplus
double softplus ( double a ) {
    return log(1 + expf(a));
}
double dsoftplus ( double a ) {
    return 1 / (1 + expf(-a));
}
/*---------------------------------------------------------*/

/*---------------MAKE YOUR CHOICES HERE----------------------*/
typedef double (*hiddenfunc1ptr)(double);
typedef double (*hiddenfunc2ptr)(double);
typedef double (*hiddenfunc3ptr)(double);
typedef double (*outputfuncptr)(double);

typedef double (*dhiddenfunc1ptr)(double);
typedef double (*dhiddenfunc2ptr)(double);
typedef double (*dhiddenfunc3ptr)(double);
typedef double (*doutputfuncptr)(double);

typedef double (*errorptr)(double,double);
typedef double (*derrorptr)(double,double);

// DEFINE FUNCTIONS!
hiddenfunc1ptr hiddenfunc1 = lrelu;
hiddenfunc2ptr hiddenfunc2 = lrelu;
hiddenfunc3ptr hiddenfunc3 = lrelu;
outputfuncptr outputfunc = sigmoid;

dhiddenfunc1ptr dhiddenfunc1 = dlrelu;
dhiddenfunc2ptr dhiddenfunc2 = dlrelu;
dhiddenfunc3ptr dhiddenfunc3 = dlrelu;
doutputfuncptr  doutputfunc = dsigmoid;

errorptr error = huber;
derrorptr derror = dhuber;
/*---------------------------------------------------*/

/*---------------MISCELANEOUS FUNCTIONS--------------*/
// Return a random int in a specified range
int randrange (int min, int max) {
    return rand() % (max-min+1) + min;
}

// Get a float between -0.5 and 0.5
double frand () {
    return (double) rand() / (double) RAND_MAX - 0.5;
}
/*---------------------------------------------------*/

/*--------------FORWARD PROPAGATION-------------------*/
void forwardpropagation (double *input, double **weights_ih, double **weights_hh, double **weights_hhh, double **weights_ho, double *hidden, double *hidden2, double *hidden3, double *output, double *bias_h, double *bias_hh, double *bias_hhh, double *bias_o) {

    // Calculate hidden layer values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * weights_ih[j][i];
        }
        hidden[i] += bias_h[i];
        hidden[i] = hiddenfunc1(hidden[i]);
    }

    // Calculate hidden2 layer values
    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden2[i] += hidden[j] * weights_hh[j][i];
        }
        hidden2[i] += bias_hh[i];
        hidden2[i] = hiddenfunc2(hidden2[i]);
    }

    // Calculate hidden3 layer values
    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        hidden3[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            hidden3[i] += hidden2[j] * weights_hhh[j][i];
        }
        hidden3[i] += bias_hhh[i];
        hidden3[i] = hiddenfunc3(hidden3[i]);
    }

    // Calculate output layer values
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            output[i] += hidden3[j] * weights_ho[j][i];
        }
        output[i] += bias_o[i];
        output[i] = outputfunc(output[i]);
    }
}
/*----------------------------------------------------*/

/*-----------BACKPROPAGATION-----------------*/
double backpropagation (double *input, double *hidden, double *hidden2, double *hidden3, double *output, double *target, double **weights_ih, double **weights_hh, double **weights_hhh, double **weights_ho, double *bias_h, double *bias_hh, double *bias_hhh, double *bias_o) {

    double output_error =                 0;               // Error to be reported
    double output_gradients   [OUTPUT_SIZE];

    double hidden3_errors     [HIDDEN_SIZE3];
    double hidden3_gradients  [HIDDEN_SIZE3];

    double hidden2_errors     [HIDDEN_SIZE2];
    double hidden2_gradients  [HIDDEN_SIZE2];

    double hidden_errors      [HIDDEN_SIZE];
    double hidden_gradients   [HIDDEN_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error += error (target[i], output[i]);   // Store error of this iteration

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
/*-------------------------------------------*/

// Randomize weights and biases
void randomizer (double **weights_ih, double **weights_hh, double **weights_hhh, double **weights_ho, double *bias_h, double *bias_hh, double *bias_hhh, double *bias_o) {

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] = frand();
            weights_ih[i][j] = frand();
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            weights_hh[i][j] = frand();
            bias_h[i] = frand();
        }
    }

    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            weights_hhh[i][j] = frand();
            bias_hh[i] = frand();
        }
    }

    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] = frand();
            bias_hhh[i] = frand();
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias_o[i] = frand();
    }
}

// Save weights and biases
void savemodel (char *model, double **weights_ih, double **weights_hh, double **weights_hhh, double **weights_ho, double *bias_h, double *bias_hh, double *bias_hhh, double *bias_o) {

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

// Load weights and biases
void readmodel (char *model, double **weights_ih, double **weights_hh, double **weights_hhh, double **weights_ho, double *bias_h, double *bias_hh, double *bias_hhh, double *bias_o) {

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

/* MAIN */
int main (int argc, char **argv) {

    if (argc != 11) {
        puts ("./CCeptron file.csv input_size hidden_size hidden_size2 hidden_size3 output_size epochs learning_rate annealing_rate saved_model norm (optional)");
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
    INPUT_SIZE      = atoi (argv[2]);
    HIDDEN_SIZE     = atoi (argv[3]);
    HIDDEN_SIZE2    = atoi (argv[4]);
    HIDDEN_SIZE3    = atoi (argv[5]);
    OUTPUT_SIZE     = atoi (argv[6]);
    EPOCHS          = atoi (argv[7]);
    LEARNING_RATE   = atof (argv[8]);
    ANNEALING_RATE  = atof (argv[9]);

    // Set this sufficient buffer limit to avoid realloc. Modify to suit needs
    int BUF_SIZE = (INPUT_SIZE + OUTPUT_SIZE) * 12;

    int rows = 0;

    // Allocate line depending on the input size
    char *line = malloc (sizeof(char) * BUF_SIZE);

    // Get line count and then reset position on file
    while (fgets(line, BUF_SIZE, datafile)!=NULL) rows++;
    rewind (datafile);

    printf ("Rows: %d\n", rows);

    // Container of all the data. Last values should be the classes
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
    // Close file, free line
    fclose (datafile);

    // Store classes separately from the container
    double targets[rows][OUTPUT_SIZE];
    for (int row = 0; row < rows; row++) {
        for (int class = 0; class < OUTPUT_SIZE; class++) {
            targets[row][class] = container[row][INPUT_SIZE + class];
        }
    }

    free (line);

    double** input =        malloc ( sizeof(double*) * rows           );
    double* hidden =        malloc ( sizeof(double)  * HIDDEN_SIZE    );
    double* hidden2 =       malloc ( sizeof(double)  * HIDDEN_SIZE2   );
    double* hidden3 =       malloc ( sizeof(double)  * HIDDEN_SIZE3   );
    double* output =        malloc ( sizeof(double)  * OUTPUT_SIZE    );

    // Weights and biases
    double **weights_ih =   malloc ( sizeof(double*)  * INPUT_SIZE    );
    double **weights_hh =   malloc ( sizeof(double*)  * HIDDEN_SIZE   );
    double **weights_hhh =  malloc ( sizeof(double*)  * HIDDEN_SIZE2  );
    double **weights_ho =   malloc ( sizeof(double*)  * HIDDEN_SIZE3  );
    double *bias_h =        malloc ( sizeof(double)  * HIDDEN_SIZE    );
    double *bias_hh =       malloc ( sizeof(double)  * HIDDEN_SIZE2   );
    double *bias_hhh =      malloc ( sizeof(double)  * HIDDEN_SIZE3   );
    double *bias_o =        malloc ( sizeof(double)  * OUTPUT_SIZE    );
    for (int i = 0; i < INPUT_SIZE;   i++) weights_ih[i]  =  malloc ( sizeof(double) * HIDDEN_SIZE  );
    for (int i = 0; i < HIDDEN_SIZE;  i++) weights_hh[i]  =  malloc ( sizeof(double) * HIDDEN_SIZE2 );
    for (int i = 0; i < HIDDEN_SIZE2; i++) weights_hhh[i] =  malloc ( sizeof(double) * HIDDEN_SIZE3 );
    for (int i = 0; i < HIDDEN_SIZE3; i++) weights_ho[i]  =  malloc ( sizeof(double) * OUTPUT_SIZE  );
    for (int i = 0; i < rows; i++)               input[i] =  malloc ( sizeof(double) * INPUT_SIZE   );

    // Train and generate weights/biases if a saved model doesn't exist
    if (!saved_model) {

        /* RANDOMIZE WEIGHTS AND BIASES */
        randomizer (weights_ih, weights_hh, weights_hhh, weights_ho, bias_h, bias_hh, bias_hhh, bias_o);

        /* TRAINING */
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

                forwardpropagation(input[row], weights_ih, weights_hh, weights_hhh, weights_ho, hidden, hidden2, hidden3, output, bias_h, bias_hh, bias_hhh, bias_o);

                // Store error
                iteration_error [epoch] = backpropagation(input[row], hidden, hidden2, hidden3, output, targets[selected_row], weights_ih, weights_hh, weights_hhh, weights_ho, bias_h, bias_hh, bias_hhh, bias_o);

            }

            fprintf (saved_errors, "%lf\n", iteration_error[epoch]);

            //Report error every 20 epochs
            if (epoch % 20 == 0) {
                //printf("\rEpoch %d/%d -- Error: %.6lf, Rate: %lf", epoch, EPOCHS, iteration_error[epoch], LEARNING_RATE);
                printf("\rEpoch %d/%d -- Error: %.6lf, Rate: %f", epoch, EPOCHS, iteration_error[epoch], LEARNING_RATE);
                fflush(stdout);
            }

            // Progressively lowering the learning rate
            LEARNING_RATE *= ANNEALING_RATE;
        }

        fclose (saved_errors);

        /* SAVE WEIGHTS AND BIASES */
        savemodel(argv[10], weights_ih, weights_hh, weights_hhh, weights_ho, bias_h, bias_hh, bias_hhh, bias_o);

    }

    // Load weights and biases if a saved model exists
    else {
        printf ("Reading weights and biases from %s.\n", argv[10]);
        readmodel (argv[10], weights_ih, weights_hh, weights_hhh, weights_ho, bias_h, bias_hh, bias_hhh, bias_o);
    }

    /* TESTING */
    for (int row = 0; row < rows; row++) {

        // Pick random row to test
        int selected_row = randrange (0, rows-1);

        for (int j = 0; j < INPUT_SIZE; j++) {
            input[row][j] = container[selected_row][j];
        }

        forwardpropagation (input[row], weights_ih, weights_hh, weights_hhh, weights_ho, hidden, hidden2, hidden3, output, bias_h, bias_hh, bias_hhh, bias_o);

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            printf("Output: %.4lf Target: %.4lf: %.2f%%\n", output[i], targets[selected_row][i], 100 - fabs(output[i] - targets[selected_row][i])*100);
        }
        printf ("------\n");
    }

    /* FREE MEMORY */
    free (hidden);
    free (hidden2);
    free (hidden3);
    free (output);

    // Free container/input
    for (int row = 0; row < rows; row++) {
        free(container[row]);
        free(input[row]);
    }
    free (container);
    free (input);

    // Free weights/biases
    for (int i = 0; i<INPUT_SIZE; i++) {
        free(weights_ih[i]);
    }
    free (weights_ih);
    for (int i = 0; i<HIDDEN_SIZE; i++) {
        free(weights_hh[i]);
    }
    free (weights_hh);
    for (int i = 0; i<HIDDEN_SIZE2; i++) {
        free(weights_hhh[i]);
    }
    free (weights_hhh);
    for (int i = 0; i<HIDDEN_SIZE3; i++) {
        free(weights_ho[i]);
    }
    free (weights_ho);

    free (bias_h);
    free (bias_hh);
    free (bias_hhh);
    free (bias_o);

    return 0;
}
