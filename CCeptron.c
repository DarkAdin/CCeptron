#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE, EPOCHS;
float LEARNING_RATE, ANNEALING_RATE;

/* ERROR FUNCTIONS */
double error (double prediction, double target) {
    float delta = 1.35;
    // HUBER LOSS
    if (fabs(prediction-target) < delta) return 0.5 * (prediction-target) * (prediction-target);
    else return (delta * fabs(prediction-target) - 0.5 * delta * delta);
}
double derror (double prediction, double target) {
    float delta = 1.35;
    // HUBER LOSS
    if (fabs(prediction-target) < delta) return prediction-target;
    else {
        // Negatives
        if (prediction-target < 0) return -delta;
        // Positives
        else if (prediction-target > 0) return delta;
        // Zero
        else return 0.0;
    }
}

/* ACTIVATION FUNCTIONS */

// Sigmoid
float sigmoid ( float a ) {
    return 1 / (1 + expf(-a));
}
float dsigmoid ( float a ) {
    return a * (1 - a);
}

// Tanh
float ttanh ( float a ) {
    return (expf(a) - expf(-a)) / (expf(a) + expf(-a));
}
float dtanh ( float a ) {
    return 1 - ttanh(a) * ttanh(a);
}

// RELU
float relu ( float a ) {
    if (a < 0) return 0;
    else return a;
}
float drelu ( float a ) {
    if (a < 0) return 0;
    else return 1;
}

// Leaky RELU
float lrelu ( float a ) {
    if (a < 0.0) return 0.01 * a;
    else return a;
}
float dlrelu ( float a ) {
    if (a < 0.0) return a;
    else return 1;
}

// Return a random int in a specified range
int randrange (int min, int max) {
    return rand() % (max-min+1) + min;
}

// Get a float between 0-1
float frand () {
    return (float) rand() / (float) RAND_MAX;
}

/* FORWARD PROPAGATION */
void forwardpropagation (double input[INPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double hidden[HIDDEN_SIZE], double hidden2[HIDDEN_SIZE2], double hidden3[HIDDEN_SIZE3], double output[OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    // Calculate hidden layer values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * weights_ih[j][i];
        }
        hidden[i] += bias_h[i];
        hidden[i] = lrelu(hidden[i]);
    }

    // Calculate hidden2 layer values
    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden2[i] += hidden[j] * weights_hh[j][i];
        }
        hidden2[i] += bias_hh[i];
        hidden2[i] = lrelu(hidden2[i]);
    }

    // Calculate hidden3 layer values
    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        hidden3[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            hidden3[i] += hidden2[j] * weights_hhh[j][i];
        }
        hidden3[i] += bias_hhh[i];
        hidden3[i] = lrelu(hidden3[i]);
    }

    // Calculate output layer values
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            output[i] += hidden3[j] * weights_ho[j][i];
        }
        output[i] += bias_o[i];
        output[i] = sigmoid(output[i]);
    }
}

/* BACKPROPAGATION */
double backpropagation (double input[INPUT_SIZE], double hidden[HIDDEN_SIZE], double hidden2[HIDDEN_SIZE2], double hidden3[HIDDEN_SIZE3], double output[OUTPUT_SIZE], double target[OUTPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    double output_error = 0; // Error to be reported
    double output_gradients     [OUTPUT_SIZE];

    double hidden3_errors       [HIDDEN_SIZE3];
    double hidden3_gradients    [HIDDEN_SIZE3];

    double hidden2_errors       [HIDDEN_SIZE2];
    double hidden2_gradients    [HIDDEN_SIZE2];

    double hidden_errors        [HIDDEN_SIZE];
    double hidden_gradients     [HIDDEN_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error += error (target[i], output[i]); // Store error of this iteration

        bias_o[i] += LEARNING_RATE * output_gradients[i];
        output_gradients[i] = derror (target[i], output[i]) * dsigmoid (output[i]);
    }

    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        hidden3_errors[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] += LEARNING_RATE * output_gradients[j] * hidden3[i];
            hidden3_errors[i] += output_gradients[j] * weights_ho[i][j];
        }
        hidden3_gradients[i] = hidden3_errors[i] * dlrelu(hidden3[i]);
        bias_hhh[i] += LEARNING_RATE * hidden3_gradients[i];
    }

    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2_errors[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            weights_hhh[i][j] += LEARNING_RATE * hidden3_gradients[j] * hidden2[i];
            hidden2_errors[i] += hidden3_gradients[j] * weights_hhh[i][j];
        }
        hidden2_gradients[i] = hidden2_errors[i] * dlrelu(hidden2[i]);
        bias_hh[i] += LEARNING_RATE * hidden2_gradients[i];
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_errors[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            weights_hh[i][j] += LEARNING_RATE * hidden2_gradients[j] * hidden[i];
            hidden_errors[i] += hidden2_gradients[j] * weights_hh[i][j];
        }
        hidden_gradients[i] = hidden_errors[i] * dlrelu(hidden[i]);
        bias_h[i] += LEARNING_RATE * hidden_gradients[i];
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] += LEARNING_RATE * hidden_gradients[j] * input[i];
        }
    }

    return output_error / OUTPUT_SIZE; // Average error of this iteration
}

// Randomize weights and biases
void randomizer (double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] = frand() - 0.5;
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            weights_hh[i][j] = frand() - 0.5;
        }
        bias_h[i] = frand() - 0.5;
    }

    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            weights_hhh[i][j] = frand() - 0.5;
        }
        bias_hh[i] = frand() - 0.5;
    }

    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] = frand() - 0.5;
        }
        bias_hhh[i] = frand() - 0.5;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias_o[i] = frand() - 0.5;
    }
}

// Save weights and biases
void savemodel (char *model, double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    FILE *saved_model = fopen (model, "w");

    // Header of the file
    fprintf (saved_model, "%d %d %d %d %d\n", INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE);

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            fprintf (saved_model, "%lf\n", weights_ih[i][j]);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            fprintf (saved_model, "%lf\n", weights_hh[i][j]);
        }
        fprintf (saved_model, "%lf\n", bias_h[i]);
    }
    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        for (int j = 0; j < HIDDEN_SIZE3; j++) {
            fprintf (saved_model, "%lf\n", weights_hhh[i][j]);
        }
        fprintf (saved_model, "%lf\n", bias_hh[i]);
    }
    for (int i = 0; i < HIDDEN_SIZE3; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            fprintf (saved_model, "%lf\n", weights_ho[i][j]);
        }
        fprintf (saved_model, "%lf\n", bias_hhh[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        fprintf (saved_model, "%lf\n", bias_o[i]);
    }

    fclose (saved_model);
}

// Load weights and biases
void readmodel (char *model, double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_hhh[HIDDEN_SIZE2][HIDDEN_SIZE3], double weights_ho[HIDDEN_SIZE3][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_hhh[HIDDEN_SIZE3], double bias_o[OUTPUT_SIZE]) {

    FILE *loaded_model = fopen (model, "r");

    if (!loaded_model) printf ("Error loading model %s.\n", model);

    else {

        // Load header
        fscanf (loaded_model, "%d %d %d %d %d\n", &INPUT_SIZE, &HIDDEN_SIZE, &HIDDEN_SIZE2, &HIDDEN_SIZE3, &OUTPUT_SIZE);

        // Load weights
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                fscanf (loaded_model, "%lf\n", &weights_ih[i][j]);
            }
        }
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE2; j++) {
                fscanf (loaded_model, "%lf\n", &weights_hh[i][j]);
            }
            fscanf (loaded_model, "%lf\n", &bias_h[i]);
        }
        for (int i = 0; i < HIDDEN_SIZE2; i++) {
            for (int j = 0; j < HIDDEN_SIZE3; j++) {
                fscanf (loaded_model, "%lf\n", &weights_hhh[i][j]);
            }
            fscanf (loaded_model, "%lf\n", &bias_hh[i]);
        }
        for (int i = 0; i < HIDDEN_SIZE3; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                fscanf (loaded_model, "%lf\n", &weights_ho[i][j]);
            }
            fscanf (loaded_model, "%lf\n", &bias_hhh[i]);
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
        puts ("./CCeptron file.csv input_size hidden_size hidden_size2 hidden_size3 output_size epochs learning_rate annealing_rate saved_model");
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

    int rows = 0;

    char line[INPUT_SIZE*10];

    // Get line count and reset position on file
    while (fgets(line, sizeof(line), datafile)!=NULL) rows++;
    rewind (datafile);

    // Container of all the data. Last values are the classes
    float container[rows][INPUT_SIZE+OUTPUT_SIZE];

    // Reset row counter
    rows -= rows;

    // Parse lines
    while (fgets(line, sizeof(line), datafile)) {
        for (int i = 0; i<INPUT_SIZE+OUTPUT_SIZE; i++) {
            if (i == 0) container[rows][i] = atof(strtok(line,","));
            else container[rows][i] = atof(strtok(NULL,","));
        }
        rows++;
    }

    // Store classes separately
    double targets[rows][OUTPUT_SIZE];
    for (int row = 0; row < rows; row++) {
        for (int class = 0; class < OUTPUT_SIZE; class++) {
            targets[row][class] = container[row][INPUT_SIZE + class];
        }
    }

    // Close file
    fclose (datafile);

    double weights_ih   [INPUT_SIZE][HIDDEN_SIZE];
    double weights_hh   [HIDDEN_SIZE][HIDDEN_SIZE2];
    double weights_hhh  [HIDDEN_SIZE2][HIDDEN_SIZE3];
    double weights_ho   [HIDDEN_SIZE3][OUTPUT_SIZE];
    double bias_h       [HIDDEN_SIZE];
    double bias_hh      [HIDDEN_SIZE2];
    double bias_hhh     [HIDDEN_SIZE3];
    double bias_o       [OUTPUT_SIZE];

    // Train and generate weights/biases if a saved model doesn't exist
    if (!saved_model) {

        /* RANDOMIZE WEIGHTS AND BIASES */
        randomizer(weights_ih, weights_hh, weights_hhh, weights_ho, bias_h, bias_hh, bias_hhh, bias_o);

        /* TRAINING */
        printf("Training for %d epochs.\n", EPOCHS);

        double iteration_error[EPOCHS];

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (int row = 0; row < rows; row++) {

                // Pick random row to train on
                int selected_row = randrange(0, rows-1);

                // Store data parameters
                double input[INPUT_SIZE];
                for (int j = 0; j < INPUT_SIZE; j++) {
                    input[j] = container[selected_row][j];
                }

                double hidden   [HIDDEN_SIZE];
                double hidden2  [HIDDEN_SIZE2];
                double hidden3  [HIDDEN_SIZE3];
                double output   [OUTPUT_SIZE];

                forwardpropagation(input, weights_ih, weights_hh, weights_hhh, weights_ho, hidden, hidden2, hidden3, output, bias_h, bias_hh, bias_hhh, bias_o);

                // Store error
                //iteration_error = backpropagation(input, hidden, hidden2, hidden3, output, targets[selected_row], weights_ih, weights_hh, weights_hhh, weights_ho, bias_h, bias_hh, bias_hhh, bias_o);
                iteration_error[epoch] = backpropagation(input, hidden, hidden2, hidden3, output, targets[selected_row], weights_ih, weights_hh, weights_hhh, weights_ho, bias_h, bias_hh, bias_hhh, bias_o);

                //backpropagation(input, hidden, hidden2, hidden3, output, targets[selected_row], weights_ih, weights_hh, weights_hhh, weights_ho, bias_h, bias_hh, bias_hhh, bias_o);
            }

            if (epoch % 50 == 0) {
                printf("Epoch %d/%d -- Error: %.8lf\n", epoch, EPOCHS, iteration_error[epoch]);
            }

            // Progressively lowering the learning rate
            LEARNING_RATE *= ANNEALING_RATE;
        }

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

        double input[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[j] = container[selected_row][j];
        }

        double hidden   [HIDDEN_SIZE];
        double hidden2  [HIDDEN_SIZE2];
        double hidden3  [HIDDEN_SIZE3];
        double output   [OUTPUT_SIZE];

        forwardpropagation(input, weights_ih, weights_hh, weights_hhh, weights_ho, hidden, hidden2, hidden3, output, bias_h, bias_hh, bias_hhh, bias_o);

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            printf("Output: %.2lf Target: %.4lf: %.4f%%\n", output[i], targets[selected_row][i], 100 - fabs(output[i] - targets[selected_row][i])*100);
        }
        printf ("------\n");
    }

    return 0;
}
