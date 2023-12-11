#include<stdio.h>
#include<unistd.h>
#include<time.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

int INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, OUTPUT_SIZE, EPOCHS;
float LEARNING_RATE, ANNEALING_RATE;

// Error functions
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

// SIGMOID
float sigmoid ( float a ) {
    return 1 / (1 + expf(-a));
}
float dsigmoid ( float a ) {
    return a * (1 - a);
}

// TANH
float dtanh ( float a ) {
    return 1 - tanh(a) * tanh(a);
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

// LEAKY RELU
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
    return (float)rand() / (float)RAND_MAX;
}

// Forward propagation
void forwardpropagation (double input[INPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_ho[HIDDEN_SIZE2][OUTPUT_SIZE], double hidden[HIDDEN_SIZE], double hidden2[HIDDEN_SIZE2], double output[OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_o[OUTPUT_SIZE]) {

    // Calculate hidden layer values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * weights_ih[j][i];
        }
        hidden[i] += bias_h[i];
        hidden[i] = tanh(hidden[i]);
    }

    // Calculate hidden2 layer values
    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden2[i] += hidden[j] * weights_hh[j][i];
        }
        hidden2[i] += bias_hh[i];
        hidden2[i] = tanh(hidden2[i]);
    }

    // Calculate output layer values
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            output[i] += hidden2[j] * weights_ho[j][i];
        }
        output[i] += bias_o[i];
        output[i] = sigmoid(output[i]);
    }
}

// Backpropagation
void backwardpropagation (double input[INPUT_SIZE], double hidden[HIDDEN_SIZE], double hidden2[HIDDEN_SIZE2], double output[OUTPUT_SIZE], double target[OUTPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2], double weights_ho[HIDDEN_SIZE2][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_hh[HIDDEN_SIZE2], double bias_o[OUTPUT_SIZE]) {

    double output_gradients[OUTPUT_SIZE];

    double hidden2_errors[HIDDEN_SIZE2];
    double hidden2_gradients[HIDDEN_SIZE2];

    double hidden_errors[HIDDEN_SIZE];
    double hidden_gradients[HIDDEN_SIZE];

    for (int a = 0; a < INPUT_SIZE; a++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            hidden_errors[h] = 0;
            for (int i = 0; i < HIDDEN_SIZE2; i++) {
                hidden2_errors[i] = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    output_gradients[j] = derror(target[j], output[j]) * dsigmoid (output[j]);

                    // Output layer
                    weights_ho[i][j] += LEARNING_RATE * output_gradients[j] * hidden2[i];
                    bias_o[j] += LEARNING_RATE * output_gradients[j];
                    hidden2_errors[i] += output_gradients[j] * weights_ho[i][j];
                }
                hidden2_gradients[i] = hidden2_errors[i] * dtanh(hidden2[i]);

                // Hidden layer 2
                weights_hh[h][i] += LEARNING_RATE * hidden2_gradients[i] * hidden[h];
                bias_hh[i] += LEARNING_RATE * hidden2_gradients[i];
                hidden_errors[h] += hidden2_gradients[i] * weights_hh[h][i];
            }
            hidden_gradients[h] = hidden_errors[h] * dtanh(hidden[h]);

            // Hidden layer 1
            weights_ih[a][h] += LEARNING_RATE * hidden_gradients[h] * input[a];
            bias_h[h] += LEARNING_RATE * hidden_gradients[h];
        }
    }
}

int main (int argc, char **argv) {

    if (argc != 9) {
        puts ("./CCeptron file.csv input_size hidden_size hidden_size2 output_size epochs learning_rate annealing_rate");
        return 1;
    }

    FILE *datafile = fopen (argv[1], "r");

    if (!datafile) {
        puts ("Invalid data file.");
        return 1;
    }

    // Seed random number generator
    srand(time(0) + getpid());

    INPUT_SIZE = atoi (argv[2]);
    HIDDEN_SIZE = atoi (argv[3]);
    HIDDEN_SIZE2 = atoi (argv[4]);
    OUTPUT_SIZE = atoi (argv[5]);
    EPOCHS = atoi (argv[6]);
    LEARNING_RATE = atof (argv[7]);
    ANNEALING_RATE = atof (argv[8]);

    int rows = 0;

    //TODO: malloc/realloc dinamically
    char line[INPUT_SIZE*10];

    // Get line count
    while (fgets(line, sizeof(line), datafile)!=NULL) rows++;

    // Reset position on file
    rewind(datafile);

    // Container of the all the data. Last value is the class.
    float container[rows][INPUT_SIZE+1];

    // Reset row counter
    rows -= rows;

    // Parse lines
    while (fgets(line, sizeof(line), datafile)) {
        strcpy (line, strtok(line,"\n"));
        for (int i = 0; i<INPUT_SIZE+1; i++) {
            if (i == 0) container[rows][i] = atof(strtok(line,","));
            else container[rows][i] = atof(strtok(NULL,","));
        }
        rows++;
    }

    // Close file
    fclose (datafile);
    printf ("File %s closed.\n", argv[1]);
    
    // Initialize weights/biases with random values
    double weights_ih[INPUT_SIZE][HIDDEN_SIZE];
    double weights_hh[HIDDEN_SIZE][HIDDEN_SIZE2];
    double weights_ho[HIDDEN_SIZE2][OUTPUT_SIZE];

    double bias_h[HIDDEN_SIZE];
    double bias_hh[HIDDEN_SIZE2];
    double bias_o[OUTPUT_SIZE];

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
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] = frand() - 0.5;
        }
        bias_hh[i] = frand() - 0.5;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias_o[i] = frand() - 0.5;
    }

    // Store classes separately
    // TODO: more than 1 class
    double targets[rows][OUTPUT_SIZE];
    for (int row = 0; row<rows; row++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            targets[row][i] = container[row][INPUT_SIZE + i];
        }
    }

    /* TRAINING */
    printf("Training for %d epochs.\n", EPOCHS);
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        if (epoch % 100 == 0) {
            printf("Epoch %d/%d\n", epoch, EPOCHS);
        }
        for (int row = 0; row < rows; row++) {

            // Pick random row to train on
            int selected_row = randrange(0, rows-1);

            // Store parameters
            double input[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[j] = container[selected_row][j];
            }

            double hidden[HIDDEN_SIZE];
            double hidden2[HIDDEN_SIZE2];
            double output[OUTPUT_SIZE];

            forwardpropagation(input, weights_ih, weights_hh, weights_ho, hidden, hidden2, output, bias_h, bias_hh, bias_o);
            backwardpropagation(input, hidden, hidden2, output, targets[selected_row], weights_ih, weights_hh, weights_ho, bias_h, bias_hh, bias_o);
        }
        LEARNING_RATE *= ANNEALING_RATE;
    }

    /* TESTING
     * TODO: use another dataset for testing
     */
    for (int row = 0; row < rows; row++) {

        // Pick random row to test
        int selected_row = randrange (0, rows-1);

        double input[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[j] = container[selected_row][j];
        }

        double hidden[HIDDEN_SIZE];
        double hidden2[HIDDEN_SIZE2];
        double output[OUTPUT_SIZE];
        forwardpropagation(input, weights_ih, weights_hh, weights_ho, hidden, hidden2, output, bias_h, bias_hh, bias_o);

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            printf("Output: %.2lf, Orig: %.2lf, Error: %.2f\n", output[i], targets[selected_row][i], fabs(output[i] - targets[selected_row][i]));
        }
        puts ( "---" );
    }

    return 0;
}
