#include "../Eigen/Dense"
#include "../include/network.h"

using namespace Eigen;
const Matrix<double, NETWORK_PCA_SIZE, NETWORK_INPUT_SIZE> network_pca = Matrix<double, NETWORK_PCA_SIZE, NETWORK_INPUT_SIZE>::Zero();

extern void network_input(int8_t* data, int length) {
    Matrix<double, 1, NETWORK_INPUT_SIZE> input_matrix = Matrix<double, 1, NETWORK_INPUT_SIZE>::Zero();
    auto result = input_matrix * network_pca.transpose();
    return;
}

extern void network_task() {
    return;
}

extern int network_get_output(void) {
    return 1;
}