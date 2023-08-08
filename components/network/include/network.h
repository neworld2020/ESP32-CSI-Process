#ifndef _ESP32_CSI_NETWORK_H_
#define _ESP32_CSI_NETWORK_H_

#include <stdint.h>

// The Input contains 104 elements, which is the number of subcarriers (52) * 2
#define NETWORK_INPUT_SIZE  104

// After PCA, the output is 5 elements
#define NETWORK_PCA_SIZE    5

// The output of the network is 1 element: 1 for Movement, 0 for Non-Movement
#define NETWORK_OUTPUT_SIZE 1

// store buffer size
#define INPUT_BUFFER_SIZE   50

#ifdef __cplusplus
extern "C" {
#endif

    // Network Initialize
    extern void network_init();

    // Input CSI data into the network: make sure the length >= NETWORK_INPUT_SIZE
    extern void network_input(int8_t* data, int length);

    // calculation: can take a long time to complete
    extern void model_prediction();

    // Get Network Output
    extern int network_get_output(void);

#ifdef __cplusplus
}
#endif

#endif // _ESP32_CSI_NETWORK_H_