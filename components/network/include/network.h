#ifndef _ESP32_CSI_NETWORK_H_
#define _ESP32_CSI_NETWORK_H_

#include <stdint.h>

// The Input contains 104 elements, which is the number of subcarriers (52) * 2
#define NETWORK_INPUT_SIZE  104

// After PCA, the output is 5 elements
#define NETWORK_PCA_SIZE    5

// The output of the network is 1 element: 1 for Movement, 0 for Non-Movement
#define NETWORK_OUTPUT_SIZE 1

#ifdef __cplusplus
extern "C" {
#endif

    // Input CSI data into the network
    extern void network_input(int8_t* data, int length);

    // A function runs forever, the only way to use it is to create a task for it
    extern void network_task();

    // Get Network Output
    extern int network_get_output(void);

#ifdef __cplusplus
}
#endif

#endif // _ESP32_CSI_NETWORK_H_