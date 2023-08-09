#include "../../tflite-lib/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "../../tflite-lib/tensorflow/lite/micro/micro_interpreter.h"
#include "../../tflite-lib/tensorflow/lite/micro/system_setup.h"
#include "../../tflite-lib/tensorflow/lite/schema/schema_generated.h"

#include "../Eigen/Dense"
#include "../include/network.h"
#include "../include/model.h"

using namespace Eigen;
typedef Matrix<float, 1, NETWORK_PCA_SIZE> PCA_Matrix;
typedef Matrix<float, 1, NETWORK_INPUT_SIZE> Input_Matrix;
typedef Matrix<float, NETWORK_PCA_SIZE, NETWORK_INPUT_SIZE> Transform_Matrix;

static Matrix<float, INPUT_BUFFER_SIZE, NETWORK_PCA_SIZE> buffer_matrix = Matrix<float, INPUT_BUFFER_SIZE, NETWORK_PCA_SIZE>::Zero();
static bool can_calculate = false;
static int buffer_index = 0;
static int buffer_result = -1;

const Transform_Matrix network_pca {
    {0.012374,0.077653,0.014368,0.077783,0.016452,0.078123,0.018041,0.079011,0.019564,0.079232,0.021520,0.079335,0.023660,0.079326,0.025321,0.078694,0.027005,0.077726,0.029581,0.077783,0.033140,0.079171,0.035616,0.079434,0.037086,0.079287,0.038577,0.080047,0.041265,0.080980,0.045081,0.080933,0.047222,0.080385,0.048812,0.080780,0.051600,0.082011,0.053874,0.083087,0.055791,0.083345,0.058600,0.083216,0.062864,0.083547,0.068111,0.083855,0.072946,0.085356,0.076754,0.087592,0.081614,0.088856,0.084827,0.090077,0.089430,0.091507,0.093430,0.093775,0.096951,0.097804,0.099427,0.101022,0.101663,0.102723,0.104424,0.104455,0.107232,0.106339,0.111747,0.109224,0.116270,0.112862,0.118326,0.115684,0.119428,0.118218,0.121352,0.122011,0.125179,0.126741,0.128794,0.130392,0.130571,0.133972,0.131954,0.140114,0.131087,0.146006,0.129323,0.150177,0.128702,0.154800,0.127167,0.158349,0.126646,0.162776,0.127023,0.168654,0.124235,0.174078,0.121162,0.178342},
    {0.077504,-0.012116,0.077559,-0.014547,0.077840,-0.017616,0.078587,-0.019725,0.078745,-0.020899,0.078963,-0.021655,0.078817,-0.023221,0.078137,-0.024845,0.077342,-0.026907,0.077586,-0.029950,0.078821,-0.033181,0.079059,-0.035698,0.079266,-0.037994,0.080151,-0.040108,0.080795,-0.042395,0.080649,-0.044947,0.079959,-0.046549,0.080413,-0.048914,0.081923,-0.052544,0.083331,-0.055114,0.083620,-0.056952,0.083170,-0.059490,0.083495,-0.063410,0.084064,-0.068809,0.085443,-0.073434,0.087294,-0.076461,0.088651,-0.081256,0.090195,-0.085414,0.091729,-0.089997,0.093805,-0.093634,0.097904,-0.097240,0.101091,-0.099996,0.102663,-0.102275,0.104410,-0.104676,0.106447,-0.107130,0.109228,-0.111438,0.112681,-0.116177,0.115659,-0.118449,0.118420,-0.119653,0.122278,-0.121330,0.127001,-0.124965,0.130446,-0.128617,0.133793,-0.130373,0.140309,-0.131482,0.146507,-0.130534,0.150707,-0.128577,0.155208,-0.128013,0.158572,-0.126610,0.162864,-0.126167,0.168708,-0.126448,0.174161,-0.123502,0.178629,-0.120232},
    {-0.190301,-0.148503,-0.191024,-0.141126,-0.191937,-0.130537,-0.191921,-0.123829,-0.188686,-0.118808,-0.186646,-0.113876,-0.182415,-0.107333,-0.176543,-0.098165,-0.170814,-0.087311,-0.167889,-0.076276,-0.168595,-0.066436,-0.165131,-0.056657,-0.158397,-0.047790,-0.152854,-0.041098,-0.147915,-0.033940,-0.143223,-0.024698,-0.136660,-0.017101,-0.129417,-0.011324,-0.122676,-0.005979,-0.116286,-0.001412,-0.107723,0.002675,-0.098783,0.006800,-0.091035,0.011247,-0.082907,0.015934,-0.073840,0.018620,-0.066601,0.019171,-0.042625,0.023871,-0.033579,0.023137,-0.023300,0.023620,-0.013620,0.024450,-0.002660,0.024686,0.008640,0.024948,0.018735,0.024274,0.027530,0.023983,0.035079,0.023855,0.045286,0.021143,0.056077,0.017206,0.063640,0.014714,0.070099,0.012366,0.078650,0.009178,0.089345,0.005194,0.099706,-0.000690,0.107378,-0.006265,0.115424,-0.008693,0.123177,-0.009316,0.129499,-0.010825,0.136737,-0.014213,0.141989,-0.017918,0.149331,-0.022830,0.159226,-0.027280,0.165171,-0.027452,0.168694,-0.026870},
    {-0.147665,0.190294,-0.141729,0.191177,-0.133442,0.192616,-0.127858,0.193105,-0.121456,0.189946,-0.114436,0.185918,-0.106475,0.181255,-0.097292,0.175444,-0.086681,0.170212,-0.076812,0.167870,-0.066010,0.167822,-0.056189,0.164793,-0.048992,0.159200,-0.042467,0.153735,-0.034672,0.148320,-0.024322,0.142261,-0.016164,0.134874,-0.010848,0.127800,-0.005888,0.122673,-0.001521,0.116641,0.002735,0.107865,0.007152,0.098373,0.011469,0.090011,0.016247,0.082045,0.018871,0.072156,0.019474,0.064022,0.025394,0.040213,0.025086,0.031789,0.025173,0.021549,0.025147,0.011700,0.025753,0.000669,0.026125,-0.009711,0.025798,-0.020043,0.024786,-0.029071,0.023947,-0.037624,0.020652,-0.047613,0.017432,-0.057664,0.015419,-0.065386,0.013546,-0.072084,0.010187,-0.079707,0.005820,-0.089679,-0.000794,-0.099935,-0.006010,-0.107840,-0.008847,-0.115674,-0.009868,-0.122693,-0.011613,-0.128734,-0.015371,-0.135855,-0.018001,-0.141178,-0.022202,-0.149093,-0.027567,-0.158247,-0.029064,-0.163376,-0.028873,-0.166553},
    {-0.069782,-0.221600,-0.065782,-0.206060,-0.062261,-0.183337,-0.054467,-0.166861,-0.045324,-0.150715,-0.039096,-0.134672,-0.030089,-0.116189,-0.020862,-0.094737,-0.013313,-0.072876,-0.006002,-0.055466,-0.001383,-0.044651,0.004908,-0.031730,0.017608,-0.014926,0.030439,0.000778,0.040333,0.013492,0.046876,0.022766,0.053241,0.035279,0.064214,0.047714,0.079797,0.056532,0.092061,0.063607,0.100721,0.068443,0.107023,0.075005,0.114004,0.077194,0.123679,0.075924,0.131049,0.076942,0.135100,0.079288,0.142078,0.083281,0.146669,0.076631,0.148269,0.075330,0.146593,0.077683,0.147981,0.075585,0.147168,0.071348,0.142077,0.068021,0.135323,0.065543,0.126037,0.066187,0.114932,0.061333,0.102047,0.053205,0.087832,0.045442,0.075424,0.038915,0.058139,0.031347,0.037941,0.025716,0.013332,0.015494,-0.014386,0.003946,-0.039221,-0.007515,-0.062055,-0.018492,-0.087131,-0.030854,-0.114523,-0.048470,-0.141722,-0.067164,-0.173478,-0.087098,-0.208808,-0.110301,-0.234829,-0.134618,-0.250750,-0.151441}
};

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
tflite::MicroMutableOpResolver<3> resolver;
int inference_count = 0;

constexpr int kTensorArenaSize = 10240;
uint8_t tensor_arena[kTensorArenaSize];
}


PCA_Matrix _pca_transform(Input_Matrix input_data)
{
    PCA_Matrix output_data = input_data * network_pca.transpose();
    return output_data;
}

extern void network_init()
{
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    // Pull in only the operation implementations we need.
    // https://netron.app/
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
    // Get Input Size and Output Size
    auto input_size = input->dims;
    for(int i=0;i<input_size->size;i++){
        MicroPrintf("input_size[%d]: %d", i, input_size->data[i]);
    }
    auto output_size = output->dims;
    for(int i=0;i<output_size->size;i++){
        MicroPrintf("output_size[%d]: %d", i, output_size->data[i]);
    }

    // Keep track of how many inferences we have performed.
    inference_count = 0;
}

extern void network_input(int8_t* data, int length) 
{
    if(can_calculate){
        // last calculation is not finished
        return;
    }
    if(length >= NETWORK_INPUT_SIZE) {
        Input_Matrix input_data = Input_Matrix::Zero();
        for(int i = 0; i < NETWORK_INPUT_SIZE; i++) {
            input_data(0, i) = (float)data[i];
        }
        auto pca_data = _pca_transform(input_data);
        // put matrix into network
        buffer_matrix.row(buffer_index) = pca_data;
        buffer_index++;
        if(buffer_index >= INPUT_BUFFER_SIZE) {
            buffer_index = 0;
            can_calculate = true;
        }
    }
    return;
}

extern int network_get_output(void) 
{
    return buffer_result;
}

extern void model_prediction(void)
{
    if(can_calculate){
        // calculate
        // input->data.f = buffer_matrix.transpose().data();
        for(int row=0;row < INPUT_BUFFER_SIZE;row++){
            for(int col=0;col < NETWORK_PCA_SIZE;col++){
                input->data.f[row*NETWORK_PCA_SIZE + col] = buffer_matrix(row, col);
            }
        }
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            MicroPrintf("Invoke failed");
            return;
        }
        // Obtain the quantized output from model's output tensor
        float move_prob = output->data.f[29*output->dims->data[2] + 0];
        float static_prob = output->data.f[29*output->dims->data[2] + 1];
        MicroPrintf("static prob: %f, move_prob: %f", static_prob, move_prob);
        if(static_prob > STATIC_THRESHOLD){
            buffer_result = 0;
        }else if(move_prob > MOVE_THRESHOLD){
            buffer_result = 1;
        }else{
            buffer_result = -1;
        }
        can_calculate = false;
    }else{
        // data is not enough
        return;
    }
}
