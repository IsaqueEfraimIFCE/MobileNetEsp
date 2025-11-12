#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <stdint.h>
#include "mobilenetv1025128.h"
#include "images128.h"
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
TfLiteTensor* output2 = nullptr;
uint8_t* input_buffer = nullptr;
uint8_t* output_buffer = nullptr;
uint8_t* output_buffer2 = nullptr;
constexpr int kTensorArenaSize = 150000;
uint8_t* tensor_arena = nullptr;
int resultados[200];
int resultados2[200];
}  

void setup() {
  model = tflite::GetModel(model2);
  tensor_arena=(uint8_t*)(malloc(kTensorArenaSize));

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<17> resolver;
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddDepthwiseConv2D();
  resolver.AddPad();
  resolver.AddAdd();
  resolver.AddMean();
  resolver.AddMul();
  resolver.AddSub();
  resolver.AddLogistic();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  output2 = interpreter->output(1);

  TfLiteIntArray* dimsinput = input->dims;
  TfLiteIntArray* dimsoutput = output->dims;
  MicroPrintf("Input tensor dims: %d\n", dimsinput->size);
  MicroPrintf("Output tensor dims: %d\n", dimsoutput->size);

  MicroPrintf("Model input shape: [%d, %d, %d, %d]", dimsinput->data[0], dimsinput->data[1], dimsinput->data[2], dimsinput->data[3]);
  MicroPrintf("Model output shape: [%d,%d]", dimsoutput->data[0],dimsoutput->data[1]);
  TfLiteType input_type = input->type;
  const char* input_type_str = "UNKNOWN";
  switch (input_type) {
    case kTfLiteFloat32:
        input_type_str = "Float32";
        break;
    case kTfLiteInt8:
        input_type_str = "Int8";
        break;
    case kTfLiteUInt8:
        input_type_str = "UInt8";
        break;
    case kTfLiteInt32:
        input_type_str = "Int32";
        break;
    case kTfLiteBool:
        input_type_str = "Bool";
        break;
    case kTfLiteFloat16:
        input_type_str = "Float16";
        break;
    default:
        input_type_str = "Unknown";
        break;
}
MicroPrintf("Model input data type: %s", input_type_str);
TfLiteType output_type = output->type;
const char* output_type_str = "UNKNOWN";

// Mapping TfLiteType enum to string representation
switch (output_type) {
    case kTfLiteFloat32:
        output_type_str = "Float32";
        break;
    case kTfLiteInt8:
        output_type_str = "Int8";
        break;
    case kTfLiteUInt8:
        output_type_str = "UInt8";
        break;
    case kTfLiteInt32:
        output_type_str = "Int32";
        break;
    case kTfLiteBool:
        output_type_str = "Bool";
        break;
    case kTfLiteFloat16:
        output_type_str = "Float16";
        break;
    default:
        output_type_str = "Unknown";
        break;
}

MicroPrintf("Model output data type: %s", output_type_str);
  input_buffer = tflite::GetTensorData<uint8_t>(input);
  output_buffer = tflite::GetTensorData<uint8_t>(output);
  output_buffer2 = tflite::GetTensorData<uint8_t>(output2);


}


void loop(int it){
    int max_index=0;
    for (int i = 0; i < 64*64*3; i++) {
        input_buffer[i] = it*64*64*3+i;
    }

    MicroPrintf("Invocando o interpreter");

    int64_t start_time = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter->Invoke();

    int64_t end_time = esp_timer_get_time();

    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return;
    }

    // Calcula e imprime o tempo gasto em milissegundos
    int64_t duration = end_time - start_time;  // Em microsegundos
    MicroPrintf("Inferencia %d levou %lld microsegundos", it, duration);
     for (int j = 0; j < 5; j++) {
      if (output_buffer[j] > output_buffer[max_index]) {
        max_index = j;
      }
    }
    MicroPrintf("%d", max_index);
    resultados[it] = max_index;
    max_index=0;
    for (int j = 0; j < 5; j++) {
      if (output_buffer2[j] > output_buffer2[max_index]) {
        max_index = j;
      }
    }
    MicroPrintf("%d", max_index);
    resultados2[it] = max_index;
  }

extern "C" void app_main(void) {
     esp_task_wdt_deinit();
    setup();
    for(int i = 0; i < 200; i++) {
            loop(i);
              }
     char buffer[1024] = "";
  for (int i = 0; i < 200; i++) {
    char temp[16];
    sprintf(temp, "%d ", resultados[i]);
    strcat(buffer, temp);
  }
   for (int i = 0; i < 200; i++) {
    char temp[16];
    sprintf(temp, "%d ", resultados2[i]);
    strcat(buffer, temp);
  }
  MicroPrintf("%s", buffer);
    }
