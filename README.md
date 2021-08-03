# Post Quantization with TensorFlow and Model Compilation with TVM
## How to Quantize a Pre-trained Float TensorFlow Model with Post Quantization 
* TensorFlow version: 2.5.0
* Refer to post-training_quantization_inception_v3.py and post-training_quantization_mobilenet_v2.py
## How to Compile a Quantized Model with TVM
* TVM commit: da27e6d9a466263a9a0025aba92086a8bf837edb
* Refer to inception_v3.py and mobilenet_v2.py
## Reference
* https://github.com/aquapapaya/InstallTVM
* https://www.tensorflow.org/lite/performance/post_training_quantization
* https://www.tensorflow.org/api_docs/python/tf/keras/applications
* https://stackoverflow.com/questions/57877959/what-is-the-correct-way-to-create-representative-dataset-for-tfliteconverter
* https://stackoverflow.com/questions/66984379/problem-in-conversion-of-pb-to-tflite-int8-for-coral-devboard-coral-ai
* https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
* https://www.tensorflow.org/lite/convert/
* https://github.com/tensorflow/models/tree/master/research
* https://android.googlesource.com/platform/external/tensorflow/+/33965c1ca30600824f1bc17d5dee30b0c80ce1b6/tensorflow/lite/g3doc/convert/python_api.md
