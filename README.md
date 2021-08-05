# Post Quantization with TensorFlow and Model Compilation with TVM
## How to Quantize a Pre-trained Float TensorFlow Model with Post Quantization 
* TensorFlow version: 2.5.0
* Refer to [post-training_quantization_inception_v3.py](post-training_quantization_inception_v3.py), [post-training_quantization_mobilenet_v2.py](post-training_quantization_mobilenet_v2.py), and [post-training_quantization_vgg16.py](post-training_quantization_vgg16.py)
* These files demonstrate full integer quantization using TensorFlow
* You can also find how to conduct integer quantization with float fallback in these files
* The images in [test](test) are used for calibration
## How to Compile a Quantized Model with TVM
* TVM commit: da27e6d9a466263a9a0025aba92086a8bf837edb
* Refer to [inception_v3.py](inception_v3.py), [mobilenet_v2.py](mobilenet_v2.py), and [vgg16.py](vgg16.py)
* 
## A TFlite Model Generated from the [script](post-training_quantization_mobilenet_v2.py)
* Refer to [mobilenet_v2_int8.tflite](mobilenet_v2_int8.tflite)
* Accuracy on the images in [image_classification_50](image_classification_50): Top-1 accuracy: 60.00%; Top-5 accuracy: 84.00%
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
