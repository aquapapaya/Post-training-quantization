"""
Deploy Pre-Trained TensorFlow Lite MobileNet V2
===============================================
By Kuen-Wey Lin
TVM commit: da27e6d9a466263a9a0025aba92086a8bf837edb
"""

######################################################################
# Set environment variables
# -------------------------

import tvm

target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)

model_path = './mobilenet_v2_int8.tflite'
input_name = 'input_1'
data_type = 'int8' # input's data type
img_path = './image_classification_50/'
resulting_file_directory = './tvm_generated_files/'

######################################################################
# Set input size
# --------------

batch_size = 1
num_class = 1000
image_dimention = 3
image_shape = (224, 224)
data_shape = (batch_size,) + image_shape + (image_dimention,)
out_shape = (batch_size, num_class)

######################################################################
# Load a TFLite model
# -------------------

import os
tflite_model_file = os.path.join(model_path)
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Convert the TFLite model into Relay IR
# --------------------------------------

import tvm.relay as relay
dtype_dict = {input_name: data_type}
shape_dict = {input_name: data_shape}

mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict=shape_dict,
                                         dtype_dict=dtype_dict)

print("Relay IR:\n", mod)

######################################################################
# Compile the Relay module
# ------------------------

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize":True}):
    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

######################################################################
# Generate the five files
# -----------------------
'''
print("Printing host code to host_code.cc...")
with open('host_code.cc', 'w') as f:
    print(lib.get_source(), file=f)

print("Printing device code to device_code.cl...")
with open('device_code.cl', 'w') as f:
    print(lib.imported_modules[0].get_source(), file=f)

print("Printing meta json to device_code.tvm_meta.json...")
lib.imported_modules[0].save("device_code", "cl")
os.remove("device_code")

print("Printing binary parameters to binary_params.bin...")
with open('binary_params.bin', 'wb') as writer:
    writer.write(relay.save_param_dict(params))
    writer.close()

print("Printing graph to graph.json...")
with open('graph.json', 'w') as f:
    print(graph, file=f)
'''
######################################################################
# Move all resulting files to a directory
# ---------------------------------------
'''
import shutil

try:
    shutil.rmtree(resulting_file_directory)
except OSError as e:
    print("Preparing a directory for resulting files")

os.mkdir(resulting_file_directory)

shutil.move('kernel.txt', resulting_file_directory)
shutil.move('host_code.cc', resulting_file_directory)
shutil.move('device_code.cl', resulting_file_directory)
shutil.move('device_code.tvm_meta.json', resulting_file_directory)
shutil.move('binary_params.bin', resulting_file_directory)
shutil.move('graph.json', resulting_file_directory)
'''
######################################################################
# Get ImageNet lable
# ------------------

import tensorflow as tf
import numpy as np
labels_path = './ImageNetLabels.txt'
imagenet_labels = np.array(open(labels_path).read().splitlines())

######################################################################
# For comparison, define a function to get TFLite's prediction
# ------------------------------------------------------------
'''
def run_tflite_model(tflite_model_buf, input_data):
    """ Generic function to execute TFLite """
    try:
        from tensorflow import lite as interpreter_wrapper
    except ImportError:
        from tensorflow.contrib import lite as interpreter_wrapper

    input_data = input_data if isinstance(input_data, list) else [input_data]

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    assert len(input_data) == len(input_details)
    for i in range(len(input_details)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])

    # Run
    interpreter.invoke()

    # get output
    tflite_output = list()
    for i in range(len(output_details)):
        tflite_output.append(interpreter.get_tensor(output_details[i]['index']))

    return tflite_output
'''
######################################################################
# Calculate Top-1 and Top-5 Accuracy for TFLite
# ---------------------------------------------
# TFLite Top-1 accuracy: 0.00%; Top-5 accuracy: 18.00% (50 images)
'''
def get_tflite_accuracy(img_name):
    print("\n")
    
    # load image
    from PIL import Image
    print("img_name:", img_name)
    img = tf.keras.preprocessing.image.load_img(img_path+img_name, target_size=image_shape)
    # apply default data preprocessing of TFLite
    image_data = tf.keras.preprocessing.image.img_to_array(img)
    image_data = tf.keras.applications.vgg16.preprocess_input(image_data[tf.newaxis, ...])
    image_data = image_data.astype(data_type)
    
    # get TFLite's prediction
    tflite_res = run_tflite_model(tflite_model_buf, image_data)
    tflite_pred = np.squeeze(tflite_res).argsort()[-5:][::-1]
    print("TFLite's prediction:", tflite_pred)

    # set ground truth
    ground_truth = os.path.splitext(img_name)[0]
    ground_truth = ground_truth.split('-')[0]
    ground_truth = int(ground_truth)
    ground_truth += 1 # offset
    print("Ground truth: %s (ID: %d)" % (imagenet_labels[ground_truth], ground_truth))

    # print top-1
    #top1 = np.argmax(tvm_output[0])
    #print("Top-1: %s (ID: %d)" % (block.classes[top1], top1))
    # print top-5
    top5 = tflite_pred
    check_top1 = 0
    check_top5 = 0
    for top_id in range(5):
        if top_id == 0 and top5[top_id] == ground_truth:
            check_top1 = 1
        if top5[top_id] == ground_truth:
            check_top5 = 1
        print("Top-%d: %s (ID: %d) " % (top_id+1, imagenet_labels[top5[top_id]], top5[top_id]))
    return check_top1, check_top5

print("\nStart to estimate the accuracy using TFLite")
file_list = os.listdir(img_path)
top1_total = 0
top5_total = 0
for file_name in file_list:
    top1_subtotal, top5_subtotal = get_tflite_accuracy(file_name)
    top1_total += top1_subtotal
    top5_total += top5_subtotal
print("\nNum of tested images:", len(file_list))
top1_tflite, top5_tflite = top1_total/len(file_list), top5_total/len(file_list)
print("TFLite Top-1 accuracy: {:.2%}; Top-5 accuracy: {:.2%}".format(top1_total/len(file_list), top5_total/len(file_list)))
'''
######################################################################
# Create TVM runtime and do inference
# -----------------------------------
# TVM Top-1 accuracy: 60.00%; Top-5 accuracy: 84.00% (50 images)

from tvm.contrib import graph_runtime
def get_tvm_accuracy(graph, lib, params, ctx, img_name):
    print("\n")
    # create module
    module = graph_runtime.create(graph, lib, ctx)

    from PIL import Image
    print("img_name:", img_name)
    img = tf.keras.preprocessing.image.load_img(img_path+img_name, target_size=image_shape)
    # apply default data preprocessing of TFLite
    image_data = tf.keras.preprocessing.image.img_to_array(img)
    image_data = tf.keras.applications.vgg16.preprocess_input(image_data[tf.newaxis, ...])
    image_data = image_data.astype(data_type)

    # set ground truth
    ground_truth = os.path.splitext(img_name)[0]
    ground_truth = ground_truth.split('-')[0]
    ground_truth = int(ground_truth)
    ground_truth += 0
    print("Ground truth: %s (ID: %d)" % (imagenet_labels[ground_truth], ground_truth))

    # get raw input
    import shutil
    flatten_image_data = image_data.flatten()
    np.savetxt('input_' + img_name + '.txt', flatten_image_data, delimiter='\n')
    if os.path.exists(resulting_file_directory + 'input_' + img_name + '.txt'):
        os.remove(resulting_file_directory + 'input_' + img_name + '.txt')
    shutil.move('input_' + img_name + '.txt', resulting_file_directory)

    # set input and parameters
    module.set_input(input_name, tvm.nd.array(image_data))
    module.set_input(**params)

    # run
    import time
    timeStart = time.time()
    module.run()
    timeEnd = time.time()
    print("Inference time: %f" % (timeEnd - timeStart))

    # get output
    tvm_output = module.get_output(0).asnumpy()

    # print top-1
    #top1 = np.argmax(tvm_output[0])
    #print("Top-1: %s (ID: %d)" % (block.classes[top1], top1))
    # print top-5
    top5 = tvm_output[0].argsort()[-5:][::-1]
    print("TVM's prediction:", top5)
    check_top1 = 0
    check_top5 = 0
    for top_id in range(5):
        if top_id == 0 and top5[top_id] == ground_truth:
            check_top1 = 1
        if top5[top_id] == ground_truth:
            check_top5 = 1
        print("Top-%d: %s (ID: %d) " % (top_id+1, imagenet_labels[top5[top_id]], top5[top_id]))
    return check_top1, check_top5

print("\nStart to estimate the accuracy using TVM")
file_list = os.listdir(img_path)
top1_total = 0
top5_total = 0
for file_name in file_list:
    top1_subtotal, top5_subtotal = get_tvm_accuracy(graph, lib, params, ctx, file_name)
    top1_total += top1_subtotal
    top5_total += top5_subtotal
print("\nNum of tested images:", len(file_list))
print("TVM Top-1 accuracy: {:.2%}; Top-5 accuracy: {:.2%}".format(top1_total/len(file_list), top5_total/len(file_list)))
print("TFLite Top-1 accuracy: {:.2%}; Top-5 accuracy: {:.2%}".format(top1_tflite, top5_tflite))

print("Model:", model_path)

