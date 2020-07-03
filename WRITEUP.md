# Project Write-Up

## Download and Convert the Models

https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model.html

### Tensorflow
 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

#### ssd_mobilenet_v1_coco

1. Download: 
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

2. Decompress: 
tar xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz

3. Model Optimizer: 
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v1_coco_2018_01_28/pipeline.config -o ssd_models/v1

#### ssd_mobilenet_v2_coco

1. Download: 
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

2. Decompress: 
tar xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

3. Model Optimizer: 
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config -o ssd_models/v2

#### ssdlite_mobilenet_v2_coco

1. Download: 
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

2. Decompress: 
tar xvzf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

3. Model Optimizer: 
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config -o ssd_models/v2-lite

#### ssd_inception_v2_coco

1. Download: 
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz

2. Decompress: 
tar xvzf ssd_inception_v2_coco_2018_01_28.tar.gz

3. Model Optimizer: 
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config -o ssd_models/v2-inc

#### faster_rcnn_inception_v2_coco

1. Download: 
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

2. Decompress: 
tar xvzf ssd_faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

3. Model Optimizer: 
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api_pipeline_config ssd_faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config -o ssd_models/v2-inc

--------------------------------------------------

### Onnx

https://github.com/onnx/models
or 
https://github.com/onnx/onnx-mxnet/blob/master/onnx_mxnet/tests/test_models.py

#### bvlc_alexnet

1. Download: 
wget https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz

2. Decompress: 
tar xvzf bvlc_alexnet.tar.gz

3. Model Optimizer: 
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx

#### Single Stage Detector - SSD

1. Download: 
wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd/model/ssd-10.tar.gz

2. Decompress: 
tar xvzf ssd-10.tar.gz

3. Model Optimizer: 
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx

#### Inception_V2

1. Download: 
wget https://s3.amazonaws.com/download.onnx/models/inception_v2.tar.gz

2. Decompress: 
tar xvzf inception_v2.tar.gz

3. Model Optimizer: 
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx


#### MobileNet v2

1. Download: 
wget https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz

2. Decompress: 
tar xvzf mobilenetv2-7.tar.gz

3. Model Optimizer: 
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model mobilenetv2-1.0.onnx

--------------------------------------------------

### Intel-OpenVINO Pretrained Models

https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/pretrained-models.html

#### High-Angle Detection

1. To navigate to the directory containing the Model Downloader: 

cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

2. Downloading Model: 
sudo ./downloader.py --name person-detection-retail-0013 -o /home/workspace

#### Pedestrian Detection

1. To navigate to the directory containing the Model Downloader: 

cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

2. Downloading Model: 
sudo ./downloader.py --name pedestrian-detection-adas-0002 -o /home/workspace


## Explaining Custom Layers

I did not have custom layers.

## Comparing Model Performance

The difference between the accuracy of the models was depended on how I could use the results in my code. The simple version that worked perfectly with person-detection-retail-0013.xml, did not meet my expectations with the others. I should include many more things like the distance and keep track of the person with multiple parameters. Almost all the models lose human appearance when the person is stable. The accuracy was a tragedy... However, the pedestrian-detection-adas-0002.xml was a disappointment,ent as it tracked a person as multiple people or it also missed them sometimes. I think the person-detection-retail-0013.xml was perfectly made for the current video and similar simple occasions.


The size of the model pre- and post-conversion was:
(ls -l --block-size=K)
| Model Name         		| Size pre- |  Size post-  | 
| ----------------------------- | ----------------- |-----------|
| **Tensorflow** 	|     |   |
| ssd_mobilenet_v1_coco|  27.7   |  26 |
| ssd_mobilenet_v2_coco|  66.5   |   64 |
| ssd_inception_v2_coco |  98 |  96  |
| **Intel**||   |
| adas-0002 (FP32)|  -  | 5M  | 
| retail-0013 (FP32)|  -  |  2M | 


The inference time of the model pre- and post-conversion was:

| Model Name  | Inference pre- (ms) |  Inference post- (ms)  | 
| ----------------------------- | ----------------- |-----------|
| **Tensorflow** 	|     |   |
| ssd_mobilenet_v1_coco|  42**   |  51 |
| ssd_mobilenet_v2_coco|  30**   |   72 |
| ssd_inception_v2_coco |  26** |  159  |
| **Intel**||   |
| adas-0002 (FP32)|  -  |  56 | 
| retail-0013 (FP32)|  -  | 49  |

**https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

## Assess Model Use Cases

Some of the potential use cases of the people counter app are at the statistic services of a museum to rank the items depend on the attention of the visitors or at retail to track people's interest and mine data about how customers decide what to buy or their second or even third choices. And many other occasions especially now with the COVID-19 that people should keep some meters distance in stores, buildings, etc.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

The lightning, the angle of the object/person, or even the camera focal length/image size (especially with very bad resolution) can change the shape and the features of it and the system could potentially ignore it. For example a faded figure from the sunlight, or a person too far away or a human in a strange position, it is hard to get the proper attention and the system detects it.
The model accuracy is very important as the model should not miss, in our scenario people, or detects other objects as people. The model should be also trained to find people in many positions. A trained model of 2 pictures can not be a good example, as well as an overtrained. We also should be aware of the resources(ie. memory) and the system that we use. 


