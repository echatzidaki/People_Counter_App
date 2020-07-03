"""People Counter. Eleftheria Chatzidaki"""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
RUN

python3 main.py -m models/intel-models/retail-13/person-detection-retail-0013.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

"""

    
import os
import sys
import time
import socket
import json
import cv2
import math

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from random import randint
# import winsound


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

DEF_INPUT_STREAM = "/resources/Pedestrian_Detect_2_1_1.mp4"
DEF_CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
LE_MODEL = "/home/workspace/models/pedestrian-detection-adas-0002.xml"
DEF_TARGET_DEVICE = "CPU"
DEVICES = ["CPU", "GPU", "MYRIAD", "HETERO:FPGA,CPU", "HDDL"]

# Weightage/ratio to merge (for integrated output) people count frame and colorMap frame(sum of both should be 1)
P_COUNT_FRAME_WEIGHTAGE = 0.65
COLORMAP_FRAME_WEIGHTAGE_1 = 0.35

def_prob = 0.5

input_type = None
truly_async_mode = False


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    
    return parser


def globs_args(args):
    """
    Some checks and pass global variables to  global_vars file
    :return: command line arguments
    """
#     log.info("Check command line args...and pass them to global_vars file...")
   
    if 'MULTI' in args.device:
        targ_devices = args.device.split(':')[1].split(',')
        for devices in targ_devices:
            if devices not in DEVICES:
                print("Unsupported device: " + args.device)
#                 log.info("Unsupported device!")
                sys.exit(1)
        
    return args


def get_input(input_stream):
    
    # Normalise any letter in input file extension to lowercase
    input_stream = input_stream.lower()
    # file_extension = os.path.splitext(input_stream)
    
    input_type = None
    # Check if input file has video extension from VIDEO_EXT_LIST
    if input_stream.endswith('.mp4') :
        #         log.info("It's a video file...")
        input_type = 2
    elif input_stream.endswith('.jpg') or input_stream.endswith('.jpeg') or input_stream.endswith('.png'):
        input_type = 1        
    elif input_stream.lower() == "cam":
        input_type = 0
    else:
        print("Please check again input file or cam. Input is unknown. If it is video or image, maybe you should update extension list -(global_vars.py).")
        print("File: ", os.path.isfile(input_stream) )
        # log.info("Please check again input file or cam.")
        sys.exit(1)
               
    return input_type


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


# Create a video writer for the output video
def video_writer(h, w):

    if input_type == 0 :
        # Mac uses cv2.VideoWriter_fourcc('M','J','P','G') to write an .mp4 file
        # Linux uses 0x00000021 to write an .mp4 file
        # wxh to match desired resizing
        out = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (w, h))
    else:
        out = None
        
    return out
# Change image-data layout from HWC to CHW        
def preprocessing_image(in_image, h, w):  
    
    in_image = cv2.resize(in_image, (w, h))
    in_image = in_image.transpose((2, 0, 1))
    in_image = in_image.reshape(1, *in_image.shape)
#     log.info("HWC to CHW - Preprocessing, Done...")
    
    return in_image

def draw_bounding_box(res_output, frame, h, w, prob_threshold, en_dist):
    """
    Some checks and pass global variables to  global_vars file
    :return: command line arguments
    """
    people_counter = 0
    cam_distance = en_dist
    for obj in res_output[0][0]:
        # Draw objects only when probability is more than specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * w)
            ymin = int(obj[4] * h)
            xmax = int(obj[5] * w)
            ymax = int(obj[6] * h)
            # Draw the bounding box
#             class_id = int(obj[1])
#             color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
            color = (255, 50, 50)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            # Update detected human counter 
            people_counter = people_counter + 1
            
            cord_x = (xmax + xmin)/2 - (frame.shape[1])/2
            cord_y = (ymax + ymin)/2 - (frame.shape[0])/2
            cam_distance = math.sqrt((cord_x)**2 + (cord_y)**2 ) 
    return people_counter, frame, cam_distance


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    args = globs_args(args)
    input_type = get_input(args.input)
    
    # render_time = 0      
    truly_async_mode = False
    threshold_frames = 10
    
    # Initialise the class
    inf_net = Network()
    
    # Number of request id(s)
    n_req=1
    
    # Async mode: while waiting for Current inference request to finish, start Next inference request
    req_id = 0
    next_req_id = 1

    # Load the model through `infer_network` into the IE #
    # One variable for each dimension, 
    # n: number of image
    # c: color layer/s of the image (1: grayscale, 3: RGB) 
    # h: height of image
    # w: width of imge
    n, c, h, w = inf_net.load_model(args.model, args.device, n_req, args.cpu_extension)
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.input)
    assert cap.isOpened(), "Fail! Can not open: " + args.input
#     log.info("Succeed! Input is Opened...")

    # Input stream (is) - Grab the shape of the input 
    is_width = int(cap.get(3))
    is_height = int(cap.get(4))
    
    # if not input_type == 1:
        # out = cv2.VideoWriter('out_video.mp4', 0x00000021, 30, (100,100)) 
#         log.info("Created a video writer for the output video ...")


    # Variables
    total_people_counter = 0
    prev_people_counter = 0
    distance = 0
    en_dist = 0
    f_frame_time = 0
    # nf_frame_time = 0
    frame_count = 0
    n_frame_count = 0
    dur_time = 0
    stay_time = 0
    stay = 0
    in_people_counter = 0
    duration_time = 0
    
    if truly_async_mode:
        ret, frame = cap.read()
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
#         log.info("While input is Opened...")
        # Capture frame-by-frame. 
        # Frame will get the next frame in the camera
        # Ret will obtain return value from getting the camera frame or not - boolean
        if truly_async_mode:
            ret, next_frame = cap.read()
            n_frame_count +=1
            ns_frame_time = time.time()
        else:
            ret, frame = cap.read()
            frame_count +=1
            s_frame_time = time.time()
        if not ret:
#             log.info("Can not find flag...Finish...")
            break
        
        key_pressed = cv2.waitKey(60)
             
        # HWC to CHW        
#         input_stream = preprocessing_image(frame, h, w)
        # Start async inference for specified request
        inf_start = time.time()
#         inf_net.exec_net(input_stream, req_id)
        
        if truly_async_mode:
            # HWC to CHW        
            out_next_frame = preprocessing_image(next_frame, h, w)
            inf_net.exec_net(next_frame, next_req_id)
            
        else:
            # HWC to CHW        
            out_frame = preprocessing_image(frame, h, w)
            inf_net.exec_net(out_frame, req_id)
        # Wait for the result
        if inf_net.wait(req_id) ==0:
            # End async inference for specified request
            inf_end = time.time()
            det_time = inf_end - inf_start
            
            res_output = inf_net.get_output(req_id=req_id)
            
            people_counter, frame, distance = draw_bounding_box(res_output, frame, is_height, is_width, args.prob_threshold, en_dist)
            
            if truly_async_mode:
                inf_time_message = "Inference time: N\A for truly async mode"
                cv2.putText(frame, inf_time_message, (100, 20), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)
                truly_async_message = "Truly Async mode. Processing request with id: {}".format(n_req_id)
                cv2.putText(frame, truly_async_message, (100, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)
            else:
                inf_time_message = "Inference time: {:.3f}".format(det_time * 1000)
                cv2.putText(frame, inf_time_message + "ms", (100, 20), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)
                regular_async_message = "Reg Async mode. Request id: {}".format(req_id)
                cv2.putText(frame, regular_async_message, (100, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)
            
            distance_message = "Distance: {:.2f} ".format(distance)
            cv2.putText(frame, distance_message, (100, 80), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)
        
            f_frame_time += time.time() - s_frame_time
            fps_count = frame_count/ float(f_frame_time)
            fps_time_message = "FPS: ({:.2})".format(fps_count)
            cv2.putText(frame, fps_time_message, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)

            people_count_message = "People Count: {} ".format(people_counter)
            cv2.putText(frame, people_count_message, (100, 60), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)

            if people_counter == prev_people_counter:
                stay +=1
                if stay >= threshold_frames:                    
                    if prev_people_counter > in_people_counter and stay == threshold_frames:                        
                        total_people_counter += people_counter - in_people_counter 
                    if prev_people_counter < in_people_counter and stay == threshold_frames:
                        duration_time = int((dur_time/threshold_frames)*1000)
                        if duration_time is not None:
                            # Send the class name: Person duration and its time-number to the MQTT server
                            client.publish("person/duration", json.dumps({"duration": duration_time}))
                    stay_time = int((stay/threshold_frames))
                    if people_counter > 0 and (stay_time > threshold_frames):  
                        alert_message = "ALERT! Still here (>10 sec)!: {}".format(stay_time) + " sec"
                        cv2.putText(frame, alert_message, (100, 140), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)
                client.publish("person", json.dumps({"count": people_counter, "total": total_people_counter}))
            else:                
                in_people_counter = prev_people_counter
                prev_people_counter = people_counter
                if stay > threshold_frames:
                    dur_time = stay
                    stay = 0
                else:
                    dur_time += stay
            total_people_counter_message = "Total: {}".format(total_people_counter)
            cv2.putText(frame, total_people_counter_message, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 0.45, (250, 0, 0), 1)
            
            en_dist = distance         
     
        ### Send frame to the ffmpeg server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
            
        # Write out the frame, depending on image or video
        if input_type == 1:
             cv2.imwrite('output_image.jpg', frame)
        # else:
            # frame = cv2.resize(frame, (100,100))
            # frame = cv2.Canny(frame, 100, 200)
            # out.write(frame)

        
#         render_start = time.time()
#         cv2.imshow("Detection Results", frame)
#         render_end = time.time()
#         render_time = render_end - render_start

        if truly_async_mode:
            req_id = next_req_id
            next_req_id = req_id
            frame = next_frame

#         if not input_type == 1:
#             out.release()   
            
        # Break if escapturee key pressed
        if key_pressed == 27:
                    break
        
    # Release the capture and destroy any OpenCV windows
    # if input_type == 0 or input_type == 2:
       # out.release()
    cap.release()
    cv2.destroyAllWindows()    
    client.disconnect()

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
           


if __name__ == '__main__':
        main()

        