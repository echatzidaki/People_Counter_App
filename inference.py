#!/usr/bin/env python3
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        '''
        Initialize class variables
        '''
        
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.infer_request_handler = None


    def load_model(self, model, target_device, num_req, cpu_ext=None, plugin=None):
        """
         Loads a network and an image to the Inference Engine plugin.
        """
       

        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()
#         log.info("Initialize the plugin - IECore.")

        # Add a CPU extension, if applicable
#         if cpu_ext and "CPU" in target_device:
        self.plugin.add_extension(cpu_ext, target_device)
#             log.info("Add a CPU extension.")

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
#         log.info("Read the IR as a IENetwork.")
        
        # Check for unsupported layers
        if target_device == "CPU":
            self.check_supported_layers(self.network, self.plugin, target_device)
            
        # Load the IENetwork into the plugin
        if num_req == 0:
            self.net_plugin = self.plugin.load_network(network=self.network, device_name=target_device)
        else:
            self.net_plugin = self.plugin.load_network(network=self.network, num_requests=num_req, device_name=target_device)
            log.info("The IENetwork into the plugin loaded.")

        # Get the input layer
        try:
            self.input_blob = next(iter(self.network.inputs))
        except:
            print("Error in input_blob!")
#             log.info("Error in input_blob!")
            
        try:
            self.output_blob = next(iter(self.network.outputs))
        except:
            print("Error in output_blob!")
#             log.info("Error in output_blob!")
            
#         log.info("Input/Output_blob ok.")
        

        return self.get_input_shape()

    
    def check_supported_layers(self, network, plugin, device):
        '''
        Check for if all layers are supported or exit and warn the user
        '''
        
        # Get the supported layers of the network
        supported_layers = self.plugin.query_network(network, device)
#         log.info("Get the supported layers of the network.")
        
        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
#             log.info("Fail--Unsupported layers found!")
            exit(1)
            
        return 
    
    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        
        # Return the shape of the input layer 
#         log.info("Input shape of the network.")
        return self.network.inputs[self.input_blob].shape


    def exec_net(self, image, req_id=0):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        
        # Start an asynchronous request
#         self.infer_request_handle = 
        self.net_plugin.start_async(request_id=0, inputs={self.input_blob: image})
#         log.info("Started an asynchronous request.")
        return self.net_plugin


    def wait(self, req_id=0):
        '''
        Checks/Waits the status of the inference request to not be used.
        '''
        
        # Wait for the request to be complete. 
        wait_req = self.net_plugin.requests[req_id].wait(-1)
#         log.info("Waited for the request.")
        return wait_req


    def get_output(self, req_id=0, output=None):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        
        # Extract and return the output results
        if output:
            out = self.infer_request_handle.outputs[output]
        else:
            out = self.net_plugin.requests[req_id].outputs[self.output_blob]
        return out
    

    def clean(self):
        '''
        Deletes/cleans all the instances.
        '''
        
        # Delete the following class variables
        del self.network
        del self.plugin
        del self.net_plugin
        del self.output_blob
        del self.input_blob
#         log.info("Class variables deleted.")
        
    
