import cv2
import numpy as np

import torch
import torch.nn as nn

class GradCam(nn.Module):

    def __init__(self, model, target_layer, target_class, multi_label = True, no_relu = False, negative_activation = False):
        super().__init__()

        self.model = model.eval()
        self.target_layer = target_layer
        self.target_class = target_class

        self.activation = None
        self.gradient = None

        self.activations = []
        self.gradients = []

        self.multi_label = multi_label
        self.no_relu = no_relu
        self.negative_activation = negative_activation


        self.target_layer.register_forward_hook(self.get_activation)
        self.target_layer.register_backward_hook(self.get_gradients)

    def get_activation(self, module, input, output):
        """
        Output: Output of forward() of the module
        """
        self.activation = output
        #self.activations.append(output)

    def get_gradients(self, module, grad_input, grad_output):
        """
        grad_output: gradients with respect to the output of the layer
        """
        self.gradient = grad_output[0]
        #self.gradients = [grad_output[0]] + self.gradients


    def forward(self, input, target_class = None):
        print("Focus class:", target_class)

        output = self.model(input)

        if self.multi_label:
            output = torch.sigmoid(output)
            pred = output > 0.5
        else:
            pred = torch.argmax(output, axis = 1)

        self.model.zero_grad()
        
        if target_class is not None:
            target_output = output[:, target_class]
        else:
            target_output = output[:, self.target_class]
        
        target_output.backward(retain_graph = True)

        
        # Compute importance weight over width and heigth
        weights = torch.mean(self.gradient, axis = (2,3)).reshape(-1,1,1)

        # Compute class discriminative localization map
        # loc_map = self.activations[-1] * weights
        loc_map = self.activation.squeeze() * weights
        loc_map = torch.sum(loc_map, axis = 0)
        if self.no_relu:
            pass
        elif self.negative_activation:
            loc_map = torch.relu(-1*loc_map)
        else:
            loc_map = torch.relu(loc_map)

        heatmap = cv2.resize(loc_map.cpu().detach().numpy(), (input.shape[3], input.shape[2]))

        return heatmap, pred
        
