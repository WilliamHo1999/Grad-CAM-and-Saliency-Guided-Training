import torch
import torch.nn as nn
import torch.nn.functional as F


class Saliency_loss(nn.modules.loss._Loss):

    def __init__(self, classification_loss, lamb = 1, multi_label = True):
        super(Saliency_loss, self).__init__()

        self.classification_loss = classification_loss
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target = False)
        self.lamb = lamb

        self.multi_label = multi_label

    def forward(self, input, target, masked_inputs):
        # Ended up treating multi-class and multi-label the same way.
        kl_input = F.log_softmax(masked_inputs, dim=1)
        kl_target = F.softmax(input, dim = 1)

        loss = self.classification_loss(input, target) + self.lamb * self.kl_loss(kl_input,kl_target)
        # useful to look at the loss sometimes
        #print(self.classification_loss(input, target).item(),"\t", self.lamb * self.kl_loss(kl_input, kl_target).item(), "\t", loss.item())
        return loss


class Masker(nn.Module):

    def __init__(self, model, optimizer, k_upper, multi_label = True, absolute_min = True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.model = model
        self.optimizer = optimizer

        self.multi_label = multi_label

        self.device = device

        self.k_upper = k_upper

        self.absolute_min = absolute_min

    def forward(self, x, labels = None):
        if self.multi_label == True:
            input_gradients = self._get_input_gradients_multi_label(x, labels)
        else:
            input_gradients = self._get_input_gradients(x, labels)

        sorted_gradients, sorted_index = self._sort(input_gradients)

        masked_input = self._mask(x, sorted_gradients, sorted_index)

        return masked_input

    def _get_input_gradients_multi_label(self, x, labels):
        
        x.requires_grad = True
        self.model.eval()
        
        output = self.model(x)
        #output.retain_grad()

        # Only want gradients for true classes
        masked_output = output * labels

        # Sum on a multi-label output to enable gradients.
        output_sum = masked_output.sum()

        output_sum.backward(retain_graph = True)
        self.optimizer.zero_grad()
        
        return x.grad

    def _get_input_gradients(self, x, label):
        

        x.requires_grad = True
        self.model.eval()    
        output = self.model(x)

        # Only want gradients for true classes for each sample
        targets = torch.zeros(x.shape[0]).to(self.device)
        for batch_index, batch in enumerate(output):
            targets[batch_index] = batch[label[batch_index]]
        # Sum to enable gradients to flow back - is the same as calling backward
        # on each of the samples in the for loop.
        targets.sum().backward(retain_graph = True)
        self.optimizer.zero_grad()

        return x.grad

    def _sort(self, gradient):
        """
        The paper replaced the gradients with the smallest values, but I'll use the smallest absolute values
        since a really negative gradient is crucial information, but a gradient close to 0 is not.
        """

        # Flatten gradients
        flat_gradient = gradient.view(gradient.shape[0], -1).detach()
                
        # Sort, and get the lowest gradients.
        if self.absolute_min:
            gradient_values, index = torch.topk(flat_gradient.abs(), self.k_upper, dim = 1, largest = False, sorted = False)
        else:    
            gradient_values, index = torch.topk(flat_gradient, self.k_upper, dim = 1, largest = False, sorted = False)
            
        return gradient_values, index

        
    def _mask(self, input_tensor, sorted_gradients, sorted_index):
        """
        As in the paper, replace low gradient values with random values within the feature range
        """

        input_copy = input_tensor.view(input_tensor.shape[0], -1).detach().clone()

        # Since a batch has several images, we need to mask them separetaly
        for batch in range(input_copy.shape[0]):
            prev_channel_start = 0
            channel_size = input_tensor.shape[2] * input_tensor.shape[3]

            for channel in range(input_tensor.shape[1]):
                
                # Since our gradients are in a 1D tensor, we need to split the channels by indexes
                channel_indexes = (channel+1)*(channel_size) -1
                this_channel = input_copy[batch,prev_channel_start:channel_indexes]
                # Find all top k elements within this channel.
                sorted_indexes_for_this_channel = torch.where((sorted_index[batch] <= channel_indexes) & (sorted_index[batch] >= prev_channel_start), sorted_index[batch], -1)
                sorted_indexes_for_this_channel = sorted_indexes_for_this_channel[sorted_indexes_for_this_channel!=-1]

                # Find features min max for this channel.
                min_feature = torch.min(input_copy[batch,prev_channel_start:channel_indexes]).item()
                max_feature = torch.max(input_copy[batch,prev_channel_start:channel_indexes]).item()

                # The k elements are split among 3 channels.
                num_elements_to_mask = len(sorted_indexes_for_this_channel)
                
                # Mask
                mask = torch.FloatTensor(num_elements_to_mask).uniform_(min_feature, max_feature).to(self.device)
                input_copy[batch][sorted_indexes_for_this_channel] = mask
                
                prev_channel_start = channel_indexes +1 
                

                

        input_copy = input_copy.view(input_tensor.shape)

        return input_copy

        
    def _replace_low_values(self, x, min, max):
        pass

