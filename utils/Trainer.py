import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import os
import sys
from tqdm import tqdm

class Trainer():

    def __init__(self, model, loss_fn, classes, saliency_guided_training = False, device = 'gpu', num_epochs = 1, model_name=None, full_train=False, model_params = None, multi_label = False, sailency = False, masker = None):
        

        self.saliency_guided_training = saliency_guided_training

        self.model = model
        self.device = device
        self.model_name = model_name
        self.model_params = model_params
        self.class_names = classes
        self.num_classes = len(classes)
        self.classes = [i for i in range(self.num_classes)]
        self.full_train = full_train

        if self.device == "gpu":
            self.model.to("cuda")
            self.data_device = "cuda"
        else:
            self.model.to("cpu")
            self.data_device = "cpu"

        self.loss_fn = loss_fn
        self.data_loader = model_params['data_loader']
        self.num_epochs = num_epochs

        self.epoch_loss = {'train': [], 'val': []}
        self.epoch_mape = {'train': [], 'val': []}
        self.epoch_acc = {'train': [], 'val': []}
        self.confusion_matrix = {
            "train": np.zeros(shape=(self.num_classes , self.num_classes , self.num_epochs)),
            "val": np.zeros(shape=(self.num_classes , self.num_classes , self.num_epochs)),
        }
        self.epoch_ap = {'train': np.zeros((1,self.num_classes)), 'val': np.zeros((1,self.num_classes))}

        self.best_model = None
        self.best_measure = 0
        self.train_measure_at_best = 0
        self.train_loss_at_best = 0
        self.best_eval_loss = 0
        self.best_epoch = 0
        self.latest_epoch = 0

        self.optimizer = model_params['optimizer']
        self.scheduler = model_params['scheduler']

        self.stop_iteration = 0

        self.multi_label = multi_label

        self.sailency = sailency

        self.masker = masker


    def train(self):

        for current_epoch in range(self.num_epochs):
            self.latest_epoch = current_epoch

            self.model.train()
            train_dict = self.run_epoch(
                self.model,
                current_epoch,
                is_training=True,
                mode='train',
            )

            self.model.eval()
            eval_dict = self.run_epoch(
                self.model,
                current_epoch,
                is_training=False,
                mode='val',
            )

            self.scheduler.step()

            eval_measure = eval_dict['measure']

            if eval_measure > self.best_measure:
                self.stop_iteration = 0
                self.best_measure = eval_measure
                self.train_loss_at_best = train_dict['loss']
                self.train_measure_at_best = train_dict['measure']
                self.best_model = self.model
                self.best_loss = self.epoch_loss['val'][-1]
                self.best_epoch = current_epoch

                if self.full_train:
                    assert self.best_model is not None
                    pt = './best_models'
                    if not os.path.isdir(pt):
                        os.makedirs(pt)
                    torch.save(
                        self.best_model.state_dict(),
                        pt + "/" + self.model_name,
                        #self.model_name
                    )
                    #self.model_path = pt + "/" + self.model_name
                    self.model_path = self.model_name

            else:
                self.stop_iteration += 1
                
                # No improvement in 5 iterations
                if self.stop_iteration == 6:
                    return

            # print progress
            if self.multi_label:
                print("Current Epoch:",current_epoch)
                print("Eval  Model: ", self.model_name, ". MAPE: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
                print("Train Model: ", self.model_name, ". MAPE", self.epoch_mape['train'][-1], "Avg loss:",  self.epoch_loss['train'][-1])
                print("Current MAPE: ",self.best_measure, self.best_loss, "at epoch",self.best_epoch)
                for ii in range(len(self.classes)):
                    ap = self.epoch_ap['val'][-1, ii]
                    print(
                        f'AP of {str(self.class_names[ii]).ljust(30)}: {ap}'
                        f'{ap*100:.01f}%'
                    )
            else:
                print("Current Epoch:",current_epoch)
                print("Eval  Model: ", self.model_name, ". Acc: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
                print("Train Model: ", self.model_name, ". Acc", self.epoch_acc['train'][-1], "Avg loss:",  self.epoch_loss['train'][-1])
                print("Current acc: ",self.best_measure, self.best_loss, "at epoch",self.best_epoch)
                for ii in range(len(self.classes)):
                    acc = self.confusion_matrix['val'][ii, ii, current_epoch] / np.sum(
                        self.confusion_matrix['val'][ii, :, current_epoch], )
                    print(
                        f'Accuracy of {str(self.class_names[ii]).ljust(30)}: {acc}'
                    )


    def run_epoch(self, model, current_epoch, is_training, mode):
        """
        Run epoch
        """
         
        if self.device == "gpu":
            torch.cuda.empty_cache()

        total_loss = 0
        total_samples = len(self.data_loader[mode].dataset)
        total_batches = len(self.data_loader[mode])
        total_correct = 0

        if self.multi_label:
            concat_pred = np.zeros((1,self.num_classes))
            concat_labels = np.zeros((1,self.num_classes))
            avgprecs = np.zeros(self.num_classes)
        else:
            confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes))

            
        for index, data in enumerate(self.data_loader[mode]):
            if is_training:
                inputs = data.get('image').to(self.data_device)
                labels = data.get('label').to(self.data_device)

                if self.sailency:
                    masked_input = self.masker(inputs, labels)
                    self.model.train()
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    masked_output = self.model(masked_input)
                    loss = self.loss_fn(outputs, labels, masked_output)

                else:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

            else:
                if self.sailency:
                    inputs = data['image'].to(self.data_device)   
                    labels = data['label'].to(self.data_device) 
                    outputs = self.model(inputs)
                    masked_input = self.masker(inputs, labels)
                    self.model.eval()

                    with torch.no_grad():
                        masked_output = self.model(masked_input)
                        
                        loss = self.loss_fn(outputs, labels, masked_output)
                       

                else:
                    with torch.no_grad():
                        inputs = data['image'].to(self.data_device)       
                        labels = data['label'].to(self.data_device)
                        
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, labels) 
                
                


            total_loss += loss.item()
            if self.multi_label:
                cpuout= outputs.detach().to('cpu')
                pred_scores = cpuout.numpy() 
                concat_pred = np.append(concat_pred, pred_scores, axis = 0)
                concat_labels = np.append(concat_labels, labels.cpu().numpy(), axis = 0)
            else:
                preds = torch.argmax(outputs, dim = 1)
                total_correct += preds.eq(labels).cpu().sum().numpy()
            
                confusion_matrix += metrics.confusion_matrix(
                    labels.cpu().numpy(), preds.cpu().numpy(), labels=self.classes,
                )

            if is_training:
                loss.backward()
                self.optimizer.step()
                # Just to make sure not to mess with the masker
                self.optimizer.zero_grad()


        if self.multi_label:
            concat_pred = concat_pred[1:,:]
            concat_labels = concat_labels[1:,:]
            
            for c in range(self.num_classes):   
                avgprecs[c]=  metrics.average_precision_score(concat_labels[:,c], concat_pred[:,c])

            measure = np.mean(avgprecs)
            self.epoch_mape[mode].append(measure)
            self.epoch_ap[mode] = np.append(self.epoch_ap[mode], avgprecs.reshape(1,-1), axis = 0)
        else:
            measure = total_correct/total_samples
            self.epoch_acc[mode].append(measure)
            self.confusion_matrix[mode][:, :, current_epoch] = confusion_matrix
        
        

        loss = total_loss/total_batches

        self.epoch_loss[mode].append(loss)

        return {"measure": measure, "loss": loss}