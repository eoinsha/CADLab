"""
Yuxing Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
March 2020

THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
# from sklearn.metrics import roc_auc_score
# from PIL import Image
# import time

from torch.utils.data import DataLoader
# from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from CXR_Data_Generator import DataGenerator

import matplotlib as mpl
mpl.use('Agg')

arch = os.getenv('ARC', 'densenet121')
img_size = int(os.getenv('IMG_SIZE', 256))
crop_size = int(os.getenv('CROP_SIZE', 224))
epoch = int(os.getenv('EPOCH', 50))
batch_size = int(os.getenv('BATCH_SIZE', 64))
learning_rate = float(os.getenv('LEARNING_RATE', 0.001))
test_labels = os.getenv('TEST_LABELS', 'att')
num_class = 1

model = models.__dict__[arch](pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_class)

model = model.cuda()

model_path = f'./trained_models_nih/{arch}_{img_size}_{batch_size}_{learning_rate}'
split_file_dir = './dataset_split'
split_name = 'test'
splits = [split_name]
model.load_state_dict(torch.load(model_path)['state_dict'])
split_file_suffix = '_attending_rad.txt' if test_labels == 'att' else '_rad_consensus_voted3.txt'
split_files = {}
split_test = os.path.join(split_file_dir, 'test' + split_file_suffix)
gpu_id = 'cpu'  # TODO


def run_test(jobs):
    normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    data_transforms = {split_name: transforms.Compose([
        transforms.Resize(img_size),
        # transforms.RandomResizedCrop(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(normalizer[0], normalizer[1])])}

    # -------------------- SETTINGS: DATASET BUILDERS -------------------
    datasetTest = DataGenerator(jobs, transform=data_transforms[split_name])
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size,
                                shuffle=False, num_workers=32, pin_memory=True)

    dataloaders = {}
    dataloaders[split_name] = dataLoaderTest

    print('Number of testing CXR images: {}'.format(len(datasetTest)))
    dataset_sizes = {split_name: len(datasetTest)}

    # -------------------- TESTING -------------------
    model.eval()
    running_corrects = 0
    output_list = []
    label_list = []
    preds_list = []

    with torch.no_grad():
        # Iterate over data.
        for data in dataloaders[split_name]:
            inputs, labels, img_names = data

            labels_auc = labels
            labels_print = labels
            labels_auc = labels_auc.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)  # add for BCE loss

            # TODO: wrap them in Variable
            # inputs = inputs.cuda(gpu_id, non_blocking=True)
            # labels = labels.cuda(gpu_id, non_blocking=True)
            # labels_auc = labels_auc.cuda(gpu_id, non_blocking=True)

            labels = labels.view(labels.size()[0], -1)  # add for BCE loss
            labels_auc = labels_auc.view(
                labels_auc.size()[0], -1)  # add for BCE loss
            # forward
            outputs = model(inputs)
            # _, preds = torch.max(outputs.data, 1)
            score = torch.sigmoid(outputs)
            score_np = score.data.cpu().numpy()
            preds = score > 0.5
            preds_np = preds.data.cpu().numpy()
            preds = preds.type(torch.cuda.LongTensor)

            labels_auc = labels_auc.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()

            for j in range(len(img_names)):
                print(str(img_names[j]) + ': ' + str(score_np[j]
                                                     ) + ' GT: ' + str(labels_print[j]))

            for i in range(outputs.shape[0]):
                output_list.append(outputs[i].tolist())
                label_list.append(labels_auc[i].tolist())
                preds_list.append(preds_np[i].tolist())

            # running_corrects += torch.sum(preds == labels.data)
            # labels = labels.type(torch.cuda.FloatTensor)
            # add for BCE loss
            running_corrects += torch.sum(preds.data == labels.data)

    acc = np.float(running_corrects) / dataset_sizes[split_name]
    auc = metrics.roc_auc_score(np.array(label_list), np.array(output_list), average=None)
    # print(auc)
    fpr, tpr, _ = metrics.roc_curve(np.array(label_list), np.array(output_list))
    roc_auc = metrics.auc(fpr, tpr)

    ap = metrics.average_precision_score(np.array(label_list), np.array(output_list))

    tn, fp, fn, tp = metrics.confusion_matrix(label_list, preds_list).ravel()

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2*precision*recall/(precision+recall)
    sensitivity = recall
    specificity = tn/(tn+fp)
    PPV = tp/(tp+fp)
    NPV = tn/(tn+fn)
    print('Test Accuracy: {0:.4f}  Test AUC: {1:.4f}  Test_AP: {2:.4f}'.format(
        acc, auc, ap))
    print('TP: {0:}  FP: {1:}  TN: {2:}  FN: {3:}'.format(tp, fp, tn, fn))
    print('Sensitivity: {0:.4f}  Specificity: {1:.4f}'.format(
        sensitivity, specificity))
    print('Precision: {0:.2f}%  Recall: {1:.2f}%  F1: {2:.4f}'.format(
        precision*100, recall*100, f1))
    print('PPV: {0:.4f}  NPV: {1:.4f}'.format(PPV, NPV))
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of abnormal/normal classification: '+arch)
    plt.legend(loc="lower right")
    plt.savefig(f'ROC_abnormal_normal_cls_{arch}_{test_labels}.pdf', bbox_inches='tight')
    plt.show()
