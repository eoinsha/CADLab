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

import os
import torch
import torch.nn as nn
from torchvision import models, transforms

from torch.utils.data import DataLoader
from CXR_Data_Generator import DataGenerator

import matplotlib as mpl
mpl.use('Agg')

arch = os.getenv('ARC', 'densenet121')
img_size = int(os.getenv('IMG_SIZE', 256))
crop_size = int(os.getenv('CROP_SIZE', 224))
epoch = int(os.getenv('EPOCH', 50))
batch_size = int(os.getenv('BATCH_SIZE', 64))
# num_workers is set to 0 by default to turn of /dev/shm usage.
# This will make loading less efficient when processing multiple images.
num_workers = int(os.getenv('NUM_DATA_WORKERS', 0))
learning_rate = float(os.getenv('LEARNING_RATE', 0.001))
test_labels = os.getenv('TEST_LABELS', 'att')
model_path = os.getenv('MODEL_PATH', './trained_models_nih')
num_class = 1

model = models.__dict__[arch](pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_class)

# model = model.cuda()

model_path = f'{model_path}/{arch}_{img_size}_{batch_size}_{learning_rate}'
split_file_dir = './dataset_split'
split_name = 'test'
splits = [split_name]
model.load_state_dict(torch.load(
    model_path, map_location=torch.device('cpu'))['state_dict'])
split_file_suffix = '_attending_rad.txt' if test_labels == 'att' else '_rad_consensus_voted3.txt'
split_files = {}
split_test = os.path.join(split_file_dir, 'test' + split_file_suffix)


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
                                shuffle=False, num_workers=num_workers, pin_memory=True)

    dataloaders = {}
    dataloaders[split_name] = dataLoaderTest

    print('Number of testing CXR images: {}'.format(len(datasetTest)))

    # -------------------- TESTING -------------------
    model.eval()
    output_list = []
    preds_list = []
    scores_list = []

    with torch.no_grad():
        # Iterate over data.
        for data in dataloaders[split_name]:
            inputs, img_names = data

            # forward
            outputs = model(inputs)
            # _, preds = torch.max(outputs.data, 1)
            score = torch.sigmoid(outputs)
            score_np = score.data.cpu().numpy()
            preds = score > 0.5
            preds_np = preds.data.cpu().numpy()
            # preds = preds.type(torch.cuda.LongTensor)
            preds = preds.type(torch.LongTensor)

            outputs = outputs.data.cpu().numpy()

            for j in range(len(img_names)):
                print(str(img_names[j]) + ': ' + str(score_np[j]))

            for i in range(outputs.shape[0]):
                output_list.append(outputs[i].tolist())
                scores_list.append(score_np[i].tolist())
                preds_list.append(preds_np[i].tolist())

    res = {
        'jobs': jobs,
        'output_list': output_list,
        'preds_list': preds_list,
        'scores_list': scores_list,
    }
    return res
