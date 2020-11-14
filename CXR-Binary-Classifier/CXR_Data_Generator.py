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
from io import BytesIO
import os

import logging

import boto3
from torch.utils.data import Dataset
from PIL import Image

BUCKET_NAME = os.environ['BUCKET_NAME']

s3_client = boto3.client('s3')
logger = logging.getLogger(__name__)


class DataGenerator(Dataset):

    def __init__(self, jobs, transform):
        self.transform = transform
        self.jobs = jobs

    def __getitem__(self, index):
        print(f'Index: {index}')
        key = self.jobs[index]['Key']
        logger.info(f'Fetching {key}')
        obj = BytesIO()
        s3_client.download_fileobj(Bucket=BUCKET_NAME, Key=key, Fileobj=obj)
        obj.seek(0)
        image_data = Image.open(obj).convert('RGB')
        image_data = self.transform(image_data)
        # image_label= torch.FloatTensor(self.img_label_list[index])
        image_label = int(self.jobs[index]['Label'])
        return (image_data, image_label, key)

    def __len__(self):
        return len(self.jobs)
