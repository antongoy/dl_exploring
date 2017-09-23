import time
import argparse
import requests

import numpy as np

from PIL import Image

from torch.nn import Softmax2d
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import squeezenet1_1, resnet152, vgg19_bn

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

labels = {
    int(key): value for (key, value) in requests.get(LABELS_URL).json().items()
}

parser = argparse.ArgumentParser()
parser.add_argument('image_path', metavar='PATH', help='Path to the image')
parser.add_argument('--top-n', default=5, type=int, help='Print top n results')
parser.add_argument('--net', default='squeeze', help='Type of network')

args = parser.parse_args()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

with open(args.image_path, 'rb') as image_file:
    image = Image.open(image_file)

    image_tensor = preprocess(image)
    image_tensor.unsqueeze_(0)
    image_variable = Variable(image_tensor)

if args.net == 'squeeze':
    net = squeezenet1_1(pretrained=True)
elif args.net == 'resnet152':
    net = resnet152(pretrained=True)
elif args.net == 'vgg19':
    net = vgg19_bn(pretrained=True)

net.cuda()

start_time = time.monotonic()

fc_out = net(image_variable)
scores = fc_out.data.numpy()
scores = scores.squeeze()
sorted_labels_idx = np.argsort(scores)

end_time = time.monotonic()

print('Top results:')
for i in range(1, args.top_n + 1):
    label_idx = sorted_labels_idx[-i]
    print('Top {}:'.format(i), labels[label_idx])

print('Elapsed = {:.3f}s'.format(end_time - start_time))