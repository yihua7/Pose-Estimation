from load_data.load_data import load_data
from model.hourglass_net import hourglass

block_number = 7
layers = 5
w_dim = [3, 3, 3, 3, 3]
lr = 0.01
maxepoch = 10000

image, label = load_data()

model = hourglass(block_number, layers, w_dim, lr)
model.train(image, label, maxepoch)
