from load_data.load_data import load_data
from model.hourglass_net import Stacked_Hourglass

block_number = 8
layers = 3
lr = 0.01
out_dim = 256
point_num = 14
maxepoch = 10000

image, label = load_data()

model = Stacked_Hourglass(block_number=block_number, layers=layers, out_dim=out_dim, point_num=point_num, lr=lr)
model.train(image, label, maxepoch)
