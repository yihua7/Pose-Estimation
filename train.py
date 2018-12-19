from model.hourglass_net import Stacked_Hourglass

block_number = 8
layers = 3
lr = 2e-4
out_dim = 256
point_num = 14
maxepoch = 300001
dropout_rate = 0.2

image_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\augmented_dataset\\images_random_padding'
label_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\augmented_dataset\\'
batch_size = 1

# Create Model
model0 = Stacked_Hourglass(block_number=block_number, layers=layers, out_dim=out_dim, point_num=point_num, lr=2.5e-4,
                          training=True, dropout_rate=dropout_rate)
# model1 = Stacked_Hourglass(block_number=block_number, layers=layers, out_dim=out_dim, point_num=point_num, lr=5e-5,
#                           training=True, dropout_rate=dropout_rate)
# Training

cycle0 = 2000
cycle1 = 1000
cycle = cycle0 + cycle1

model0.train(image_path, label_path, batch_size, cycle0 + 1, False, 0, step=[0], augment=augment)
# model1.train(image_path, label_path, batch_size, cycle1 + 1, True, step=[0], augment=augment)
for i in range(1, block_number):
    model0.train(image_path, label_path, batch_size, (i + 1) * cycle0 + 1, True, base=i*cycle, step=[i], augment=augment)
    # model0.train(image_path, label_path, batch_size, (i + 1) * cycle + 1, True, base=i*cycle+cycle0, step=[i], augment=augment)
    print("%d layers have been trained." %i)

model1.train(image_path, label_path, batch_size, (block_number + 1) * cycle + 1, True, cycle1, step='all', augment=augment)
print("Training End")
