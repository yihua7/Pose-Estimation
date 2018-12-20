from model.hourglass_net import Stacked_Hourglass

data_set_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\'

block_number = 8
layers = 3
lr = 2e-4
out_dim = 256
point_num = 14
maxepoch = 300001
dropout_rate = 0.2
data_set = 'Didi'

if data_set == 'augment':
    image_path = data_set_path + 'augmented_dataset\\images_random_padding\\'
    label_path = data_set_path + 'augmented_dataset\\'
elif data_set == 'Didi':
    image_path = data_set_path + 'video_frame_resize\\'
    label_path = data_set_path + 'video_frame_resize\\'

batch_size = 1

# Create Model
model = Stacked_Hourglass(block_number=block_number, layers=layers, out_dim=out_dim, point_num=point_num, lr=2.5e-4,
                           training=True, dropout_rate=dropout_rate)

cycle = 10000

image_path = data_set_path + 'augmented_dataset\\images_random_padding\\'
label_path = data_set_path + 'augmented_dataset\\'
model.train(image_path, label_path, batch_size, cycle+1, False, 0, step='all', data_set='augment')

image_path = data_set_path + 'video_frame_resize\\'
label_path = data_set_path + 'video_frame_resize\\'
model.train(image_path, label_path, batch_size, 2*cycle+1, False, cycle, step='all', data_set='Didi')
