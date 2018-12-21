from model.hourglass_net import Stacked_Hourglass

block_number = 8
layers = 3
lr = 0.0001
out_dim = 256
point_num = 14
maxepoch = 10001

image_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\video_frame_resize\\SB\\ori\\'

model = Stacked_Hourglass(block_number=block_number, layers=layers, out_dim=out_dim, point_num=point_num, lr=lr, training=False)
# model.test_label()
image_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\video_frame_resize\\SB\\ori\\'
model.use(image_path, 'SB')

image_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\video_frame_resize\\taiji\\ori\\'
model.use(image_path, 'taiji')

image_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\video_frame_resize\\sb_taiji\\ori\\'
model.use(image_path, 'sb_taiji')
