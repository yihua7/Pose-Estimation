from PIL import Image
import numpy as np
import os


def add_color(heatmap, x, y):
    heatmap[x, y, 0] = 143
    heatmap[x, y, 1] = 245
    heatmap[x, y, 2] = 255


def add_line(heatmap, cdi_one, cdi_two, line_width):
    gap_x = int(cdi_one[0]-cdi_two[0])
    gap_y = int(cdi_one[1]-cdi_two[1])
    signx = int(np.sign(gap_x))
    signy = int(np.sign(gap_y))
    # print(gap_x, '\n', gap_y)
    if(gap_x != 0):
        for i in range(0, abs(gap_x)):
            x = int(cdi_two[0] + i * signx)  # np.sign(cdi_two[0]-cdi_one[0])
            y = int(cdi_two[1] + abs(i*gap_y/gap_x)*signy)
                # print(y,' ', cdi_one[1],' ', )
            # print(x,' ',y)
            for j in range(0, line_width):
                yy = max(min(y + j - int(line_width/2), 255), 0)
                #print(type(x))
                #print(type(yy))
                add_color(heatmap, int(x), int(yy))
                # heatmap[x][yy] = 255
    else:
        for i in range(0, abs(gap_y)):
            x = int(cdi_one[0])
            y = int(cdi_two[1] + i*np.sign(gap_y))
                # print(y,' ', cdi_one[1],' ', )
            # print(x,' ',y)
            for j in range(0, line_width):
                yy = max(min(y + j - int(line_width/2), 255), 0)
                #print(type(x))
                #print(type(yy))
                add_color(heatmap, int(x), int(yy))
                # heatmap[x][yy] = 255
    return heatmap


def hotmap_visualization(hotmaps, point_size, filename='image', path='', line_width=3, threshold=-1e-5, raw_image=None):
    "hotmap是64×64×14的array，raw_image是原图256×256×3，point_size是打算让最终可视化的点有多大"
    # 首先要找到hotmap的最大值
    hotmaps = np.squeeze(np.array(hotmaps))
    max_coords = []
    for i in range(0, 14):
        max_value = -1
        max_x = -1
        max_y = -1
        for j in range(0, 63):
            for k in range(0, 63):
                if (hotmaps[j, k, i] > max_value).any():
                    max_value = hotmaps[j, k, i]
                    max_x = j
                    max_y = k
        assert max_y != -1 and max_x != -1
        if max_value >= threshold:
            max_coords.append([max_y * 4, max_x * 4])  # 换了一下顺序。可能python与matlab的index方式不同

    # 把原图的特定位置染色
    if raw_image is None:
        raw_image = np.zeros((256, 256, 3))
    raw_image = np.squeeze(np.array(raw_image))
    # print(max_coords)
    for max_coord in max_coords:
        max_x = max_coord[0]
        max_y = max_coord[1]
        # 找到点的起始x、结束x，起始y、结束y
        start_x = max(max_x - point_size, 0)
        stop_x = min(max_x + point_size, 255)
        start_y = max(max_y - point_size, 0)
        stop_y = min(max_y + point_size, 255)
        # 染色
        for temp_x in range(start_x, stop_x + 1):
            for temp_y in range(start_y, stop_y + 1):
                raw_image[temp_x, temp_y, 0] = 255
                raw_image[temp_x, temp_y, 1] = 0
                raw_image[temp_x, temp_y, 2] = 0

    # 连线
    add_line(raw_image, max_coords[12], max_coords[13], line_width)  # neck & head top
    # print((np.array(max_coords[2]) + np.array(max_coords[3]))/2)
    add_line(raw_image, max_coords[12], (np.array(max_coords[2]) + np.array(max_coords[3]))/2, line_width)  # neck & mid of hip
    add_line(raw_image, max_coords[9], max_coords[10], line_width)  # left shoulder & left elbow
    add_line(raw_image, max_coords[10], max_coords[11], line_width)  # left elbow & left wrist
    add_line(raw_image, max_coords[7], max_coords[8], line_width)  # right shoulder & right elbow
    add_line(raw_image, max_coords[6], max_coords[7], line_width)  # right elbow & right wrist
    add_line(raw_image, max_coords[3], max_coords[4], line_width)  # left hip & left knee
    add_line(raw_image, max_coords[4], max_coords[5], line_width)  # left knee & left ankle
    add_line(raw_image, max_coords[1], max_coords[2], line_width)  # right hip & right knee
    add_line(raw_image, max_coords[0], max_coords[1], line_width)  # right knee & right ankle

    # 输出图片
    img = Image.fromarray(raw_image.astype('uint8')).convert('RGB')
    if not os.path.exists(path):
        os.makedirs(path)
    img.save(path + filename + ".jpg")
    img.show()
