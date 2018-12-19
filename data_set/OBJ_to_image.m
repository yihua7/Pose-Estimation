image_path = './lsp_dataset/images';
image_list = dir(image_path);
load('./lsp_dataset/joints.mat');
image = [];
for i = 3:size(image_list,1)
    im = imread([image_path, '/',image_list(i).name]);
    [x, y, c]=size(im);
    M = zeros(256,256,3);
    M(1:x,1:y,1:c) = im;
    M = uint8(M);
    imwrite(M, ['./lsp_dataset/images_padding/', num2str(i-2, '%04d'), '.jpg']);
    M = double(M);
%     image = [image; M];
end
% save('./image_data.mat', 'image');

for i = 3:size(image_list,1)
    im = imread([image_path, '/',image_list(i).name]);
    [x, y, c]=size(im);
    M = 255*rand(256,256,3);
    M(1:x,1:y,1:c) = im;
    M = uint8(M);
    imwrite(M, ['./lsp_dataset/images_random_padding_LT/', num2str(i-2, '%04d'), '.jpg']);
    M = double(M);
%     image = [image; M];
end
save('./lsp_dataset/joints_LT.mat','joints_LT');


for i = 3:size(image_list,1)
    im = imread([image_path, '/',image_list(i).name]);
    [x, y, c]=size(im);
    M = 255*rand(256,256,3);
    M(256-x+1:256,1:y,1:c) = im;
    M = uint8(M);
    imwrite(M, ['./lsp_dataset/images_random_padding_LB/', num2str(i-2, '%04d'), '.jpg']);
    M = double(M);
%     image = [image; M];
end
joints_LB=zeros(3,14,2000);
for i = 1:2000
    im = imread([image_path, '/',image_list(i+2).name]);
    [x, y, c]=size(im);
    for j = 1:14
        y0=joints(1,j,i); 
        x0=joints(2,j,i);
        visibility=joints(3,j,i);
        joints_LB(2,j,i) = x0+256-x;
        joints_LB(1,j,i) = y0;
        joints_LB(3,j,i) = visibility;
    end
end
save('./lsp_dataset/joints_LB.mat','joints_LB')

for i = 3:size(image_list,1)
    im = imread([image_path, '/',image_list(i).name]);
    [x, y, c]=size(im);
    M = 255*rand(256,256,3);
    M(1:x,256-y+1:256,1:c) = im;
    M = uint8(M);
    imwrite(M, ['./lsp_dataset/images_random_padding_RT/', num2str(i-2, '%04d'), '.jpg']);
    M = double(M);
%     image = [image; M];
end
joints_RT=zeros(3,14,2000);
for i = 1:2000
    im = imread([image_path, '/',image_list(i+2).name]);
    [x, y, c]=size(im);
    for j = 1:14
        y0=joints(1,j,i); 
        x0=joints(2,j,i);
        visibility=joints(3,j,i);
        joints_RT(2,j,i) = x0;
        joints_RT(1,j,i) = y0+256-y;
        joints_RT(3,j,i) = visibility;
    end
end
save('./lsp_dataset/joints_RT.mat','joints_RT')


for i = 3:size(image_list,1)
    im = imread([image_path, '/',image_list(i).name]);
    [x, y, c]=size(im);
    M = 255*rand(256,256,3);
    M(256-x+1:256,256-y+1:256,1:c) = im;
    M = uint8(M);
    imwrite(M, ['./lsp_dataset/images_random_padding_RB/', num2str(i-2, '%04d'), '.jpg']);
    M = double(M);
%     image = [image; M];
end
joints_RB=zeros(3,14,2000);
for i = 1:2000
    im = imread([image_path, '/',image_list(i+2).name]);
    [x, y, c]=size(im);
    for j = 1:14
        y0=joints(1,j,i); 
        x0=joints(2,j,i);
        visibility=joints(3,j,i);
        joints_RB(2,j,i) = x0+256-x;
        joints_RB(1,j,i) = y0+256-y;
        joints_RB(3,j,i) = visibility;
    end
end
save('./lsp_dataset/joints_RB.mat','joints_RB');

image_path = './lsp_dataset/images_random_padding_LT';
image_list = dir(image_path);
for i = 3:size(image_list,1)
    im = imread([image_path, '/',image_list(i).name]);
    im90 = imrotate(im,-90);
    im180 = imrotate(im,-180);
    im270 = imrotate(im,-270);
    imwrite(im90, ['./lsp_dataset/images_random_padding_ratate_90_clockwise/', num2str(i-2, '%04d'), '.jpg'])
    imwrite(im180, ['./lsp_dataset/images_random_padding_ratate_180_clockwise/', num2str(i-2, '%04d'), '.jpg'])
    imwrite(im270, ['./lsp_dataset/images_random_padding_ratate_270_clockwise/', num2str(i-2, '%04d'), '.jpg'])
end


joints_ratate_90_clockwise=zeros(3,14,2000);
joints_ratate_180_clockwise=zeros(3,14,2000);
joints_ratate_270_clockwise=zeros(3,14,2000);

for i = 1:2000
    for j = 1:14
        x0=joints(2,j,i); 
        y0=joints(1,j,i);
        visibility=joints(3,j,i);
        joints_ratate_90_clockwise(2,j,i) = y0;
        joints_ratate_90_clockwise(1,j,i) = 256-x0;
        joints_ratate_90_clockwise(3,j,i) = visibility;
        joints_ratate_180_clockwise(2,j,i) = joints_ratate_90_clockwise(1,j,i);
        joints_ratate_180_clockwise(1,j,i) = 256-joints_ratate_90_clockwise(2,j,i);
        joints_ratate_180_clockwise(3,j,i) = visibility;
        joints_ratate_270_clockwise(2,j,i) = joints_ratate_180_clockwise(1,j,i);
        joints_ratate_270_clockwise(1,j,i) = 256-joints_ratate_180_clockwise(2,j,i);
        joints_ratate_270_clockwise(3,j,i) = visibility;
    end
end

save('./lsp_dataset/joints_ratate_90_clockwise.mat','joints_ratate_90_clockwise')
save('./lsp_dataset/joints_ratate_180_clockwise.mat','joints_ratate_180_clockwise')
save('./lsp_dataset/joints_ratate_270_clockwise.mat','joints_ratate_270_clockwise')