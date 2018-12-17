image_path = './lsp_dataset/images';
image_list = dir(image_path);

image_data
for i = 3:size(image_list,1)
    im = imread([image_path, '/',image_list(i).name]);
    [x, y, c]=size(im);
    M = zeros(256,256,3);
    M(1:x,1:y,1:c) = im;
    M = uint8(M);
    imwrite(M, ['./lsp_dataset/images_padding/', num2str(i-2, '%04d'), '.jpg']);
    M = double(M);
    image = [image; M];
end
save('./image_data.mat', 'image', '-v.7');