image = zeros(256,256,3);
label = zeros(64,64,14);

[x, y]=meshgrid(linspace(1,64,64)',linspace(1,64,64)');
X=[x(:) y(:)];

A = load('D:\CS\机器学习大作业\Pose-Detection\data_set\augmented_dataset\joints_LT.mat');
joints = A.joints;
for i = 3:2002
%     im = imread([image_path, '/',image_list(i).name]);
%     [x, y, c]=size(im);
%     image(1:x,1:y,1:c) = im;
    
    for j = 1:14
        if joints(3,j,i-2) == 0
            a = joints(1,j,i-2)/4;
            b = joints(2,j,i-2)/4;
            z = mvnpdf(X,[a,b],eye(2));
            label(:,:,j) = reshape(z,64,64)';
        else
            label(:,:,j) = zeros(64,64);
        end
    end
    label = label*2*pi;
    save(['D:\CS\机器学习大作业\Pose-Detection\data_set\augmented_dataset\heatmap_LT\im',num2str(i-2,'%04d'),'.mat'],'image','label');
end