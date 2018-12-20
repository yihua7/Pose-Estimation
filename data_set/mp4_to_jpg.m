clc
clear
%%将视频转换成帧图片
name = 'sb_taiji';
filename = [name, '.mp4'];
%% 读取视频
video_path=['./video/', filename];    
video_obj=VideoReader(video_path);   

frame_number=video_obj.NumberOfFrames;

%% 存储每一帧图片到文件夹image
if ~exist(['./video_frame_resize/', name, '/ori'],'dir')
    mkdir(['./video_frame_resize/', name, '/ori']);
    disp('successfully create directory!');
end

for i=1:frame_number
    image_name=['.\video_frame_resize/' name '/ori/' sprintf('%04d', i) '.jpg'];
    frame=read(video_obj,i);
    
    frame = permute(frame, [2,1,3]);
    
    fsize = size(frame);
    height = fsize(1);
    width = fsize(2);
    
    rateh = height/256;
    ratew = width/256;
    
    scale = max(rateh, ratew);
 
    newframe = zeros([256, 256, 3]);
    frame = double(frame);
    
    for c=1:3
        for x=1:int32(height/scale)
            for y=1:int32(width/scale)
                sum = 0;
                count = 0;
                for xx=int32(scale*x-scale+1):int32(scale*x)-1
                   for yy=int32(scale*y-scale+1):int32(scale*y)-1
                      sum = sum + frame(xx, yy, c);
                      count = count + 1;
                   end
                end
                newframe(x, y, c) = sum/count;
            end
        end
    end

    newframe = uint8(newframe);
    
    imwrite(newframe,image_name, 'jpg');
end

disp('all images are written into directory image')
