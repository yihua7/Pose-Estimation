clc
clear
%%将视频转换成帧图片
name = 'shake';
filename = [name, '.mp4'];
%% 读取视频
video_path=['./video/', filename];    
video_obj=VideoReader(video_path);   

frame_number=video_obj.NumberOfFrames;

%% 存储每一帧图片到文件夹image
if ~exist(['./video_frame_resize/', name],'dir')
    mkdir(['./video_frame_resize/', name]);
    disp('successfully create directory!');
end

for i=1:frame_number
    image_name=['.\video_frame_resize/' name '/' sprintf('%04d', i) '.jpg'];
    frame=read(video_obj,i);
    
    fsize = size(frame);
    height = fsize(1);
    width = fsize(2);
    
    rateh = height/256;
    ratew = width/256;
 
    newframe = zeros([256, 256, 3]);
    frame = double(frame);
    
    for c=1:3
        for x=1:int8(height/4)
            for y=1:int8(width/4)
                sum = 0;
                count = 0;
                for xx=4*x-3:4*x
                   for yy=4*y-3:4*y
                      sum = sum + frame(xx, yy, c);
                      count = count + 1;
                   end
                end
                newframe(x, y, c) = sum/count;
            end
        end
    end
    newframe = uint8(newframe);
    
%     for c=1:3
%         for x=1:256
%             for y=1:256
%                 pixsum = 0;
%                 count = 0;
%                 for xx=int32(x*rateh-rateh+1):int32(x*rateh)
%                     for yy=int32(y*ratew-ratew+1):int32(y*ratew)
%                         pixsum = pixsum + double(frame(xx, yy, c));
%                         count = count + 1;
%                     end
%                 end
%                 newframe(x, y, c) = uint8(pixsum/count);
%             end
%         end
%     end
    
    imwrite(newframe,image_name, 'jpg');
end

disp('all images are written into directory image')
