function [] = change_pic_to_video(pathName,videoName, fps, format)
% %  change_pic_to_video  
pathName=[pathName,'\'];
%sort the picture name
fileName = dir([pathName, '*.', format]);
count = size(fileName, 1);

[~,order]=sort_nat({fileName.name});
fileName=fileName(order);
%make picture to video
videoRoute = [pathName, videoName,'.mp4'];
videoObj = VideoWriter(videoRoute,'MPEG-4');
videoObj.FrameRate = fps;
open(videoObj);
for i = 1:count
    img = imread([pathName, fileName(i).name]);  %num2str(i-1)
    writeVideo(videoObj, (img));
end
close(videoObj);

% %framesPath :图像序列所在路径，同时要保证图像大小相同
% %videoName:  表示将要创建的视频文件的名字
% %quality:    生成视频的质量 0-100
% %Compressed: 压缩类型， 'Indeo3'（默认）, 'Indeo5', 'Cinepak', 'MSVC', 'RLE' or 'None'
% %fps: 帧率
% %startFrame ,endFrame ;表示从哪一帧开始，哪一帧结束
% 
% if(exist('videoName','file'))
%     delete videoName.avi
% end
% 
% %生成视频的参数设定
% aviobj=VideoWriter(videoName);  %创建一个avi视频文件对象，开始时其为空
% aviobj.Fps=fps;
% 
% %读入图片
% for i=startFrame:endFrame
%     fileName=sprintf('%08d',i);    %根据文件名而定 我这里文件名是00000001.jpg 00000002.jpg ....
%     frames=imread([framesPath,fileName,'.jpg']);
%     aviobj=writeVideo(aviobj,uint8(frames));
% end
% close(aviobj); % 关闭创建视频
end


% videoRoute = [pathName, videoName];
% videoObj = VideoWriter(videoRoute);
% videoObj.FrameRate = fps;
% picName = dir([pathName, '*.png']);
% count = size(picName, 1);
% open(videoObj);
% for i = 1:5451
%     img = imread([pathName, num2str(i-1), '.png']);
%     writeVideo(videoObj, im2frame(img));
% end
% close(videoObj);
