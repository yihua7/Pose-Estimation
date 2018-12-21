% to video
function tovideo(workdir,fps,suffix)
if nargin == 2
    suffix='png';
end
videoName='x';
% change_filename(workdir,suffix,1);
% workdir=[workdir,'\'];
change_pic_to_video(workdir,videoName, fps, suffix)
