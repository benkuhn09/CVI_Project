%  part 1: ground truth visualization
%  reads gt.txt and draws bounding boxes on each frame
 
clear all, close all
 
% config
framesPath  = '../frames/View_001/';   
gtFile      = '../data/gt/gt.txt';     
nFrames     = 795;                     % # frames in View001
pauseTime   = 0.02;                    
 
% load gt
fprintf('Loading ground truth...\n');
gt = load(gtFile);
 
% keep only valid detections (confidence == 1)
gt = gt(gt(:,7) == 1, :);
 
fprintf('GT loaded: %d detections, frames %d to %d\n', ...
    size(gt,1), min(gt(:,1)), max(gt(:,1)));
 
% assign unique colour per pedestrian ID
uniqueIDs = unique(gt(:,2));
nIDs      = length(uniqueIDs);
colorMap  = hsv(nIDs);
 

% looping thru frames & drawing bounding boxes
figure('Name','Part 1 - Ground Truth','NumberTitle','off');
 
for f = 1 : nFrames

    frameName = sprintf('%sframe_%04d.jpg', framesPath, f-1);
 
    if ~exist(frameName, 'file')
        continue
    end
 
    img = imread(frameName);
 
    % get all gt detections for this frame
    frameGT = gt(gt(:,1) == f, :);
 
    imshow(img); hold on;
    title(sprintf('Frame %d/%d  |  GT detections: %d', f, nFrames, size(frameGT,1)), ...
        'FontSize', 12);
 
    for k = 1 : size(frameGT, 1)
 
        pedID     = frameGT(k, 2);
        bb_left   = frameGT(k, 3);
        bb_top    = frameGT(k, 4);
        bb_width  = frameGT(k, 5);
        bb_height = frameGT(k, 6);
 
        colorIdx = find(uniqueIDs == pedID);
        c = colorMap(colorIdx, :);
 
        rectangle('Position', [bb_left, bb_top, bb_width, bb_height], ...
                  'EdgeColor', c, 'LineWidth', 2);
 
        text(bb_left, bb_top - 4, sprintf('ID:%d', pedID), ...
             'Color', c, 'FontSize', 9, 'FontWeight', 'bold', ...
             'BackgroundColor', 'k');
    end
 
    drawnow;
    hold off;
    pause(pauseTime);
end
 

 