%  part 2: our detector
%  
%  step 1: estimate bkgd from median (cBkg_1.m)
%  step 2: subtract bkgd, threshold, morphological cleanup 
%          (Object-detection.m, Brain_tissue_detection.m)
%  step 3: bwlabel & regionprops & bounding boxes (Coins_Detection.m, 
%          Object-detection.m)
%  step 4: generate ground truth detections file
 
clear all, close all
 
% config
framesPath  = '../frames/View_001/';
nFrames     = 795;
pauseTime   = 0.02;
 
% detect params, we can tune 
thr         = 30;      
minArea     = 400;     
seSize      = 3;     
 
% bkgd est params
nBkgFrames  = 100;     % # frames used for bkgd est
bkgStep     = 5;       % sample step size when building bkgd
 
% ground truth output file
detOutputFile = '../data/our_detections.txt';

% step 1
fprintf('Estimating background from %d frames...\n', nBkgFrames);
 
i = 1;
for k = 1 : bkgStep : (nBkgFrames * bkgStep)
    frameName = sprintf('%sframe_%04d.jpg', framesPath, k-1);
    if ~exist(frameName, 'file'), continue, end
    vid4D(:,:,:,i) = imread(frameName);
    i = i + 1;
end
 
% moving peds shouldn't persist over the median
bkg = median(double(vid4D), 4);
 
fprintf('Background estimated.\n');
 
% steps 2 & 3
se = strel('disk', seSize); 
 
% Initialize ground truth storage
ourDetections = []; % format: [frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z]

figure('Name', 'Part 2 - Detector Output', 'NumberTitle', 'off');
 
for f = 1 : nFrames
 
    frameName = sprintf('%sframe_%04d.jpg', framesPath, f-1);
    if ~exist(frameName, 'file'), continue, end
 
    imgfr = imread(frameName);
 
    % bkgd subtraction
    imgdif = (abs(double(imgfr(:,:,1)) - bkg(:,:,1)) > thr) | ...
             (abs(double(imgfr(:,:,2)) - bkg(:,:,2)) > thr) | ...
             (abs(double(imgfr(:,:,3)) - bkg(:,:,3)) > thr);
 
    bw = imclose(imgdif, se);
    bw = imerode(bw, se);
 
    % label connectiosn
    [lb, num] = bwlabel(bw);
    regionProps = regionprops(lb, 'Area', 'Centroid', 'BoundingBox');
 
    % use minArea to filter noise
    inds = find([regionProps.Area] > minArea);
    nDet = length(inds);
 
    imshow(imgfr); hold on;
    title(sprintf('Frame %d/%d  |  Detections: %d', f, nFrames, nDet), ...
        'FontSize', 12);
 
    for j = 1 : nDet
        % getting bb
        bb = regionProps(inds(j)).BoundingBox;
        cx = regionProps(inds(j)).Centroid(1);
        cy = regionProps(inds(j)).Centroid(2);
 
        % Store detection in ground truth format: [frame, id, left, top, width, height, conf, x, y, z]
        ourDetections = [ourDetections; f, j, bb(1), bb(2), bb(3), bb(4), 1.0, cx, cy, 0];

        % drawing bb
        rectangle('Position', bb, 'EdgeColor', [1 1 0], 'LineWidth', 2);
 
        % label bb
        text(bb(1), bb(2) - 4, sprintf('Det:%d', j), ...
             'Color', [1 1 0], 'FontSize', 9, 'FontWeight', 'bold', ...
             'BackgroundColor', 'k');
 
        % mark centroid
        %plot(cx, cy, 'g.', 'MarkerSize', 12);
    end
 
    drawnow;
    hold off;
    %pause(pauseTime);
end

% Save our detections to file
fprintf('\nSaving our detector results to: %s\n', detOutputFile);
dlmwrite(detOutputFile, ourDetections, 'delimiter', ',');
fprintf('Saved %d total detections across %d frames\n', size(ourDetections, 1), nFrames);

fprintf('\nPart 2 complete. Ground truth generated for evaluation in Part 7.\n');