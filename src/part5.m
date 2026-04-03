%  part 5: occupancy heatmap
%
%  (i)  static heatmap  - all frames accumulated at once
%  (ii) dynamic heatmap - accumulates live as video plays
%
%  centroid positions sourced from det.txt

clear all, close all

% config
framesPath = '../frames/View_001/';
detFile = '../data/det/det.txt';
nFrames    = 795;
pauseTime  = 0.03;

% gaussian kernel params, tune as needed
gaussSize  = 51;   % must be odd
gaussSigma = 15;   % larger = broader spread/detection

% load detections
fprintf('Loading detections...\n');
det = load(detFile);

% get image size from first frame
firstFrame = imread(sprintf('%sframe_0000.jpg', framesPath));
[imgH, imgW, ~] = size(firstFrame);

% build gaussian kernel (using fspecial pattern from medianExp_lesson1.m)
gaussKernel = fspecial('gaussian', gaussSize, gaussSigma);


% static heatmap

% accum all centroid hits across all frames then convolve w gaussian
fprintf('Building static heatmap...\n');
accumulator = zeros(imgH, imgW);
for f = 1 : nFrames
    frameDet = det(det(:,1) == f, :);
    for k = 1 : size(frameDet, 1)
        % centroid from bounding box
        cx = round(frameDet(k,3) + frameDet(k,5)/2);
        cy = round(frameDet(k,4) + frameDet(k,6)/2);
        % clamping to image bounds
        cx = max(1, min(imgW, cx));
        cy = max(1, min(imgH, cy));
        accumulator(cy, cx) = accumulator(cy, cx) + 1;
    end
end

% convolve w gaussian to spread each hit
heatmapStatic = filter2(gaussKernel, accumulator);
figure('Name', 'Part 5 - Static Heatmap', 'NumberTitle', 'off');
imshow(firstFrame); hold on;
h = imagesc(heatmapStatic);
set(h, 'AlphaData', mat2gray(heatmapStatic) * 0.7);
colormap hot; colorbar;
title('Static Occupancy Heatmap (all frames)', 'FontSize', 13);
hold off;
saveas(gcf, '../results/part5_static_heatmap.png');
fprintf('Static heatmap done.\n');


% dynamic heatmap

% same accum but updated live each frame
fprintf('Starting dynamic heatmap...\n');
accumDynamic = zeros(imgH, imgW);
figure('Name', 'Part 5 - Dynamic Heatmap', 'NumberTitle', 'off');
for f = 1 : nFrames
    frameName = sprintf('%sframe_%04d.jpg', framesPath, f-1);
    if ~exist(frameName, 'file'), continue, end
    imgfr    = imread(frameName);
    frameDet = det(det(:,1) == f, :);
    % add centroid hits for current frame
    for k = 1 : size(frameDet, 1)
        cx = round(frameDet(k,3) + frameDet(k,5)/2);
        cy = round(frameDet(k,4) + frameDet(k,6)/2);
        cx = max(1, min(imgW, cx));
        cy = max(1, min(imgH, cy));
        accumDynamic(cy, cx) = accumDynamic(cy, cx) + 1;
    end
    % apply gaussian to current accum state
    heatmapDynamic = filter2(gaussKernel, accumDynamic);
    imshow(imgfr); hold on;
    h = imagesc(heatmapDynamic);
    set(h, 'AlphaData', mat2gray(heatmapDynamic) * 0.7);
    colormap hot;
    title(sprintf('Dynamic Heatmap  |  Frame %d/%d', f, nFrames), 'FontSize', 12);
    drawnow;
    hold off;
    pause(pauseTime);
end