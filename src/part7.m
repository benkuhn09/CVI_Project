%% Part 7: Evaluation of our detector against ground truth
clear all; close all; clc;

%% Configuration
gtFile      = '../data/gt/gt.txt';           % manual ground truth
ourDetFile  = '../data/our_detections.txt';  % our detector output from Part 2
nFrames     = 795;


% Read files: [frame, id, left, top, width, height, conf, x, y, z]
gtRaw = dlmread(gtFile);
detRaw = dlmread(ourDetFile);

gtByFrame = cell(nFrames, 1);
detByFrame = cell(nFrames, 1);

for i = 1:size(gtRaw, 1)
    f = gtRaw(i, 1);
    bb = gtRaw(i, 3:6); % [left, top, width, height]
    gtByFrame{f} = [gtByFrame{f}; bb];
end

for i = 1:size(detRaw, 1)
    f = detRaw(i, 1);
    bb = detRaw(i, 3:6);
    detByFrame{f} = [detByFrame{f}; bb];
end

% Function to compute IoU between two boxes
iou = @(boxA, boxB) rectint(boxA, boxB) / (boxA(3)*boxA(4) + boxB(3)*boxB(4) - rectint(boxA, boxB));

%% Per-frame matching and accumulation of metrics
totalGT = 0;  % total number of ground truth objects over all frames
totalDet = 0; % total number of detections over all frames
totalTP = 0;  % true positives (matched with IoU >= 0.5)
totalFP = 0;  % false positives (detections not matched)
totalFN = 0;  % false negatives (GT not matched)

% For success plot: store the maximum IoU achieved for each GT object
frameMaxIoU = zeros(795, 1);

% For visualisation: collect frames with both FP and FN
framesWithIssues = [];

% Matching IoU threshold (used to define a valid match)
matchThresh = 0.5;

for f = 1:nFrames
    gtBoxes = gtByFrame{f};
    detBoxes = detByFrame{f};

    nGT = size(gtBoxes, 1);
    nDet = size(detBoxes, 1);

    totalGT = totalGT + nGT;
    totalDet = totalDet + nDet;

    if nGT == 0 && nDet == 0
        continue;
    end

    % Compute IoU matrix (nGT x nDet)
    iouMat = zeros(nGT, nDet);
    for i = 1:nGT
        for j = 1:nDet
            iouMat(i,j) = iou(gtBoxes(i,:), detBoxes(j,:));
        end
    end

    % Greedy matching: repeatedly take the highest IoU above threshold
    matchedGT = false(nGT, 1);
    matchedDet = false(nDet, 1);

    % Record max IoU for each GT (needed for success plot)
    frameMaxIoU(f) = max(iouMat(:));

    % Perform matching using a sorted list of candidate pairs
    [sortedIoU, idx] = sort(iouMat(:), 'descend');
    for k = 1:length(sortedIoU)
        if sortedIoU(k) < matchThresh
            break;
        end
        [gtIdx, detIdx] = ind2sub([nGT, nDet], idx(k));
        if ~matchedGT(gtIdx) && ~matchedDet(detIdx)
            matchedGT(gtIdx) = true;
            matchedDet(detIdx) = true;
            totalTP = totalTP + 1;
        end
    end

    % False positives = unmatched detections
    fpThisFrame = sum(~matchedDet);
    totalFP = totalFP + fpThisFrame;

    % False negatives = unmatched GT
    fnThisFrame = sum(~matchedGT);
    totalFN = totalFN + fnThisFrame;

    % Remember frames that contain both FP and FN for visualisation
    if fpThisFrame > 0 && fnThisFrame > 0
        framesWithIssues = [framesWithIssues, f];
    end
end

%% Compute percentages and display
fnPercent = (totalFN / totalGT) * 100;
fpPercent = (totalFP / totalDet) * 100;

fprintf('=== Evaluation Results (match IoU >= %.2f) ===\n', matchThresh);
fprintf('Total ground truth objects : %d\n', totalGT);
fprintf('Total detections           : %d\n', totalDet);
fprintf('True Positives             : %d\n', totalTP);
fprintf('False Negatives (missed)   : %d (%.2f%%)\n', totalFN, fnPercent);
fprintf('False Positives (spurious) : %d (%.2f%%)\n', totalFP, fpPercent);

%% Success plot (percentage of GT objects with max IoU >= threshold)
binEdges = 0:0.05:1;
binCenters = binEdges(1:end-1) + 0.05;
binCounts = histcounts(frameMaxIoU, binEdges);

% Create bar chart
figure('Name', 'Success Plot - Frames by Max IoU', 'NumberTitle', 'off');
bar(binCenters, binCounts, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'k');
xlabel('Maximum IoU in Frame');
ylabel('Number of Frames');
title('Success Plot: Distribution of Best Detection Overlap per Frame');
grid on;
xlim([0 1]);
xticks(0:0.1:1);

% Add value labels on bars
for i = 1:length(binCounts)
    text(binCenters(i), binCounts(i) + 1, num2str(binCounts(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

%% Visualise example frames with false positives and false negatives
framesPath = '../frames/View_001/';
if ~isempty(framesWithIssues)
    % Pick up to 3 example frames
    nExamples = min(3, length(framesWithIssues));

    rng(42); % for reproducibility, remove this line for true randomness
    randomIndices = randperm(length(framesWithIssues), nExamples);
    exampleFrames = framesWithIssues(randomIndices);

    for ex = 1:nExamples
        f = exampleFrames(ex);
        img = imread(sprintf('%sframe_%04d.jpg', framesPath, f-1));
        gtBoxes = gtByFrame{f};
        detBoxes = detByFrame{f};

        % Recompute matching for this frame to know which are TP/FP/FN
        nGT = size(gtBoxes, 1);
        nDet = size(detBoxes, 1);
        if nGT == 0 && nDet == 0, continue; end

        iouMat = zeros(nGT, nDet);
        for i = 1:nGT
            for j = 1:nDet
                iouMat(i,j) = iou(gtBoxes(i,:), detBoxes(j,:));
            end
        end

        matchedGT = false(nGT, 1);
        matchedDet = false(nDet, 1);
        [sortedIoU, idx] = sort(iouMat(:), 'descend');
        for k = 1:length(sortedIoU)
            if sortedIoU(k) < matchThresh, break; end
            [gtIdx, detIdx] = ind2sub([nGT, nDet], idx(k));
            if ~matchedGT(gtIdx) && ~matchedDet(detIdx)
                matchedGT(gtIdx) = true;
                matchedDet(detIdx) = true;
            end
        end

        % Draw the frame
        figure('Name', sprintf('Example Frame %d (FP+FN)', f), 'NumberTitle', 'off');
        imshow(img); hold on;

        % Draw ground truth: green for matched, red for FN (unmatched)
        for i = 1:nGT
            bb = gtBoxes(i,:);
            if matchedGT(i)
                color = [0 1 0];  % green = true positive
                label = 'TP';
            else
                color = [1 0 0];  % red = false negative
                label = 'FN';
            end
            rectangle('Position', bb, 'EdgeColor', color, 'LineWidth', 2);
            text(bb(1), bb(2)-5, label, 'Color', color, 'FontSize', 8, ...
                'FontWeight', 'bold', 'BackgroundColor', 'k');
        end

        % Draw detections: cyan for matched (already drawn as TP), magenta for FP
        for j = 1:nDet
            if ~matchedDet(j)
                bb = detBoxes(j,:);
                rectangle('Position', bb, 'EdgeColor', [1 0 1], 'LineWidth', 2); % magenta
                text(bb(1), bb(2)-5, 'FP', 'Color', [1 0 1], 'FontSize', 8, ...
                    'FontWeight', 'bold', 'BackgroundColor', 'k');
            end
        end

        title(sprintf('Frame %d: Green=TP, Red=FN (missed), Magenta=FP (spurious)', f));
        hold off;
    end
else
    disp('No frame contains both false positives and false negatives.');
end