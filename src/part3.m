%  part 3: plot performed trajectories dynamically
%
%  Shows tracked trajectories over time using the detector from part 2.

clear all, close all

% config
framesPath  = '../frames/View_001/';
nFrames     = 795;
pauseTime   = 0.02;

% detect params
thr         = 30;
minArea     = 400;
seSize      = 3;

% bkgd est params
nBkgFrames  = 100;
bkgStep     = 5;

% tracking params
maxMatchDist    = 45;
maxInvisible    = 10;
maxTraceLength  = 60;
trackPalette    = lines(128);
tracks          = emptyTracks();
nextTrackID     = 1;

seClose = strel('disk', seSize);
seErode = strel('disk', seSize);

fprintf('Estimating background from %d frames...\n', nBkgFrames);

i = 1;
for k = 1 : bkgStep : (nBkgFrames * bkgStep)
    frameName = sprintf('%sframe_%04d.jpg', framesPath, k - 1);
    if ~exist(frameName, 'file'), continue, end
    vid4D(:,:,:,i) = imread(frameName); %#ok<AGROW>
    i = i + 1;
end

bkg = median(double(vid4D), 4);
fprintf('Background estimated.\n');

figure('Name', 'Part 3 - Dynamic Trajectories', 'NumberTitle', 'off');

for f = 1 : nFrames
    frameName = sprintf('%sframe_%04d.jpg', framesPath, f - 1);
    if ~exist(frameName, 'file'), continue, end

    imgfr = imread(frameName);

    imgdif = (abs(double(imgfr(:,:,1)) - bkg(:,:,1)) > thr) | ...
             (abs(double(imgfr(:,:,2)) - bkg(:,:,2)) > thr) | ...
             (abs(double(imgfr(:,:,3)) - bkg(:,:,3)) > thr);

    bw = imclose(imgdif, seClose);
    bw = imerode(bw, seErode);

    [lb, ~] = bwlabel(bw);
    regionProps = regionprops(lb, 'Area', 'Centroid', 'BoundingBox');

    inds = find([regionProps.Area] > minArea);
    [detBBoxes, detCentroids] = buildDetections(regionProps, inds);
    nDet = size(detBBoxes, 1);
    detAppearances = zeros(nDet, 3);

    for j = 1 : nDet
        detAppearances(j, :) = computeAppearance(imgfr, detBBoxes(j, :));
    end

    [tracks, nextTrackID] = updateTracks( ...
        tracks, detBBoxes, detCentroids, detAppearances, ...
        nextTrackID, maxMatchDist, maxInvisible, maxTraceLength, ...
        trackPalette);

    if isempty(tracks)
        nActiveTracks = 0;
    else
        nActiveTracks = sum([tracks.invisibleCount] == 0);
    end

    imshow(imgfr); hold on;
    title(sprintf(['Frame %d/%d  |  Detections: %d  |  ' ...
        'Trajectories: %d'], f, nFrames, nDet, nActiveTracks), ...
        'FontSize', 12);

    for t = 1 : numel(tracks)
        history = tracks(t).history;

        if size(history, 1) > 1
            plot(history(:,1), history(:,2), '-', ...
                'Color', tracks(t).color, 'LineWidth', 1.5);
        end

        if tracks(t).invisibleCount == 0
            bb = tracks(t).bbox;
            cx = tracks(t).centroid(1);
            cy = tracks(t).centroid(2);

            rectangle('Position', bb, 'EdgeColor', tracks(t).color, ...
                'LineWidth', 2);
            plot(cx, cy, '.', 'Color', tracks(t).color, 'MarkerSize', 16);
        end
    end

    drawnow;
    hold off;
    pause(pauseTime);
end

function tracks = emptyTracks()

tracks = struct('id', {}, 'bbox', {}, 'centroid', {}, 'history', {}, ...
    'age', {}, 'visibleCount', {}, 'invisibleCount', {}, ...
    'appearance', {}, 'color', {});

end

function [tracks, nextTrackID] = updateTracks(tracks, detBBoxes, ...
    detCentroids, detAppearances, nextTrackID, maxMatchDist, ...
    maxInvisible, maxTraceLength, trackPalette)

nTracks = numel(tracks);
nDet = size(detCentroids, 1);

assignments = zeros(0, 2);
unmatchedTracks = 1:nTracks;
unmatchedDetections = 1:nDet;

if nTracks > 0 && nDet > 0
    [assignments, unmatchedTracks, unmatchedDetections] = ...
        assignDetections(tracks, detCentroids, maxMatchDist);
end

for k = 1 : size(assignments, 1)
    trackIdx = assignments(k, 1);
    detIdx = assignments(k, 2);
    newCentroid = detCentroids(detIdx, :);

    tracks(trackIdx).bbox = detBBoxes(detIdx, :);
    tracks(trackIdx).centroid = newCentroid;
    tracks(trackIdx).history = appendHistory( ...
        tracks(trackIdx).history, newCentroid, maxTraceLength);
    tracks(trackIdx).age = tracks(trackIdx).age + 1;
    tracks(trackIdx).visibleCount = tracks(trackIdx).visibleCount + 1;
    tracks(trackIdx).invisibleCount = 0;
    tracks(trackIdx).appearance = detAppearances(detIdx, :);
end

for k = unmatchedTracks
    tracks(k).age = tracks(k).age + 1;
    tracks(k).invisibleCount = tracks(k).invisibleCount + 1;
end

for detIdx = unmatchedDetections
    tracks(end + 1) = createTrack( ...
        nextTrackID, detBBoxes(detIdx, :), detCentroids(detIdx, :), ...
        detAppearances(detIdx, :), trackPalette); %#ok<AGROW>
    nextTrackID = nextTrackID + 1;
end

if ~isempty(tracks)
    keepMask = [tracks.invisibleCount] <= maxInvisible;
    tracks = tracks(keepMask);
end

end

function [assignments, unmatchedTracks, unmatchedDetections] = ...
    assignDetections(tracks, detCentroids, maxMatchDist)

nTracks = numel(tracks);
nDet = size(detCentroids, 1);
costMatrix = inf(nTracks, nDet);

for t = 1 : nTracks
    for d = 1 : nDet
        distToTrack = norm(tracks(t).centroid - detCentroids(d, :));

        if distToTrack <= maxMatchDist
            costMatrix(t, d) = distToTrack;
        end
    end
end

assignments = zeros(0, 2);
usedTracks = false(1, nTracks);
usedDetections = false(1, nDet);

while true
    [minCost, linearIdx] = min(costMatrix(:));

    if isempty(minCost) || isinf(minCost)
        break
    end

    [trackIdx, detIdx] = ind2sub(size(costMatrix), linearIdx);
    assignments(end + 1, :) = [trackIdx, detIdx]; %#ok<AGROW>
    usedTracks(trackIdx) = true;
    usedDetections(detIdx) = true;
    costMatrix(trackIdx, :) = inf;
    costMatrix(:, detIdx) = inf;
end

unmatchedTracks = find(~usedTracks);
unmatchedDetections = find(~usedDetections);

end

function track = createTrack(trackID, bbox, centroid, appearance, trackPalette)

colorIdx = mod(trackID - 1, size(trackPalette, 1)) + 1;

track.id = trackID;
track.bbox = bbox;
track.centroid = centroid;
track.history = centroid;
track.age = 1;
track.visibleCount = 1;
track.invisibleCount = 0;
track.appearance = appearance;
track.color = trackPalette(colorIdx, :);

end

function history = appendHistory(history, centroid, maxTraceLength)

history = [history; centroid];

if size(history, 1) > maxTraceLength
    history = history(end - maxTraceLength + 1:end, :);
end

end

function [detBBoxes, detCentroids] = buildDetections(regionProps, inds)

detBBoxes = zeros(0, 4);
detCentroids = zeros(0, 2);

for j = 1 : numel(inds)
    props = regionProps(inds(j));
    detBBoxes(end + 1, :) = props.BoundingBox; %#ok<AGROW>
    detCentroids(end + 1, :) = props.Centroid; %#ok<AGROW>
end

end

function appearance = computeAppearance(imgfr, bbox)

[imgH, imgW, ~] = size(imgfr);
x1 = max(1, floor(bbox(1)));
y1 = max(1, floor(bbox(2)));
x2 = min(imgW, ceil(bbox(1) + bbox(3) - 1));
y2 = min(imgH, ceil(bbox(2) + bbox(4) - 1));

if x2 < x1 || y2 < y1
    appearance = [0 0 0];
    return
end

patch = double(imgfr(y1:y2, x1:x2, :)) / 255;
appearance = squeeze(mean(mean(patch, 1), 2))';

if isempty(appearance) || any(~isfinite(appearance))
    appearance = [0 0 0];
end

end
