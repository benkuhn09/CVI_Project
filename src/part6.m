%  part 6: statistical analysis of trajectories using EM
%
%  fits a gaussian mixture model (GMM) to all pedestrian
%  centroid positions using expectation-maximization (EM).
%  each gaussian component represents a region of high activity.

clear all, close all

% config
framesPath  = '../frames/View_001/';
detFile     = '../data/det/det.txt';
nFrames     = 795;
nComponents = 6;   % # of gaussian components, can tune this
margin      = 100;  % ignores detections this close to image border

% load detections
fprintf('Loading detections...\n');
det = load(detFile);

% get image size from first frame
firstFrame = imread(sprintf('%sframe_0000.jpg', framesPath));
[imgH, imgW, ~] = size(firstFrame);

% collect all centroids across all frames
allCentroids = [];
for f = 1 : nFrames
    frameDet = det(det(:,1) == f, :);
    for k = 1 : size(frameDet, 1)
        cx = frameDet(k,3) + frameDet(k,5)/2;
        cy = frameDet(k,4) + frameDet(k,6)/2;
        allCentroids = [allCentroids; cx, cy]; %#ok<AGROW>
    end
end

% remove detections near image borders (bc false positives)
allCentroids = allCentroids( ...
    allCentroids(:,1) > margin & allCentroids(:,1) < imgW - margin & ...
    allCentroids(:,2) > margin & allCentroids(:,2) < imgH - margin, :);

fprintf('Centroid positions after border filter: %d\n', size(allCentroids,1));

% fit GMM using EM
fprintf('Fitting GMM with %d components...\n', nComponents);
gmm = fitgmdist(allCentroids, nComponents, ...
    'CovarianceType', 'full', ...
    'RegularizationValue', 1, ...
    'Replicates', 5, ...
    'Options', statset('MaxIter', 500));
fprintf('GMM fitted.\n');

diary('../results/part6_gmm_stats.txt');
fprintf('\n--- GMM Statistical Summary ---\n');
for c = 1 : nComponents
    mu    = gmm.mu(c,:);
    sigma = squeeze(gmm.Sigma(:,:,c));
    w     = gmm.ComponentProportion(c);
    fprintf('Component %d:\n', c);
    fprintf('  Weight (mixing proportion): %.3f\n', w);
    fprintf('  Mean position: x=%.1f, y=%.1f\n', mu(1), mu(2));
    fprintf('  Std dev: sx=%.1f, sy=%.1f\n', sqrt(sigma(1,1)), sqrt(sigma(2,2)));
end
fprintf('--------------------------------\n\n');
diary off;

% plot 1: scatter of all centroids + gmm ellipses
figure('Name', 'Part 6 - GMM Components', 'NumberTitle', 'off');
imshow(firstFrame); hold on;
title(sprintf('EM/GMM - %d Gaussian Components', nComponents), 'FontSize', 13);

colors = lines(nComponents);

% scatter all centroid points
plot(allCentroids(:,1), allCentroids(:,2), '.', ...
    'Color', [0.7 0.7 0.7], 'MarkerSize', 2);

% draw each gaussian component as an ellipse
for c = 1 : nComponents
    mu    = gmm.mu(c,:);
    sigma = squeeze(gmm.Sigma(:,:,c));
    drawGaussianEllipse(mu, sigma, colors(c,:));
    plot(mu(1), mu(2), '+', 'Color', colors(c,:), ...
        'MarkerSize', 12, 'LineWidth', 2);
    text(mu(1)+5, mu(2)-5, sprintf('C%d', c), ...
        'Color', colors(c,:), 'FontSize', 9, 'FontWeight', 'bold');
end
hold off;
saveas(gcf, '../results/part6_gmm_ellipses.png');

% plot 2: density map from gmm
fprintf('Computing density map...\n');

[xGrid, yGrid] = meshgrid(1:imgW, 1:imgH);
gridPoints     = [xGrid(:), yGrid(:)];
pdfVals        = pdf(gmm, gridPoints);
densityMap     = reshape(pdfVals, imgH, imgW);

% boosting contrast to show hotspots better
densityMap = densityMap .^ 0.5;
densityMap = densityMap / max(densityMap(:));

figure('Name', 'Part 6 - GMM Density Map', 'NumberTitle', 'off');
imshow(firstFrame); hold on;
h = imagesc(densityMap);
set(h, 'AlphaData', densityMap * 0.8);
colormap hot; colorbar;
title('GMM Density Map (EM)', 'FontSize', 13);
hold off;
saveas(gcf, '../results/part6_gmm_density.png');

fprintf('Done! Part 6 complete.\n');

% draws 2-sigma ellipse for a gaussian component
function drawGaussianEllipse(mu, sigma, color)
    [V, D]  = eig(sigma);
    angles  = linspace(0, 2*pi, 100);
    unit    = [cos(angles); sin(angles)];
    ellipse = V * (2 * sqrt(D)) * unit;
    plot(mu(1) + ellipse(1,:), mu(2) + ellipse(2,:), ...
        '-', 'Color', color, 'LineWidth', 2);
end