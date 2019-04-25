function plotMatchingFeatures(im1, im2, f1, f2, n, doRandomColor)
%PLOTMATCHES Concatenates images and plots matching features
% IN
%  im1 : first input image (plotted on the left)
%  im2 : second input image (plotted on the right)
%  f1 : features in first input image
%  f2 : features in second input image
%  n : amount of features to plot
%  doRandomColor : if true, use random colors for lines between matches

% -------  Giving unspecified vars values ----------
if nargin < 5
    n = -1;
end
if nargin < 6
    doRandomColor = false;
end

%  -------- select random features ---------------
if n >= 0
    perm = randperm(size(f1, 2));
    sel = perm(1:min(n, length(perm)));
    f1 = f1(:, sel);
    f2 = f2(:, sel);
end

[h1, xoff, ~] = size(im1);
h2 = size(im2, 1);

hdiff = abs(h1 - h2);
yoff = fix(hdiff / 2);

% ----------- padd image and adjust matching coordinates -------
if h1 < h2
    im1 = padarray(im1, yoff, 'pre');
    im1 = padarray(im1, yoff + mod(hdiff, 2), 'post');
    f1(2, :) = f1(2, :) + yoff;
else
    im2 = padarray(im2, yoff, 'pre');
    im2 = padarray(im2, yoff + mod(hdiff, 2), 'post');
    f2(2, :) = f2(2, :) + yoff;
end
f2(1, :) = f2(1, :) + xoff;

x1 = f1(1, :);
y1 = f1(2, :);
x2 = f2(1, :);
y2 = f2(2, :);

%  -------- plot concatenated images -----------
canvas = horzcat(im1, im2);
imshow(canvas);

hold on

% -------- plot lines between matching features -----------
lineWidth = 2.5;
if doRandomColor
    for ii = 1:length(x1)
        line([x1(ii); x2(ii)], [y1(ii); y2(ii)], 'Color', rand(1, 3), ...
            'LineWidth', lineWidth);
    end
else
    colorOrder = get(gca, 'ColorOrder');
    line([x1; x2], [y1; y2], 'Color', colorOrder(1, :), ...
        'LineWidth', lineWidth);
end

% plot features
h1 = vl_plotframe([f1 f2]);
h2 = vl_plotframe([f1 f2]);
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

hold off

end