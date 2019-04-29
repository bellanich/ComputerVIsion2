% answers Image Alignment - Question 1.1 and 1.2
function [ matches, fa, fb ] = keypoint_matching (Ia, Ib, thresh, show, presel)
if nargin == 2
    thresh = 1.5;
    show = true;
    presel = [];
elseif nargin == 3
    show = false;
    presel = [];
end
% matching keypoints from different images

% vl_sift identifies keypoints of interest
[fa, da] = vl_sift(Ia) ;
[fb, db] = vl_sift(Ib) ;
% vl_ubcmatch matches
[matches, scores] = vl_ubcmatch(da, db, thresh) ;

% note: we may have to work with Threshold:
%VL_UBCMATCH(DESCR1, DESCR2, THRESH)

% we try to visualise some matches. 
if isempty(presel) 
    p = 10;
    perm = randperm(size(matches, 2));
    sel = perm(1:p);
else
    sel = presel;
end
m1 = matches(1, sel);   
m2 = matches(2, sel);
score = scores(:, sel);

if show == true
    figure ('Name', 'Keypoint Matching: Some matched Keypoints')
    vl_tightsubplot(1,2,1);
    imshow(Ia)
    h1 = vl_plotframe(fa(:,m1)) ;
    h2 = vl_plotframe(fa(:,m1)) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;
    if length(sel) < 8
        title(['points = ', num2str(sel)])
    end

    vl_tightsubplot(1,2,2);
    imshow(Ib)
    h1 = vl_plotframe(fb(:,m2)) ;
    h2 = vl_plotframe(fb(:,m2)) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;
    if length(sel) < 3
        title(['score = ', num2str(score)])
    end

    set(gca,'Clipping','Off');   % to print lines over 2 subplots

    pointsa = fa(1:2, m1); % xy coordinates of a side
    pointsb = fb(1:2, m2); % xy coordinates of matching b side
    x_trans = size(Ia, 2);

    for point = 1:length(sel)  % number of selected matches  
        % show a line for each matching point with alternating colors
        xa = pointsa(1 ,point);
        ya = pointsa(2, point);
        xb = pointsb(1, point);
        yb = pointsb(2, point);
        colors = 'krgybmw';
        color = colors(mod(point, length(colors))+1);
        h = line([xa-x_trans xb], [ya yb], 'Color', color);
        set(h,'LineWidth',2)
    end
end
end
