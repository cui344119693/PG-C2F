function main_rw_simple()
    clc; clear; close all;

    % ================= 参数设置 =================
    conf.imgDir = 'NUDT-SIRST/images';             % 原始图片路径
    conf.maskDir = 'NUDT-SIRST/masks_centroid';    % 点标注路径
    conf.outDir = 'NUDT-SIRST/output';% 伪掩码保存路径
    
    % Random Walker 参数
    conf.beta = 200;                    % 按照你示例中的 beta 设置
    
    % BBox 生成参数 (用于确定 local_img 的范围)
    conf.patchSize = 40;       
    conf.sigmaScale = 3.0;     
    conf.expansion = 3;        % 扩张3像素，保证最外圈通常是背景
    conf.minSigma = 1.0;
    conf.maxSigma = 10.0;
    % ===========================================

    if ~exist(conf.outDir, 'dir'), mkdir(conf.outDir); end
    
    imgFiles = dir(fullfile(conf.imgDir, '*.png')); 
    
    fprintf('开始处理 %d 张图片...\n', length(imgFiles));

    for i = 1:length(imgFiles)
        imgName = imgFiles(i).name;
        imgPath = fullfile(conf.imgDir, imgName);
        maskPath = fullfile(conf.maskDir, imgName);

        if ~exist(maskPath, 'file'), continue; end

        imgRaw = imread(imgPath);
        if size(imgRaw, 3) == 3, imgGray = rgb2gray(imgRaw); else, imgGray = imgRaw; end
        imgDouble = im2double(imgGray);
        
        maskRaw = imread(maskPath);
        if size(maskRaw, 3) == 3, maskRaw = rgb2gray(maskRaw); end
        
        full_mask = zeros(size(imgDouble));

        % 获取标注点
        stats = regionprops(maskRaw > 0, 'Centroid');
        
        for k = 1:length(stats)
            centroid = stats(k).Centroid;
            cx = round(centroid(1));
            cy = round(centroid(2));

            % 1. 获取高斯拟合的 BBox
            %    这个 BBox 将作为我们的 local_img 范围
            bbox = fit_gaussian_and_get_bbox(imgDouble, cx, cy, conf);
            
            x1 = bbox(1); y1 = bbox(2);
            x2 = bbox(3); y2 = bbox(4);
            
            % 2. 截取局部图像 (local_img)
            local_img = imgDouble(y1:y2, x1:x2);
            [H_loc, W_loc] = size(local_img);
            
            % 3. 构建种子 (Seeds) 和 标签 (Labels)
            % 映射中心点到局部坐标
            loc_cx = cx - x1 + 1;
            loc_cy = cy - y1 + 1;
            
            % [前景种子]: 原始标注点 -> Label 1
            fg_ind = sub2ind([H_loc, W_loc], loc_cy, loc_cx);
            
            % [背景种子]: 局部图最外面一圈 -> Label 2
            tmp_map = zeros(H_loc, W_loc);
            tmp_map(1,:) = 1;       % 上边
            tmp_map(end,:) = 1;     % 下边
            tmp_map(:,1) = 1;       % 左边
            tmp_map(:,end) = 1;     % 右边
            
            tmp_map(loc_cy, loc_cx) = 0; 
            
            bg_ind = find(tmp_map == 1);
            
            seeds = [fg_ind; bg_ind];
            labels = [1; 2 * ones(length(bg_ind), 1)];
            
            % 4. Random Walker
            try
                [~, probabilities] = random_walker(local_img, seeds, labels, conf.beta);
                
                prob_fg = probabilities(:,:,1);
                
                local_mask = prob_fg > 0.1;
                
                % 5. 填回全图
                roi_mask = full_mask(y1:y2, x1:x2);
                full_mask(y1:y2, x1:x2) = roi_mask | local_mask;
                
            catch ME
                fprintf('Error on %s: %s\n', imgName, ME.message);
            end
        end

        imwrite(uint8(full_mask * 255), fullfile(conf.outDir, imgName));
        
        if mod(i, 10) == 0
            fprintf('已处理 %d / %d\n', i, length(imgFiles));
        end
    end
    fprintf('处理完成。结果保存在: %s\n', conf.outDir);
end

%% ================= 高斯 BBox 生成函数 (保持不变) =================
function final_bbox = fit_gaussian_and_get_bbox(img, cx, cy, conf)
    [H, W] = size(img);
    halfSize = floor(conf.patchSize / 2);

    % Patch 范围
    px1 = max(1, cx - halfSize); py1 = max(1, cy - halfSize);
    px2 = min(W, cx + halfSize); py2 = min(H, cy + halfSize);
    
    patch = img(py1:py2, px1:px2);
    [Hp, Wp] = size(patch);
    
    raw_box = [cx-2, cy-2, cx+2, cy+2]; 

    if Hp >= 5 && Wp >= 5 && (max(patch(:)) - min(patch(:)) > 1e-5)
        [xMesh, yMesh] = meshgrid(1:Wp, 1:Hp);
        xData = [xMesh(:), yMesh(:)];
        zData = patch(:);
        zNorm = (zData - min(zData)) / (max(zData) - min(zData));

        gauss2D = @(p, xy) p(6) + p(1) * exp( - ( ...
            (xy(:,1) - p(2)).^2 ./ (2*p(4)^2) + ...
            (xy(:,2) - p(3)).^2 ./ (2*p(5)^2) ) );
            
        p0 = [1, Wp/2, Hp/2, 2, 2, 0];
        lb = [0, -Wp, -Hp, conf.minSigma, conf.minSigma, -1]; 
        ub = [2, 2*Wp, 2*Hp, conf.maxSigma, conf.maxSigma, 1];
        
        opts = optimset('Display', 'off', 'TolX', 1e-3);
        try
            pOpt = lsqcurvefit(gauss2D, p0, xData, zNorm, lb, ub, opts);
            g_cx = px1 + pOpt(2) - 1; g_cy = py1 + pOpt(3) - 1;
            rx = conf.sigmaScale * pOpt(4); ry = conf.sigmaScale * pOpt(5);
            raw_box = [g_cx - rx, g_cy - ry, g_cx + rx, g_cy + ry];
        catch
        end
    end

    bx1 = raw_box(1); by1 = raw_box(2); bx2 = raw_box(3); by2 = raw_box(4);

    % 约束：包含中心点
    bx1 = min(bx1, cx); by1 = min(by1, cy);
    bx2 = max(bx2, cx); by2 = max(by2, cy);
    
    % 扩张
    bx1 = bx1 - conf.expansion; by1 = by1 - conf.expansion;
    bx2 = bx2 + conf.expansion; by2 = by2 + conf.expansion;
    
    final_x1 = floor(max(px1, bx1));
    final_y1 = floor(max(py1, by1));
    final_x2 = ceil(min(px2, bx2));
    final_y2 = ceil(min(py2, by2));

    if final_x2 < final_x1, final_x2 = final_x1; end
    if final_y2 < final_y1, final_y2 = final_y1; end
    
    final_bbox = [final_x1, final_y1, final_x2, final_y2];
end
