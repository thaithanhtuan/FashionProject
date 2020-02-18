% Heejune Ahn 2019
% Evaluating IoU of GMM and SSIM of TON for VITON data format 
% 

% Heejune Ahn 2019
% Evaluating IoU of GMM and SSIM of TON for VITON data format 
% 
clear all;  % all variable cleared
close all;  % all figures closed

%%%%%%% VITON DATASET %%%%%%%%%%%%%%%%%%%%
%%%  train   --+-- cloth      : cloth images [hxw =256x192]  jpg
%%%  or test   +-- cloth-mask : FG mask of cloth images [fg: white]  %%% Some are not clean, JPG ^^
%%%            +-- image      : model image [256x192x3] jpg 
%%%            +-- image-pare : segmentation label image PNG
%%%            +-- pose       : joint info JSON 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATA_ROOT    ='D/Datasets/viton_resize/test/';            
% DATA_ROOT    ='../Dataset/viton_resize/train/';           %%% TODO 
% RESULT_ROOT  ='./results_classified02/stage2/images/';
RESULT_ROOT = './CP-VTON/full test set - different clothes/gmm_final.pth/test/';


% IoU for GMM
% warped cloth mask   : ./results/stage2/images/humanNumber_0.jpg_clothNumber_1.jpg_mask.png
% GT (segmentation):  DATA_ROOT/viton_resize/test/image-parse/xxxxxxx_0.png

% SSIM for TON
% result viton : ./<results>/stage2/images/humanNumber_0.jpg_clothNumber_1.jpg_final.png
% GT:          :   DATA_ROOT/viton_resize/test/image/xxxxxxx_0.png


%{
labels = {"background", #     0
            "hat", #            1
            "hair", #           2 
            "sunglass", #       3
            "upper-clothes", #  4
            "skirt",  #          5
            "pants",  #          6
            "dress", #          7
            "belt", #           8
            "left-shoe", #      9
            "right-shoe", #     10
            "face",  #           11
            "left-leg", #       12
            "right-leg", #      13
            "left-arm",#       14
            "right-arm", #      15   
            "bag", #            16
            "scarf" #          17    
        ]  
%}

% DATA_TOP ='D:\3.Project\9.Fashion\3.Dataset\VITON_TPS\viton_resize\train';

%DATA_ROOT='cloth/woman_top';               % in-shop cloth 
%DATA_ROOT= [DATA_TOP,'/image-parse/'];       
%MODEL_ROOT=[DATA_TOP, '/image/'];       

%MASK_DIR='results/stage1/tps/00015000_';   % magmm_gt_imgsk for cloth area in model using NN model
%MASK_DIR  = [DATA_TOP,'/cloth-mask/'];
%CLOTH_DIR = [DATA_TOP,'/cloth/'];

% pairs_file = 'data/viton_test_pairs_classified_linux.txt';
% [image1, image2, comment] = textread(pairs_file, '%s %s %s');


% using a smaller height and width for the shape context matching
% can save time without hurting the perform too much.

% only run over 1 image (for now)

% image_name1 = image1{i};
% image_name2 = image2{i};

% image_name1 = '000097_0.jpg';
% image_name2 = '019522_1.jpg';

% gmm_out_file = [RESULT_ROOT,  image_name1, '_', image_name2, '_mask.png'];    
% gmm_gt_file  = [DATA_ROOT, 'image-parse/', strrep(image_name1, 'jpg', 'png')];

% gmm_out_file = 'D:/Research/Fashion-Project-SeoulTech/7. 2D VTON/Results/viton_original_test_results/results/stage2/images/007129_0.jpg_000220_1.jpg_mask.png';    % warped cloth mask

% gmm_out_file = 'D:/Research/Fashion-Project-SeoulTech/7. 2D VTON/Results/CP-VTON/full test set - different clothes/gmm_final.pth/test/warp-mask/010543_1.jpg';    % warped cloth mask
% gmm_out_file = 'D:/Research/Fashion-Project-SeoulTech/7. 2D VTON/Results/CP-VTON/full test set - same clothes/gmm_final.pth/test/warp-mask/010543_0.jpg';    % warped cloth mask
% gmm_out_file = 'D:/Research/Fashion-Project-SeoulTech/7. 2D VTON/Results/same-model-experiment/cp-vton/diff-clothes/gmm_final.pth/test/warp-mask/05_1.jpg';

% gmm_out_file = 'D:/Research/Fashion-Project-SeoulTech/7. 2D VTON/Results/CP-VTON+/full test set - same clothes/gmm_final.pth/test/warp-mask/010543_0.jpg';    % warped cloth mask
% gmm_out_file = 'D:/Research/Fashion-Project-SeoulTech/7. 2D VTON/Results/CP-VTON+/full test set - different clothes/gmm_final.pth/test/warp-mask/010543_1.jpg';    % warped cloth mask
gmm_out_file = 'D:/Research/Fashion-Project-SeoulTech/7. 2D VTON/Results/same-model-experiment/cp-vton+/diff-clothes/gmm_final.pth/test/warp-mask/05_1.jpg';

% gmm_gt_file  = 'D:/Datasets/viton_resize/test/image-parse/000877_0.png';    % target human segmentation
gmm_gt_file  = 'D:/Datasets/SeoulTechFashion/same-model-dataset/image-parse/05_0.png';    % target human segmentation

% ton_out_file = [RESULT_ROOT,  image_name1, '_', image_name2, '_final.png'];
% ton_gt_file  = [DATA_ROOT, 'image/', image_name1];

disp(gmm_out_file);
disp(gmm_gt_file); 

% figure(1);
% subplot(2,2,1);
% subplot(1,2,1);
gmm_out_img = imread(gmm_out_file);    
% imshow(gmm_out_img);
% title('gmm out');
% subplot(2,2,2);
% subplot(1,2,2);
gmm_gt_img = imread(gmm_gt_file);    
gmm_gt_img = gmm_gt_img == 5;
% imshow(gmm_gt_img);
% title('gmm gt');

%unique(gmm_gt_img(:))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate the result 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluate the GMM by IoU
% 1. convert to logical type 
gmm_out_img = gmm_out_img > 20;  % TODO
union_area = gmm_gt_img | gmm_out_img;
intersect_area = gmm_gt_img & gmm_out_img;
xor_area = xor( gmm_gt_img , gmm_out_img);
iouval = sum(intersect_area(:))/sum(union_area(:));


% evaluate the TON by SSIM 
% [ssimval,ssimmap] = ssim(ton_out_img, ton_gt_img); 

% msg = sprintf('IOU=%f, SSIM=%f', iouval, ssimval);
msg = sprintf('IOU=%f', iouval);
disp(msg);

% figure(2);
% subplot(1,1,1);
% imshow(xor_area);

%subplot(1,2,2); 
%imagesc(ssimmap); %imshow(uint8(warp_im*255.0));
%axis('image');
%title(['overlayed(', msg, ')']);
%drawnow;