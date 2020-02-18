clear all;  % all variable cleared
close all;  % all figures closed

%%%%%%% VITON DATASET %%%%%%%%%%%%%%%%%%%%
%%%  train   --+-- cloth      : cloth images [hxw =256x192]  jpg
%%%  or test   +-- cloth-mask : FG mask of cloth images [fg: white]  %%% Some are not clean, JPG ^^
%%%            +-- image      : model image [256x192x3] jpg 
%%%            +-- image-pare : segmentation label image PNG
%%%            +-- pose       : joint info JSON 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATA_ROOT    ='../../data/test/';   
% RESULT_FOLDER = '../gmmplus_test_test_1STN_gic_DT1_1_40_same/';
% RESULT_FOLDER = '../gmm_test_test_affine_nogic_same/';
RESULT_FOLDER = '../gmm_test_test_affine_nogic_TPS_1_40_same/';
RESULT_ROOT = [RESULT_FOLDER,'test/warp-cloth/'];

pairs_file = './test_pairs.txt';
[image1, image2] = textread(pairs_file, '%s %s');
result = [];
for i = 1:length(image1) % only run over 1 image (for now)
    image_name1 = image1{i};
    image_name2 = image2{i};
    
    
    gmm_out_file = [RESULT_ROOT,  image_name1];    
    gmm_gt_file  = [DATA_ROOT, 'image-parse/', strrep(image_name1, 'jpg', 'png')];
    gmm_gt_image_file  = [DATA_ROOT, 'image/', image_name1];
    
    %disp(gmm_out_file);
    %disp(gmm_gt_file);
    
    %figure(1);
    %subplot(2,2,1);
    gmm_out_img = imread(gmm_out_file);    
    %imshow(gmm_out_img);
    %title('gmm out');
    %subplot(2,2,2);
    gmm_gt_img = imread(gmm_gt_file);
    
    gmm_gt_extracted = imread(gmm_gt_image_file);
    %imshow(gmm_gt_extracted);
    gmm_gt_img = gmm_gt_img == 5;
    gmm_gt_img_inv = uint8(gmm_gt_img ~= 1)*255;
    gmm_gt_extracted = double(gmm_gt_extracted) .* cat(3,double(gmm_gt_img),double(gmm_gt_img), double(gmm_gt_img));
    gmm_gt_extracted = gmm_gt_extracted + cat(3, double(gmm_gt_img_inv),double(gmm_gt_img_inv), double(gmm_gt_img_inv));
    gmm_gt_extracted = uint8(gmm_gt_extracted);

    
    % evaluate the GMM by IoU
    % 1. convert to logical type 
    gmm_out_img = gmm_out_img > 20;  % TODO
    uinon_area = gmm_gt_img | gmm_out_img;
    intersect_area = gmm_gt_img & gmm_out_img;
    xor_area = xor( gmm_gt_img , gmm_out_img);
    iouval = sum(intersect_area(:))/sum(uinon_area(:));
    
 
    % evaluate the TON by SSIM 
    gmm_out_img = imread(gmm_out_file); 
    
    [ssimval,ssimmap] = ssim(gmm_out_img, gmm_gt_extracted); 
    
    msg = sprintf('%d : IOU=%f, SSIM=%f', i, iouval, ssimval);
    disp(msg);    
    
    result = [result; [iouval, ssimval]];
end
disp('mean IOU, mean SSIM');
disp(mean(result));
csvwrite([RESULT_FOLDER,'iou_ssim.csv'],result)
