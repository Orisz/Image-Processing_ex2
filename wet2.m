%%
% wet2_201536109_201110830
clc;close all;clear all;
%% Q1 a
crazyBioComp = imread('../crazyBioComp.jpg');
imshow(crazyBioComp);

SE1 = strel('square',12);
SE2 = strel('disk',4,8);

plot1= {imerode(crazyBioComp,SE1),imdilate(crazyBioComp,SE1),imopen(crazyBioComp,SE1),...
    imclose(crazyBioComp,SE1),imtophat(crazyBioComp,SE1),imbothat(crazyBioComp,SE1)};

plot2 = {imerode(crazyBioComp,SE2),imdilate(crazyBioComp,SE2),imopen(crazyBioComp,SE2),...
    imclose(crazyBioComp,SE2),imtophat(crazyBioComp,SE2),imbothat(crazyBioComp,SE2)};
titles_1 = {'erode , square' , 'dialate , square' , 'open , square' ,...
    'close , square' , 'tophat , square' , 'bothat , square'};

titles_2 = {'erode , disk' , 'dialate , disk' , 'open , disk' ,...
    'close , disk' , 'tophat , disk' , 'bothat , disk'};
figure(2); 
for j = 1:6
    subplot(2,3,j);
    imshow(cell2mat(plot1(j)), []);
    title(titles_1(j));
end
 figure(3);
  for j = 1:6
    subplot(2,3,j);
    imshow(cell2mat(plot2(j)), []);
    title(titles_2(j));
  end
%% Q1 b
keyboard = imread('../keyboard.jpg');
figure(4);
imshow(keyboard);
%% Q1 c
 SE3 = strel('line',8,90);
 SE4 = strel('line',8,0);
 SE5 = strel('square',8);
 
erode_vertical = imerode(keyboard,SE3);
erode_horizontal = imerode(keyboard,SE4);
 
 figure(5);
 subplot(1,2,1);
 imshow(erode_vertical);
 title('vertical erode');
 subplot(1,2,2);
 imshow(erode_horizontal);
 title('horizontal erode');
 

%% Q1 d
figure(6)
KeyBoardSum= erode_vertical + erode_horizontal; 
subplot (2,1,1)
imshow(KeyBoardSum,[]);
title('keyboard erode sum'); 
subplot (2,1,2)
bwKeyBoardSum = im2bw(KeyBoardSum, 0.2);
imshow(bwKeyBoardSum ,[]);
title('keyboard black&white sum');
%% Q1 e
negKBsum = not(bwKeyBoardSum);
medFilt_KB = medfilt2(negKBsum , [8 8]);
figure(7);
imshow(medFilt_KB , []);
%% Q1 f
eroded_filt_KB = imerode(medFilt_KB , SE5);
figure(8);
imshow(eroded_filt_KB , []);
%% Q1 g
eroded_filt_KB_uint8 = uint8(eroded_filt_KB);
comman_image = eroded_filt_KB_uint8 .* keyboard;
sharp_comman_image = imsharpen(comman_image);
figure(9);
imshow(sharp_comman_image , []);
%% Q1 h
final = im2bw(sharp_comman_image , 0.2);%using histogram we found a good threshold
figure(10);
imshow(final ,[]);
% END Q1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc; clear all; close all;
%% Q2 section a
tresh_1 = 0.065;
tresh_2 = 0.07;
tresh_3 = 0.13;
sigma = 3.5;
Original_Pic = im2double(imread('../Bananas.jpg'));
Sobel_Edge = edge(Original_Pic,'Sobel',tresh_1);
Roberts_Edge = edge(Original_Pic,'Roberts',tresh_2);
Canny_Edge = edge(Original_Pic,'Canny',tresh_3, sigma); %finds edges by looking for local maxima of the gradient of the picture

figure();
subplot(1,3,1);
imshow(Sobel_Edge);
title(['Bananas - Sobel method, threshold ' num2str(tresh_1)]);
subplot(1,3,2);
imshow(Roberts_Edge);
title(['Bananas - Roberts method, threshold ' num2str(tresh_2)]);
subplot(1,3,3);
imshow(Canny_Edge);
title(['Bananas - Canny method, threshold ' num2str(tresh_3) ' sigma ' num2str(sigma)]);
suptitle('Edge detection');

%% Section b
Noised_Pic = imnoise(imread('../Bananas.jpg'),'poisson');
Sobel_Edge = edge(Noised_Pic,'Sobel',tresh_1);
Roberts_Edge = edge(Noised_Pic,'Roberts',tresh_2);
Canny_Noised_Edge = edge(Noised_Pic,'Canny',tresh_3, sigma); %finds edges by looking for local maxima of the gradient of the picture

figure();
subplot(1,3,1);
imshow(Sobel_Edge);
title(['Noised Bananas - Sobel method, threshold ' num2str(tresh_1)]);
subplot(1,3,2);
imshow(Roberts_Edge);
title(['Noised Bananas - Roberts method, threshold ' num2str(tresh_2)]);
subplot(1,3,3);
imshow(Canny_Noised_Edge);
title(['Noised Bananas - Canny method, threshold ' num2str(tresh_3) ' sigma ' num2str(sigma)]);
suptitle('Edge detection');

%% Section c
Adaptive_Canny_Edge = mod_edge(Original_Pic,'Canny',tresh_3, sigma);
figure();
subplot(1,2,1);
imshow(Adaptive_Canny_Edge);
title('canny new method');
subplot(1,2,2);
imshow(Canny_Edge);
title('canny old method');

%% Section d
a=0;
figure()
for i=1:6
    h = [0, -a, 0; -a, (1+4*a), -a; 0, (-a), 0];
    filtered_img=imfilter(Original_Pic, h);
    filtered_img(find(filtered_img>1))=1; %clipping
    filtered_img(find(filtered_img<0))=0;
    subplot(2,3,i);
    imshow(filtered_img);
    title(['Bananas - Laplacian, a= ' num2str(a)]);
    a=a+0.125;
end

%% Section e
a=0.25;
Noised_Pic = imnoise(Original_Pic,'salt & pepper',0.04);
figure()
for i=1:3
    h = [0, -a, 0; -a, (1+4*a), -a; 0, (-a), 0];
    filtered_img=imfilter(Noised_Pic, h);
    filtered_img(find(filtered_img>1))=1; %clipping
    filtered_img(find(filtered_img<0))=0;
    subplot(2,3,i);
    imshow(filtered_img);
    title(['S&P noised Bananas - Laplacian, a= ' num2str(a)]);
    a=a+0.25;
end

%% Section f
a=0.25;
Noised_Pic = imnoise(Original_Pic,'salt & pepper',0.04);
Filtered_pic = medfilt2(Noised_Pic);
figure()
for i=1:3
    h = [0, -a, 0; -a, (1+4*a), -a; 0, (-a), 0];
    filtered_img=imfilter(Filtered_pic, h);
    filtered_img(find(filtered_img>1))=1; %clipping
    filtered_img(find(filtered_img<0))=0;
    subplot(2,3,i);
    imshow(filtered_img);
    title(['Filtered S&P noised Bananas - Laplacian, a= ' num2str(a)]);
    a=a+0.25;
end
% END Q2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Q3 section a - find a match of a template inside an image text
clc;clear all;close all;

Text_pic = im2double(imread('..\Text.jpg'));

%prepare matches
E_10 = im2double(imread('..\E10.jpg'));
E_11 = im2double(imread('..\E11.jpg'));
E_12 = im2double(imread('..\E12.jpg'));
E_14 = im2double(imread('..\E14.jpg'));
E_16 = im2double(imread('..\E16.jpg'));

figure()
e1 = subplot(2,3,1)
imshow(E_10);
title('letter E - size 10');
e2 = subplot(2,3,2);
imshow(E_11);
title('letter E - size 11');
e3 = subplot(2,3,3);
imshow(E_12);
title('letter E - size 12');
e4 = subplot(2,3,4);
imshow(E_14);
title('letter E - size 14');
e5 = subplot(2,3,5);
imshow(E_16);
title('letter E - size 16');
linkaxes([e5,e4,e3,e2,e1])

%find best result
Letters_vec = {E_10, E_11, E_12, E_14, E_16};
res_vec = zeros(size(Letters_vec));
for i = 1:size(res_vec,2)
    s_mat = SSD_find(Text_pic, Letters_vec{i});
    res_vec(i) = min(s_mat(:));
end

[M, idx] = min(res_vec);
E_str = [10, 11, 12, 14, 16];
fprintf(['best result is letter: E' num2str(E_str(idx))]);

%% Section b - identify number of letters appearence in an image text
%crop letters from the text
a = Text_pic(14:26,57:64);              
A = Text_pic(149:163,17:26);
t = Text_pic(56:70,36:42);
T = Text_pic(117:132,15:25);

figure()
suptitle('Letters to find');
subplot(2,2,1);
imshow(a);
subplot(2,2,2);
imshow(A);
subplot(2,2,3);
imshow(t);
subplot(2,2,4);
imshow(T);

%find best result
Letters_vec = {a, A, t, T};
count_vec = zeros(size(Letters_vec));
for i = 1:size(Letters_vec,2)
    s_mat = SSD_find(Text_pic, Letters_vec{i});
    thresh = max(s_mat(:))*0.05;
    s_threshed_mat = s_mat < thresh;
    count_vec(i) = sum(s_threshed_mat(:));
end

%% Section c - letters replacement in an image text
figure()
subplot(1,2,1);
imshow(Text_pic);
title('Text without replacements');

letter_c = im2double(imread('..\c.jpg')); % to replace
letter_k = im2double(imread('..\k.jpg')); % to replace with
[M,N] = size(letter_c);

%find a match
s_mat = SSD_find(Text_pic, letter_c);
thresh = max(s_mat(:))*0.05;
s_threshed_mat = s_mat < thresh;
replace_counter = sum(s_threshed_mat(:));

%replace
[x_idx, y_idx]= find(s_threshed_mat);
 for i=1:replace_counter
     Text_pic(x_idx(i)-floor(M/2):x_idx(i)+floor(M/2), y_idx(i)-floor(N/2):y_idx(i)+floor(N/2)) = letter_k;
 end

%show
subplot(1,2,2);
imshow(Text_pic);
title('Text with replacements');

% END Q 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc;clear all;close all;

%% Q 4 b
heisenberg = imread('../heisenberg.jpg');
heisen_entropy = getEntropy(heisenberg);
%% Q4 d
[code,dict,avglen] = huffcoding(heisenberg);
[row , col] = size(heisenberg);
num_of_pixels = row*col;
num_of_bits = num_of_pixels*8;
codeLength = length(code);
r = num_of_bits / codeLength;
%% Q4 e
dsig = huffmandeco(code,dict);
dec_im = uint8(reshape(dsig ,[row , col]));
imshow(dec_im ,[]);
MSE = immse(heisenberg,dec_im);
%% Q4 f
mauritius = imread('../mauritius.jpg');
gray_mauritius = rgb2gray(mauritius);
mauritius_entropy = getEntropy(gray_mauritius);
[~,~,avglen_mauritius] = huffcoding(gray_mauritius);
%% Q4 g
sunset = imread('../sunset.jpg');
sunsetCS = sunset(:);
sunsetHilbert = hilbertord(sunsetCS,'direct');
CSDiff = diff(sunsetCS);
HilbertDiff = diff(sunsetHilbert);
CSDiff = [sunsetCS(1) ; CSDiff];
HilbertDiff = [sunsetHilbert(1), HilbertDiff];
[row , col] = size(sunset);
im_CSDiff = reshape(CSDiff , [row , col]);
im_HilbertDiff = uint8(reshape(HilbertDiff, [row , col]));
figure();
subplot(1,2,1);
imhist(im_CSDiff);
title('CS');
subplot(1,2,2);
imhist(im_HilbertDiff);
title('Hilbert');
%huffman
[CS_code,CS_dict,CS_avglen] = huffcoding(im_CSDiff);
[Hilbert_code,Hilbert_dict,Hilbert_avglen] = huffcoding(im_HilbertDiff);
%% Q4 h
ampersand = imread('../ampersand.jpg');
[row_ampersand , col_ampersand] = size(ampersand);
ampersand_Entropy = getEntropy( ampersand );
[~,~,ampersand_avglen] = huffcoding(ampersand);
%zero_indecis = find(ampersand<1);
%ampersand_positive(zero_indecis) = 1;
[ampersand_counts,ampersand_binLocations] = imhist(ampersand);
non_zero_indecies = find(ampersand_counts~=0);
counts_non_zero = ampersand_counts(non_zero_indecies);
binLocations_non_zero = ampersand_binLocations(non_zero_indecies);
binLocations_non_zero = binLocations_non_zero + 1;
%another methos where the symbol is simply the picture 
symbol = ampersand(:);
symbol = symbol +1;
%end of another method
ampersand_code = arithenco(symbol,counts_non_zero);
avg_arith_len = length(ampersand_code)/(row_ampersand*col_ampersand);