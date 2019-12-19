function [ Entropy ] = getEntropy( image )
%this func get image and by calc the histogram 
%find each gray level probability and calc the entropy
[counts,~] = imhist(image);
dim = size(image);
numOfPixels = dim(1)*dim(2);
p = counts ./ numOfPixels;
non_zero_ind = find(p~=0);
non_zero_p = p(non_zero_ind);
Entropy = -sum(non_zero_p.*log2(non_zero_p));
end

