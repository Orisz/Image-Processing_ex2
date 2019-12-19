function [code,dict,avglen] = huffcoding(im)
[row,col] = size(im);
num_of_pixels = row*col;
colStack = reshape(im , [num_of_pixels 1]);
[counts,binLocations] = imhist(colStack);
non_zero_indecies = find(counts~=0);
counts_non_zero = counts(non_zero_indecies);
binLocations_non_zero = binLocations(non_zero_indecies);
p = counts_non_zero ./ num_of_pixels;
[dict,avglen] = huffmandict(binLocations_non_zero,p);
code = huffmanenco(colStack,dict);
end

