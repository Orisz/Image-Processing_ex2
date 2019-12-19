function [ S_mat ] = SSD_find( image, E_option )
    
    %calculates the SSD 
    [M,N] = size(E_option);
    I_pow2 = imfilter(image.^2, ones(M,N));
    T_pow2 = E_option.*E_option;
    S_mat = I_pow2 - 2*imfilter(image, E_option) + sum(T_pow2(:));  
end

