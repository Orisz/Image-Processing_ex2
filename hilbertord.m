function d = hilbertord(src,dir)
% Extract source values in direct or inverse Hilbert curve order
% (direct means from columnwise to hilbert)
%
% src - source values (in vector form)
% dir - direct or inverse.
%
% d - destination container (in vector form)

n = log2(sqrt(length(src)));
[xd,yd] = hilbertc(n);
c(:,1) = floor(((xd+0.5)*2^n)+1);
c(:,2) = 2^n - floor(((yd+0.5)*2^n)+1) + 1;

if strcmp(dir,'inverse')
    M = zeros(sqrt(length(src)));
    for i = 1:length(src)
        M(c(i,1),c(i,2)) = src(i);
    end
    d = M(:)';
elseif strcmp(dir,'direct')
    M = reshape(src,sqrt(length(src)),[]);
    d = zeros(1,size(c,1));
    for i = 1:length(d)
        d(i) = M(c(i,1),c(i,2));
    end
else
    d = -1;
end

return;

