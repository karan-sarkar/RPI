% This M-file constructs the individual images for 60 digits
% and plots them to a file.

clear
format short g
load zip.train
digits=zip(:,1);
grayscale=zip(:,2:end);

[coeff, score] = pca(grayscale);

[n,d]=size(grayscale);
w=floor(sqrt(d));


%i = 1;
curimage=reshape(grayscale(2,:),w,w);
curimage=curimage';
%displayimage(curimage)

%l=displayimage(curimage);
%sstr=['IndividualImages/image',int2str(i)];
%eval(['print -deps ',sstr]);

data = score(digits == 1 | digits == 5, :);
labels = digits(digits == 1 | digits == 5, :)
gscatter(data(:, 1), data(:, 2), labels,'br','ox')
xlabel('PC1')
ylabel('PC2')