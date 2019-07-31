clear all;clc;

v = VideoReader('snow_cut.mp4');
nFrames = v.NumberOfFrames;
IMG = [];
for i=1:nFrames
img = rgb2gray(read(v,i)); % get one RGB image
img = reshape(img,720*720,1);
IMG = [IMG img];
end
IMG = double(IMG);
t2 = linspace(0,v.CurrentTime,nFrames+1);
t = t2(1:end-1);
dt=t(2)-t(1); 

%%
X1 = IMG(:,1:end-1); X2 = IMG(:,2:end);

[U2,Sigma2,V2] = svd(X1, 'econ');
figure(1)
plot(diag(Sigma2)/sum(diag(Sigma2)),'ro')
ylabel('Singular Values')
xlabel('Image Framse')
title('Singular Value Spectrum on snow video')
print(gcf,'-dpng','rank_snow.png');
r=1; 
U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);
Atilde = U'*X2*V/Sigma; % low rank subspace of A
[W,D] = eig(Atilde);
Phi=X2*V/Sigma*W; % DMD modes, project back out to real space

mu=diag(D);
omega=log(mu)/dt; % e^mu = omega, DMD eigenvalues

x1 = X1(:,1);
y0 = Phi\x1; 

% reconstruct in time
modes = zeros(r,length(t));
for iter = 1:length(t)
    modes(:,iter) = (y0.*exp(omega*t(iter)));
end;
Xdmd = Phi * modes;

%%
X_sparse = IMG-Xdmd;
R_Matrix = X_sparse.*(X_sparse<0);
X_sparse_dmd = X_sparse - R_Matrix;
%%
figure(2)
for i = 1:nFrames
    subplot(1,3,1)
    img1 = uint8(IMG(:,i));
    imshow(reshape(img1,720,720))
    title('Original Video')
    subplot(1,3,2)
    img2 = uint8(Xdmd(:,i));
    imshow(reshape(img2,720,720))
    title('Background')
    subplot(1,3,3)
    img3 = real(X_sparse_dmd(:,i));
    imshow(reshape(img3,720,720))
    title('Foreground')
    drawnow
end
print(gcf,'-dpng','Images_snow.png');
