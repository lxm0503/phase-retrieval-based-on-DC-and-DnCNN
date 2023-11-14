%This file demonstrates my PR demo.

addpath(genpath('..'))
addpath('D:\app\matlab2022a\toolbox\matconvnet\matconvnet-1.0-beta25\matlab');
addpath(genpath('E:\study\prDeep-master\prDeep-master\D-AMP_Toolbox-master'));
addpath(genpath('E:\study\prDeep-master\prDeep-master\fasta-matlab-master'));

%Parameters
denoiser_RED='DnCNN';%Available options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, BM3D-SAPCA, and DnCNN 
SamplingRate=4;
imsize=128; 
num_trials=4;%Number of initializations to try.
filename='house.png';
n_DnCNN_layers=20;%Other option is 17
LoadNetworkWeights(n_DnCNN_layers);

ImIn=double(imread(filename));
x_0=imresize(ImIn,imsize/size(ImIn,1));%128*128
[height, width]=size(x_0);%128
n=length(x_0(:));%128*128=16384
m=round(n*SamplingRate);%May be overwritten when using Fourier measurements 256*256=65536
errfxn = @(x_hat) PSNR(x_0,reshape(x_hat,[height width]));

measurement_method='fourier';
noise_model='poisson';
%noise_model='gaussian';
alpha=6;%Amount of Poisson noise to add 
SNR=nan;%Not used for Poisson noise
switch lower(measurement_method)
    case 'gaussian'
        %Generate Gaussian Measurement Matrix
        M_matrix=1/sqrt(2)*(1/sqrt(m)*randn(m,n)+1i/sqrt(m)*randn(m,n));
        M_matrix_pinv=pinv(M_matrix);
        M=@(x) M_matrix*x(:);
        Mpinv=@(z) M_matrix_pinv*z(:);
        Mt=@(z) M_matrix'*z(:);
        Mt_asym=Mt;
    case 'coded diffraction'
        %Generate Coded Diffraction Pattern Measurement Matrix
        if m<n
            signvec = exp(1i*2*pi*rand(n,1));
            inds=[1;randsample(n-1,m-1)+1];
            I=speye(n);
            SubsampM=I(inds,:);
            M=@(x) SubsampM*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*(1/sqrt(n))*sqrt(n/m);
            Mt=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM'*x(:),[height,width])),[n,1]))*sqrt(n)*sqrt(n/m);
            Mpinv=Mt;
            U=@(x) x(:);
            Ut= @(x) x(:);
            d=ones(m,1)*n/m;
        else
            if round(m/n)~=m/n
                error('Oversampled coded diffraction patterns need m/n to be an integer')
            end
            signvec = exp(1i*2*pi*rand(m,1));
            M=@(x) reshape(fft2(reshape(bsxfun(@times,signvec,repmat(x(:),[m/n,1])),[height,width,m/n])),[m,1])*(1/sqrt(n));
            Mt=@(z) reshape(sum(reshape(bsxfun(@times,conj(signvec),reshape(ifft2(reshape(z(:),[height,width,m/n])),[m,1])),[height,width,m/n]),3),[n,1])*sqrt(n);
            M_matrix=@(x) reshape(M(x(:)),[height,width,m/n]);
            Mt_matrix=@(z) reshape(Mt(z(:)),[height,width]);%SPAR wants the pseudoinverse, not the adjoint
            Mpinv_matrix=@(z) reshape(Mt(z(:)),[height,width])/(m/n);%SPAR wants the pseudoinverse, not the adjoint

            Mpinv=@(z) Mt(z)*n/m;
            U=@(x) x(:);
            Ut= @(x) x(:);
            d=ones(m,1)*n/m;
        end
        Mt_asym=Mt;
    case 'fourier'
        if m<=n
            error('Undersampled Fourier measurements not supported');
        else
            I=speye(m);
            mask=padarray(ones(height,width),[1/2*(sqrt(m)-sqrt(n)),1/2*(sqrt(m)-sqrt(n))]);
            OversampM=I(:,logical(mask(:)));
            M=@(x) reshape((fft2(reshape(OversampM*x(:),[sqrt(m/n)*height,sqrt(m/n)*width]))),[m,1])*(1/sqrt(m))*sqrt(n/m);
            Mt=@(z) OversampM'*reshape((ifft2(reshape(z(:),[sqrt(m/n)*height,sqrt(m/n)*width]),'symmetric')),[m,1])*sqrt(m)*sqrt(n/m);
            Mt_asym=@(z) OversampM'*reshape((ifft2(reshape(z(:),[sqrt(m/n)*height,sqrt(m/n)*width]))),[m,1])*sqrt(m)*sqrt(n/m);
            Mpinv=@(z) m/n*Mt(z);
            
            M_matrix=@(x) reshape(M(double(x(:))),[sqrt(m),sqrt(m)]);
            Mt_matrix=@(z) reshape(Mt(double(z(:))),[height,width]);
            Mpinv_matrix=@(z) reshape(Mpinv(double(z(:))),[height,width]);
            
            M_square=@(x) reshape((fft2(reshape(x(:),[sqrt(m/n)*height,sqrt(m/n)*width]))),[m,1])*(1/sqrt(m))*sqrt(n/m);%x must be 
%             Mt_square=@(z) reshape((ifft2(reshape(z(:),[sqrt(m/n)*height,sqrt(m/n)*width]))),[m,1])*sqrt(m)*sqrt(n/m);
            Mt_square=@(z) reshape((ifft2(reshape(z(:),[sqrt(m/n)*height,sqrt(m/n)*width]),'symmetric')),[m,1])*sqrt(m)*sqrt(n/m);
            Minv_square=@(z) m/n*Mt_square(z);
            U=@(x) x(:);
            Ut= @(x) x(:);
            d=ones(m,1)*n/m;
        end
    otherwise
        error('unrecognized measurement method');
end

%Sample the image
z=M(x_0(:));
switch lower(noise_model)
    case 'gaussian'
        noise=randn(m,1);
        %noise=10*randn(m,1);
        noise=noise*norm(z)/norm(noise)/SNR;
        y=abs(z)+noise;
        sigma_w=std(noise);
    case 'rician'
        noise=1/sqrt(2)*(randn(m,1)+1i*randn(m,1));
        noise=noise*norm(z)/norm(noise)/SNR;
        y=abs(z+noise);
        sigma_w=std(noise);
    case 'poisson'
%         kappa=1/alpha^2;%Parameter for SPAR.
        intensity_noise=alpha*abs(z).*randn(m,1);
        z2=abs(z).^2;
        y2=abs(z).^2+intensity_noise;
        y2=y2.*(y2>0);
%         y2_spar=y2/alpha^2;
%         y2=poissrnd(z2*kappa);
        y=sqrt(y2);
        err=y-abs(z);
        sigma_w=std(err);
end

best_resid=inf;
for ii=1:num_trials

if isequal(lower(measurement_method),'fourier')
    support=reshape((OversampM*x_0(:))>0,[sqrt(m),sqrt(m)]);
    beta=.9;
    HIO_init_iters=5e1;
    n_starts=50;
    resid_best=inf;
    t0=tic;
    x_init_best=nan(sqrt(n),sqrt(n));
    for j=1:n_starts
        x_init_i=HIO( y, M_square, Minv_square, support(:), beta,HIO_init_iters );
        resid_i=norm(y-abs(M_square(x_init_i)));
        if resid_i<resid_best
            resid_best=resid_i;
            x_init_best=x_init_i;
        end
    end
    HIO_iters=1e3;
    x_hat_HIO=HIO( y, M_square, Minv_square, support(:), beta,HIO_iters,x_init_best );
    t_HIO=toc(t0)
    x_hat_HIO = OversampM'*real(x_hat_HIO(:));
end

if isequal(lower(measurement_method),'fourier')
    x_init=x_hat_HIO;
    x_hat_HIO = disambig2Drfft(x_hat_HIO,x_0,sqrt(n),sqrt(n));
    x_hat_HIO = reshape(x_hat_HIO,[height,width]);
else
    x_init=ones(height,width);
end

%Set RED options
prox_opts=[];
prox_opts.width=width;
prox_opts.height=height;
prox_opts.denoiser=denoiser_RED;
prox_opts.prox_iters=1;
prox_opts.sigma_w=sigma_w;

fasta_opts=[];
fasta_opts.maxIters=200;
fasta_opts.tol=1e-7;
fasta_opts.recordObjective=true;%Record the value of objective function

% fasta_ops.function=errfxn;
% fasta_opts.adaptive = 0;
% fasta_opts.accelerate=0;
% fasta_opts.backtrack=0;


if  isequal(lower(measurement_method),'fourier')
    prox_opts.lambda=prox_opts.sigma_w;%Regularization term parameter
else
    prox_opts.lambda=.1;
end
t0=tic;
prox_opts.sigma_hat=90;%prox_opts.sigma_hat is noise level used in denoising
[x_hat_prDeep,outs_final1]  = my_PR( M,Mt_asym,y,x_init(:),fasta_opts,prox_opts);
prox_opts.sigma_hat=70;
[x_hat_prDeep,outs_final2]  = my_PR( M,Mt_asym,y,x_hat_prDeep(:),fasta_opts,prox_opts);
if alpha<50
    prox_opts.sigma_hat=50;
    [x_hat_prDeep,outs_final3]  = my_PR( M,Mt_asym,y,x_hat_prDeep(:),fasta_opts,prox_opts);
    prox_opts.sigma_hat=10;
    [x_hat_prDeep,outs_final4]  = my_PR( M,Mt_asym,y,x_hat_prDeep(:),fasta_opts,prox_opts);
end
t_prDeep_Amplitude=toc(t0)
x_hat_prDeep = real(reshape(x_hat_prDeep,[height,width]));
x_hat_prDeep = disambig2Drfft(x_hat_prDeep,x_0,sqrt(n),sqrt(n));
if fasta_opts.recordObjective
    if alpha<50
        %figure(5);plot([outs_final1.objective,outs_final2.objective,...
            %outs_final3.objective,outs_final4.objective]);title('prDeep Loss Function');
         figure(5);
         plot(outs_final1.objective);
         title('prDeep Loss Function1');
         figure(6);
         plot(outs_final2.objective);
         title('prDeep Loss Function2');
         figure(7);
         plot(outs_final3.objective);
         title('prDeep Loss Function3');
         figure(8);
         plot(outs_final4.objective);
         title('prDeep Loss Function4');
    else
        figure(5);plot([outs_final1.objective,outs_final2.objective]);title('prDeep Loss Function');
    end
end

resid_err=norm(y-abs(M(x_hat_prDeep)),'fro');
if resid_err<best_resid
    best_resid=resid_err;
    x_hat_prDeep_best=x_hat_prDeep;
end

end

if  isequal(lower(measurement_method),'fourier')
    perf_HIO=PSNR(x_0(:),x_hat_HIO(:));
    display([num2str(SamplingRate*100),'% Sampling HIO: PSNR=',num2str(perf_HIO), ', per trial time=',num2str(t_HIO)])
    figure(1);imshow(x_hat_HIO,[]);title('HIO');
end

perf_prDeep_Amp=PSNR(x_0,x_hat_prDeep_best);
display([num2str(SamplingRate*100),'% Sampling prDeep: PSNR=',num2str(perf_prDeep_Amp), ', per trial time=',num2str(t_prDeep_Amplitude)])
figure(2);imshow(x_hat_prDeep_best,[]);title('my reconstruction');
