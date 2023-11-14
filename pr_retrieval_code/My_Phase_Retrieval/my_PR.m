%   Solve the problem
%           min 1/(2*sigma_w^2)*\|b-|Ax|\|^2+lambda/2x'*(x-f(x)) where f(x)
%           is a denoiser
%   using the solver FASTA.  
%
%  Inputs:
%    A   : A matrix or function handle
%    At  : The adjoint/transpose of A
%    b   : A column vector of measurements
%    x0  : Initial guess of solution, often just a vector of zeros
%    opts: Optional inputs to FASTA


function [ solution, outs ] = my_PR(A,At,b,x0,opts,prox_ops)


%%  Check whether we have function handles or matrices
if ~isnumeric(A)
    assert(~isnumeric(At),'If A is a function handle, then At must be a handle as well.')
end
%  If we have matrices, create handles just to keep things uniform below
if isnumeric(A)
    At = @(x)A'*x;
    A = @(x) A*x;
end

%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end

%  Note: fasta solves min f(Ax)+g(x).

%  f(z) = 1/(2*sigma_w^2)|| abs(z) - b||^2
f    = @(z) 1/(2*prox_ops.sigma_w^2)*norm(abs(z)-b,'fro')^2;
subgrad = @(z) 1/prox_ops.sigma_w^2*(z-b.*z./abs(z));

denoi = @(noisy) denoise(real(noisy),prox_ops.sigma_hat,prox_ops.width,prox_ops.height,prox_ops.denoiser);

g = @(x) prox_ops.lambda/2*real(x)'*(real(x)-denoi(x));%Red regularizer
%g = @(x) prox_ops.lambda/2*norm(real(x)-denoi(x))^2;%L_2 regularization by
                                                      %denoising 

% proxg(z,t) = argmin 0.5||x-z||^2+t*g(x)
prox = @(z,t) iterative_prox_map(z,t,denoi,prox_ops);

%% Call solver
[solution, outs] = my_fasta_PR(A,At,f,subgrad,g,prox,x0,opts,b, prox_ops);

end

function x = iterative_prox_map(z,t,denoi,opts)
    lambda=opts.lambda;
	x=z;
    prox_iters=opts.prox_iters;
    for iters=1:prox_iters
        x=(1/(1+t*lambda))*(z+t*lambda*denoi(x));
        %x=(1/(1+t*lambda))*(z+t*lambda*(2*denoi(x)-denoi(x).^2./x));%L_2 regularization by
                                                      %denoising 
         

    end
end