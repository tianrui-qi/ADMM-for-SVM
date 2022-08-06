%% classification on ranData_rho08

load data/ranData_rho08;

%% basic parameter

[p,N] = size(Xtrain);

w_init = randn(p,1);
b_init = 0;
t_init = zeros(N,1);

%% Testing by student code
fprintf('Testing by student code\n\n');

% parameters
lam = 0.1;
opts = [];
opts.tol = 1e-3;
opts.maxit = 5000;
opts.subtol = 1e-3;
opts.maxsubit = 10000;
opts.beta = 1;
opts.w0 = w_init;
opts.b0 = b_init;
opts.t0 = t_init;

% train
t0 = tic;
[w_s,b_s,out_s] = ADMM_SVM(Xtrain, ytrain, lam, opts);
time = toc(t0);

% do classification on the testing data
pred_y = sign(Xtest'*w_s + b_s);
accu = sum(pred_y==ytest)/length(ytest);

% print results
fprintf('Running time is %5.4f\n',time);
fprintf('classification accuracy on testing data: %4.2f%%\n\n',accu*100);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(out_s.hist_pres,'b-','linewidth',2);
hold on
semilogy(out_s.hist_dres,'r-','linewidth',2);
legend('Primal residual','dual residual','location','best');
xlabel('outer iteration');
ylabel('error');
title('student: ranData\_rho08');
set(gca,'fontsize',14);
hold off;
print(fig,'-dpdf','ranData_rho08_student');


%% Testing by instructor code
fprintf('Testing by instructor code\n\n');

% parameters
lam = 0.1;
opts = [];
opts.tol = 1e-3;
opts.maxit = 10000;
opts.subtol = 1e-4;
opts.maxsubit = 100;
opts.beta = 1;
opts.w0 = w_init;
opts.b0 = b_init;
opts.t0 = t_init;

% train
t0 = tic;
[w_p,b_p,out_p] = ALM_SVM_p(Xtrain,ytrain,lam,opts);
time = toc(t0);

% do classification on the testing data
pred_y = sign(Xtest'*w_p + b_p);
accu = sum(pred_y==ytest)/length(ytest);

% print results
fprintf('Running time is %5.4f\n',time);
fprintf('classification accuracy on testing data: %4.2f%%\n\n',accu*100);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(out_p.hist_pres,'b-','linewidth',2);
hold on;
semilogy(out_p.hist_dres,'r-','linewidth',2);
legend('Primal residual','dual residual','location','best');
xlabel('outer iteration');
ylabel('error');
title('instructor: ranData\_rho08');
set(gca,'fontsize',14);
hold off;
print(fig,'-dpdf','ranData_rho08_instructor');