function newton0

%  Solve  f(x) = 0  using Newton's method

%  Input:
%	xa = starting point
%	tol = tolerance for stopping
%	f(x) and df(x) These are at end of file

xa=2;
tol=10^(-10);

% exact solution
xe=0.637;

fprintf('\n   Computed Solution    Iter. Error')

err=1;
it=0;
while err>tol

    xb=xa-f(xa)/df(xa);
    err=abs(xb-xa);
    xa=xb;
    it=it+1;
    fprintf('\n %d  %13.8e    %5.2e',it,xb,err)
    pause
end
fprintf('\n\n')

% plot error curve
clf
% get(gcf)
set(gcf,'Position', [1 925 560 420])

semilogy(iteration,error,'or','LineWidth',1.5,'MarkerSize',8)
hold on
semilogy(iteration,ierror,'sb','LineWidth',1.5,'MarkerSize',8)
legend({' Error',' Iterative Error'},'Location','NorthEast','FontSize',16)

xlabel('Iteration Step (n)')
ylabel('Error')
grid on
box on
set(gca,'FontSize',16,'FontWeight','bold')


function g=f(x)
% g=x*(x-2)*(x-4);
g=x + 2 * log10(0.01 + 0.0001 * x);

function g=df(x)
% g=(x-2)*(x-4)+x*(x-4)+x*(x-2);
g= 1 + 0.01 / (log(10) * (0.01 + 0.0001 * x));







