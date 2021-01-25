function trap0

%  trapezoidal method for various M values
%  compares with exact solution
%  uses secant solutiuon of difference equation
%  y' = f(t,y)  with   y(0) = y0

global lambda

lambda=10;
y0=0.01;
tmax=1;

clf
% get(gcf)
set(gcf,'Position', [1 1078 573 267])

% calculate and plot exact solution
tt=linspace(0,tmax,100);
a0=(1-y0)/y0;
for it=1:100
    exact(it)=1/(1+a0*exp(-lambda*tt(it)));
end
plot(tt,exact,'r','LineWidth',1.8)
hold on
grid on
box on
xlabel('t-axis')
ylabel('Solution')
set(gca,'FontSize',16,'FontWeight','bold')
axis([0 1 0 1.05])

% compute numerical solution and plot
m=1;
for im=1:3
    m=4*m
    t=linspace(0,tmax,m+1);
    k=t(2)-t(1);
    
    % trap method
    y_trap=trap(t,y0,k,m+1);
    
    
    if im==1
        plot(t,y_trap,'--ks','MarkerSize',9,'LineWidth',1.3)
        legend({' Exact',' M = 4'},'Location','NorthWest','FontSize',16,'FontWeight','bold')
        pause
    elseif im==2
        plot(t,y_trap,'--m*','MarkerSize',9,'LineWidth',1.3)
        legend({' Exact',' M = 4',' M = 16'},'Location','NorthWest','FontSize',16,'FontWeight','bold')
        pause
    else
        plot(t,y_trap,'--bo','MarkerSize',9,'LineWidth',1.3)
        legend({' Exact',' M = 4',' M = 16',' M = 64 '},'Location','NorthWest','FontSize',16,'FontWeight','bold')
    end
    
    
end

% trap method
function ypoints=trap(t,y0,h,n)
tol=1e-6;
y=y0; fold=f(t(1),y);
ypoints=y0;
for i=2:n
    %  secant method (use taylor to estimate c)
    c=y+0.5*h*fold;
    yb=y;   fb=yb-0.5*h*f(t(i),yb)-c;
    yc=y+0.1*h*fold;   fc=yc-0.5*h*f(t(i),yc)-c;
    err=10*tol;
    while err>tol
        ya=yb; fa=fb;
        yb=yc; fb=fc;
        yc=yb-fb*(yb-ya)/(fb-fa);
        fc=yc-0.5*h*f(t(i),yc)-c;
        err=abs(1-yb/yc);
    end
    y=yc; fold=f(t(i),y);
    ypoints=[ypoints, y];
end

% right-hand side of DE
function z=f(t,y)
global lambda
z=lambda*y*(1-y);
