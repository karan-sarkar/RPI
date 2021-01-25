function dataLL

% global poly interpolation: direct versus Lagrange
% example used is the 5th degree poly from data set
% demos problems arising from vandermonde matrix
% try nx = 5, 10, 40, 50

% icase = 1: direct (Vardermonde)   icase=2: Lagrange
icase=1

% 5th degree poly data
a=-1; b=1;
nx=50
xd=linspace(a,b,nx);
for iy=1:nx
    yd(iy)=(xd(iy)+0.9)*(xd(iy)+0.1)^2*(xd(iy)-0.2)*(xd(iy)-0.8);
end

n=400;
xp=linspace(a,b,n);

if icase==1
    say=['Direct (Vardermonde)'];
    aa=inv(vander(xd))*yd';
    for ii=1:n
        p(ii)=aa(nx);
        for ip=2:nx
            p(ii)=p(ii)+aa(nx-ip+1)*xp(ii)^(ip-1);
        end
    end
elseif icase==2  
    say=['Lagrange'];
    for ii=1:n
        p(ii)=0;
        for k=1:nx
            p(ii)=p(ii)+yd(k)*ell(k,xp(ii),xd);
        end
    end
end

clf
% get(gcf)
set(gcf,'Position', [4 1052 651 293])
hold on
box on
plot(xd,yd,'or','MarkerSize',7,'LineWidth',2)
plot(xp,p,'b','LineWidth',1.5)

text(-0.8,0.3,say,'FontSize',18,'FontWeight','bold')

axis([-1.1 1.1 -0.2 0.4])
grid on
xlabel('x-axis')
ylabel('y-axis')
set(gca,'FontSize',14,'FontWeight','bold')

% lagrange basis function
function p=ell(i,x,xd)
[n1 n2]=size(xd);
p=1;
for j=1:n2
    if j ~= i
        p=p*(x-xd(j))/(xd(i)-xd(j));
    end
end



