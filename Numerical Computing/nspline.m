function s = nspline(xd,yd,x)

%  natural cubic spline at equally spaced data points
%  xd, yd = data vectors
%  x = points where s(x) is evaluated

% some additional information is at the end of the file

nd=length(xd);
ndd=length(yd);
% errors
if nd < 2
    error('You need at least three data points');
end
if nd ~= ndd
    error('xd and yd must have the same length');
end

a=xd(1);
b=xd(nd);
np=length(x);

d=4*ones(nd-2,1);
dd=ones(nd-2,1);
for j=2:nd-1
    z(j-1)=6*yd(j);
end
z(1)=z(1)-yd(1);
z(nd-2)=z(nd-2)-yd(nd);
w=tri(d,dd,dd,z);
ww=[2*yd(1)-w(1), yd(1), w, yd(nd), 2*yd(nd)-w(nd-2)];

% evaluate interpolation function s(x)
h=xd(2)-xd(1);
for ix=1:np
    sum=0;
    for k=1:nd+2
        xk=a-h+h*(k-1);
        xx=(x(ix)-xk)/h;
        sum=sum+ww(k)*bbspline(xx);
    end
    s(ix)=sum;
end

% Calculate the value of a cubic B-spline at point x
function y=bbspline(x)
x=abs(x) ;
if x>2
    y=0 ;
else
    if x>1
        y=(2-x)^3/6 ;
    else
        y=2/3-x^2*(1-x/2) ;
    end
end


% tridiagonal solver
function y = tri( a, b, c, f )
N = length(f);
v = zeros(1,N);
y = v;
w = a(1);
y(1) = f(1)/w;
for i=2:N
    v(i-1) = c(i-1)/w;
    w = a(i) - b(i)*v(i-1);
    y(i) = ( f(i) - b(i)*y(i-1) )/w;
end
for j=N-1:-1:1
    y(j) = y(j) - v(j)*y(j+1);
end



%  the procedure uses cubic B-splines

%  Example:
%  xd = [ 1 2 3 4 5];  yd = [-1 2 -1 0 -1];
%  x = 1:0.05:5;  y = nspline(xd,yd,x);
%  plot(xd,yd,'o',x,y,'r')
%  nspline(xd,yd,3.1)

%  It has been tested on MATLAB, version R2010b and version R2012a
%  version: 1.0
%  March 5, 2013


















