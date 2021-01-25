%  Example:
xd = [ 0 2 4 6 8 10 12 14 16 18 20 22 24];  yd = [59 56 53 54 60 67 72 74 75 74 70 65 61];
x = 0:0.05:24;  
y = nspline(xd,yd,x);
hold all
plot(xd,yd,'o')
plot(x,y,'-')
yl = lagrange(xd, yd, x);
plot(x,yl,'--')
legend('data', 'cubic spline', 'lagrange')
hold off
fprintf('11 AM Temp: Cubic Spline: %5.3f Lagrange %5.3f\n', nspline(xd, yd, 11), lagrange(xd, yd, 11))
fprintf('1 AM Temp: Cubic Spline: %5.3f Lagrange %5.3f\n', nspline(xd, yd, 25), lagrange(xd, yd, 25))
fprintf('9 AM Temp: Cubic Spline: %5.3f Lagrange %5.3f\n', nspline(xd, yd, 33), lagrange(xd, yd, 33))