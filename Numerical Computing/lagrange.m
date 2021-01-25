function s = lagrange(xd,yd,x)
    s = 0;
    for k=1:length(xd)
        s=s+yd(k)*ell(k,x,xd);
    end
    
function p=ell(i,x,xd)
p=1;
for j=1:length(xd)
    if j ~= i
        p=p .*(x-xd(j))/(xd(i)-xd(j));
    end
end
