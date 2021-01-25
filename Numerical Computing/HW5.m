

b = 4 * 10^(-6);
a = 0.01 * 10^(-6);

n = 1000000;
h=(b-a)/n;
xi=a:h:b;
I= h/3*(Ed(xi(1))+ 2*sum(Ed(xi(3:2:end-2)))+4*sum(Ed(xi(2:2:end)))+Ed(xi(end)));

fprintf("%1.6f", I)

plot(xi, Ed(xi))

function E = Ed(Lam)
    c=3*10^8; % speed of light in vaccum
    h=6.625*10.^-34; %  Planck constant 
    k=1.38*10.^-23; %   Boltzmann constant
    T = 7000;
    %E = ((exp(x.^(-1) * 2) - 1) .* x.^5).^(-1) * 8 * pi * h * c;
    E = (8*h*c*pi)./((Lam.^5).*(exp((h.*c)./(k.*T.*Lam))-1));
    
end