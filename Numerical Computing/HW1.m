%Exercise 1.8
%a
S = 0;
for i = 0:1000
    S = S + 1/(1 + exp(-1 * i));
end
fprintf('%d\n',S)
%e^1000 is larger than the largest available float. Thus, I divided both
%the numerator and denominator by e^x to ensure that the terms could be
%computed even for large values of i.

%b
S = 0;
for i = 0:1000
    S = S + (1 + exp(-2 * i))/(1 + 2 * exp(-1 * i) - exp(-2 * i));
end
fprintf('%d\n',S)
%cosh(1000) and sinh(1000) are larger than the largest available float. 
%Thus, I converted both to expoentiats and then divided both
%the numerator and denominator by e^x to ensure that the terms could be
%computed even for large values of i.

%c
S = 0;
for i = 0:1000
    S = S + (2 * exp(-0.5 * i))/(sqrt(3 * exp(-i) + 1) + sqrt(exp(-i) + 1));
end
fprintf('%d\n',S)
%First I converted the difference of sums into the sum of differences by
%merging the summations. Because, we were computing the difference of two
%rapidly growing quantities, I put the quantity in terms of the radical
%conjugate. Lastly, I replaced the quotient of positive exponents by a
%quotient of negative exponents.

%d
S = (exp(-1001) - 1) / (-(1001 + exp(1) / (1 - exp(1))) + (exp(1) / (1 - exp(1))) * exp(-1001));
fprintf('%d\n',S)
%First, I wrote the geometric series and arithmetico-geometric series in
%closed form. Then, I combined like terms and converted a quotient of
%positive exponentials to a quotient of negative exponentials.

%e
S = 0;
for i = 1:1000
    S = S +  2 * i * (-1)^i * sin(pi / i);
end
fprintf('%d\n',S)
%First, I applied the sum to product trig identity. Then, I ended up with a
%factor of cos(i^10). Because we are only considering integers, I converted
%this to (-1)^i which is much simpler to evaluate.

