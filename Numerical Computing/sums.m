function sums

fprintf('\n Sums Example \n')

% compute partial sum of harmonic series (n terms)

for ic=1:8
    n=10^ic;
    
    %%%% from small to large
    s=0;
    for j=1:n
        s=s+1/(n-j+1);
    end
    
    %%%% from large to small
    S=0;
    for jj=1:n
        S=S+1/jj;
    end

    fprintf('\n n = %d     S-s = %5.1e \n',n,S-s)
    
    pause
end
fprintf('\n')
