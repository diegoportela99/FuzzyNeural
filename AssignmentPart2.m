%Fuzzy logic

X = [0.2 0.5 0.8 1.0 0.6 0.1]
Y = [0.3 0.6 0.9 1.0 0.6 0.3]

R = [];

for a=(1:length(X))
   for b=(1:length(Y))
       R=[R;min(X(a), Y(b))];
   end
end

fprintf('Cartesian product R = \n')
R = reshape(R, length(X), length(Y))


Z = [0.3 0.6 0.7 0.9 1 0.5]

n = []; %used for processing
S = []; %composition matrix

for a=(1:length(Z))
   for b=(1:length(R))
       n=[n;min(Z(a), R(a, b))];
   end
   
   S = [S;max(n)]; %take max for that row
   n = []; %reset for next row
   
end

fprintf('Max-min composition\n')
S = S

fprintf('Max-product composition\n')

n = []; %used for processing
S = []; %composition matrix

for a=(1:length(Z))
   for b=(1:length(R))
       n=[n;Z(a)*R(b, a)];
   end
   
   S = [S;max(n)]; %take max for that row
   n = []; %reset for next row
   
end

S = S
