%Training data
y = [0.4 0.8 0.2 0.3 0.2 1;
    1.0 0.2 0.5 0.7 -0.5 1;
    -0.1 0.7 0.3 0.3 0.9 1;
    0.2 0.7 -0.8 0.9 0.3 1;
    0.1 0.3 1.5 0.9 1.2 1];

%desired output
d = [1 -1 1 -1 1];

%starting weights, last variable is the bias for activation function
w = [0.3350;0.1723;-0.2102;0.2528;-0.1133;0.5012];

%learning rate
c = 2;
%Pattern error and Cycle Error
P = [];
C = [];

%training...
disp('___________________________________________________________')
for o=1:10
for a=1:5

    x = y(a,:)';
    g = d(a);

    %sign(x) is the TLU function
    v = w'*x;
    z = sign(v);
    e = g-z;
    r = e;
    
    w = w+c*r*x;
    
    %Pattern error
    E = (1/2)*(g - z)^2;
    P = [P; E];
    
    %used for cycle error
    S = S + E;
    
end
    %Cycle error
    C = [C; S];
    S = 0; %reset Sum
end

fprintf('Pattern error is as follows (50 steps)\n')
figure(1)
%plot(P(1:50))
P=P

fprintf('Cycle error is as follows (10 Cycles)\n')
C=C
figure(2)
%plot(C(1:10))

fprintf('final weight vector is as follows\n')
fprintf('W(1) = %5.2f\n', w(1))
fprintf('W(2) = %5.2f\n', w(2))
fprintf('W(3) = %5.2f\n', w(3))
fprintf('W(4) = %5.2f\n', w(4))
fprintf('W(5) = %5.2f\n\n', w(5))
fprintf('W(6) = %5.2f\n\n', w(6))

for u=1:5
    x = y(u,:)';
    v = w'*x;
    z = sign(v);
    fprintf('x = %5.2f %5.2f %5.2f %5.2f %5.2f || desired  output is %5.2f || Actual output: %5.2f\n', x(1), x(2), x(3), x(4), x(5), d(u), z);
end


%Assignment 1.2

clear 
disp(' ')
disp('Part 2 ---------------------------------')
%Learning rate
c = 0.2;

%Pattern error and Cycle Error
P = [];
C = [];
S = 0;

%Training data
y = [0.4 0.8 0.2 0.3 0.2 1;
    1.0 0.2 0.5 0.7 -0.5 1;
    -0.1 0.7 0.3 0.3 0.9 1;
    0.2 0.7 -0.8 0.9 0.3 1;
    0.1 0.3 1.5 0.9 1.2 1];

%desired output
d = [1 -1 1 -1 1];

%starting weights, last variable is the bias for activation function
w = [0.3350;0.1723;-0.2102;0.2528;-0.1133;0.5012];
T = []

for n=(1:50)
for a=(1:5)
   x = y(a,:)';
   g = d(a);
   
   v = w'*x;
   z = (1/(1+exp(-v)));
   e = d(a)-z;
   df = exp(v)/(exp(v)+1)^2;
   r = e*df;
   
   if (n==2)
       if (a==2)
        T = [T;z]
       end
    end
   
   w = w + c*r*x;
   
   
   %Pattern error
    E = (1/2)*(g - z)^2;
    P = [P; E];
    
    %used for cycle error
    S = S + E;
end
    %Cycle error
    C = [C; S];
    S = 0; %reset Sum
end

T = [T;z]

fprintf('final weight vector is as follows\n')
fprintf('W(1) = %5.2f\n', w(1))
fprintf('W(2) = %5.2f\n', w(2))
fprintf('W(3) = %5.2f\n', w(3))
fprintf('W(4) = %5.2f\n', w(4))
fprintf('W(5) = %5.2f\n\n', w(5))
fprintf('W(6) = %5.2f\n\n', w(6))

for u=1:5
    x = y(u,:)';
    v = w'*x;
    z = (1/(1+exp(-v)));
    fprintf('x = %5.2f %5.2f %5.2f %5.2f %5.2f || desired  output is %5.2f || Actual output: %5.2f\n', x(1), x(2), x(3), x(4), x(5), d(u), z);
end

figure(1)
plot(P(1:250))

figure(2)
plot(C(1:50))