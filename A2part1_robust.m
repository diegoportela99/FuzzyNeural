clear
clc
cycleerr =[];

test_character = [
    1 1 1 -1 -1 1 1 -1 -1 1 -1 -1 -1 1 1 -1 -1;
];

x = [
    [1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1];
    [1 1 1 -1 1 -1 -1 -1 1 1 1 -1 1 1 1 -1 -1];
    [1 1 1 -1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 -1];
    [-1 1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1];
    [-1 1 1 1 -1 1 -1 -1 -1 1 1 1 -1 1 1 1 -1];
    [-1 1 1 1 -1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1];
];

d = [
    1 -1 -1;
    -1 1 -1;
    -1 -1 1;
    1 -1 -1;
    -1 1 -1;
    -1 -1 1;
];

w = [
    0.2007 -0.0280 -0.1871;
    0.5522 0.2678 -0.7830;
    0.4130 -0.5299 0.6420;
];  % The weight of hidden layer

wp = [
    [-0.2206 0.2139 0.4764 -0.1886 0.5775 -0.7873 -0.2943 0.9803 -0.5945 -0.2076 0.1932 0.8436 -0.6475 0.3365 0.1795 -0.0542 0.6263;]
    [-0.7222 -0.6026 0.3556 -0.6024 0.7611 0.9635 -0.1627 -0.0503 0.3443 -0.4812 -0.9695 -0.2030 -0.0680 0.6924 0.5947 0.6762 0.2222;]
]; %The weight of input layer

I = 3;
J = 3;
K = 3;
n = 0.25;

for j = 1:200  % The number of cycle
    e=0;
        for i = 1:6
            td = x(i,:)';
            
            vp = wp*td;
            y = (1-exp(-vp))./(1+exp(-vp)); % The output of the hidden layer
            dy = 0.5*(1-y.^2); %f'(bar(v))
            v = w*[y; -1];
            z =(1-exp(-v))./(1+exp(-v));
            dz = 0.5*(1-z.^2); %f(v')
            r = (d(i,:)'-z);
            delta = r.*dz; % The error signal of hidden layer
            deltap = dy.*(w(:,1:J-1)'*delta); % The error signal of the output layer
            deltaw = n*delta*[y; -1]';
            deltawp = n*deltap*x(i,:);
            w = w + deltaw;
            wp = wp+deltawp;
            er = r.^2;
            e = e+ 0.5*sum(er); % cycle error
        end
    cycleerr = [cycleerr e];  % All the cycle errors
end

disp('final weights between output and hidden layer')
w
disp('final weights between hidden and input layer')
wp
plot(cycleerr)

%test character output
a = test_character(1,:)';

vp = wp*a;
y = (1-exp(-vp))./(1+exp(-vp)); % The output of the hidden layer
dy = 0.5*(1-y.^2); %f'(bar(v))
v = w*[y; -1];
z =(1-exp(-v))./(1+exp(-v))
