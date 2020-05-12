%training inputs
x = [
    1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1;
    1 1 1 -1 1 -1 -1 -1 1 1 1 -1 1 1 1 -1;
    1 1 1 -1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1;
    -1 1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 -1;
    -1 1 1 1 -1 1 -1 -1 -1 1 1 1 -1 1 1 1;
    -1 1 1 1 -1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1
];

%desired from training
d = [
    1 -1 -1;
    -1 1 -1;
    -1 -1 1;
    1 -1 -1;
    -1 1 -1;
    -1 -1 1;
];

weight_vector_to_output = [
    
];

weight_vector_to_hidden = [
    
];


%--------Variables--------

C = 200; %total cycles
n = 0.25; %learning constant

%neural network design is as follows...
%16 Inputs (1 bias), 2 Hidden (1 bias), 3 outputs
%example of data (4x4):
%[ ][X][X][X]
%[ ][ ][X][ ]
%[ ][ ][X][ ]
%[ ][ ][X][ ]

%neural network


