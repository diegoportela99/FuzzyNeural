%training inputs
x = [
    1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1;
    1 1 1 -1 1 -1 -1 -1 1 1 1 -1 1 1 1 -1 -1;
    1 1 1 -1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 -1;
    -1 1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1;
    -1 1 1 1 -1 1 -1 -1 -1 1 1 1 -1 1 1 1 -1;
    -1 1 1 1 -1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1;
];

test_character = [
    1 1 1 -1 -1 1 1 -1 -1 1 -1 -1 -1 1 1 -1;
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
    0.2007 -0.0280 -0.1871;
    0.5522 0.2678 -0.7830;
    0.4130 -0.5299 0.6420;
];

weight_vector_to_hidden = [
    -0.2206 0.2139 0.4764 -0.1886 0.5775 -0.7873 -0.2943 0.9803 -0.5945 -0.2076 0.1932 0.8436 -0.6475 0.3365 0.1795 -0.0542 0.6263;
    -0.7222 -0.6026 0.3556 -0.6024 0.7611 0.9635 -0.1627 -0.0503 0.3443 -0.4812 -0.9695 -0.2030 -0.0680 0.6924 0.5947 0.6762 0.2222;
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

for a = (1:C)
   for b = (1:6)
       training_data = x(b,:)';
       desired = d(b,:)';
       
       %2 neurons from first layer
       first_neuron_weight = weight_vector_to_hidden(1,:)';
       second_neuron_weight = weight_vector_to_hidden(2,:)';
       
       %matrix multiplication hidden
       first_vector = first_neuron_weight'*training_data;
       second_vector = second_neuron_weight'*training_data;
       
       %activation function hidden
       first_activation = (1/(1+exp(-first_vector)));
       second_activation = (1/(1+exp(-second_vector)));
       third_activation = -1; %bias
       
       tied_up_activation = [
           first_activation; second_activation; third_activation;
       ];
       
       %output layer
       first_output_neuron_weights = weight_vector_to_output(1,:)';
       second_output_neuron_weights = weight_vector_to_output(2,:)';
       third_output_neuron_weights = weight_vector_to_output(3,:)';
       
       tied_up_output_weights = [
           first_output_neuron_weights; second_output_neuron_weights; third_output_neuron_weights;
       ];
       
       %matrix multiplication output
       first_output_vector = first_output_neuron_weights'*tied_up_activation;
       second_output_vector = second_output_neuron_weights'*tied_up_activation;
       third_output_vector = third_output_neuron_weights'*tied_up_activation;
       
       tied_up_output_matrix = [
           first_output_vector; second_output_vector; third_output_vector;
       ];
       
       %activation function output
       first_output_activation = (1/(1+exp(-first_output_vector)));
       second_output_activation = (1/(1+exp(-second_output_vector)));
       third_output_activation = (1/(1+exp(-third_output_vector)));
       
       %output / desired
       first_desired_neuron = desired(1) - first_output_activation
       second_desired_neuron = desired(2) - second_output_activation
       third_desired_neuron = desired(3) - third_output_activation
       
       %back-propagation output layer
       first_function_derivative_output = exp(first_output_vector)/(exp(first_output_vector)+1)^2;
       second_function_derivative_output = exp(first_output_vector)/(exp(first_output_vector)+1)^2;
       third_function_derivative_output = exp(first_output_vector)/(exp(first_output_vector)+1)^2;
       
       %output error signal
       first_error_signal_output = first_desired_neuron*first_function_derivative_output;
       second_error_signal_output = second_desired_neuron*second_function_derivative_output;
       third_error_signal_output = third_desired_neuron*third_function_derivative_output;
       
       tied_up_output_error_signal = [
           first_error_signal_output; second_error_signal_output; third_error_signal_output;
       ];
   
       %hidden layer derivative function
       first_function_derivative_hidden = exp(first_vector)/(exp(first_vector)+1)^2;
       second_function_derivative_hidden = exp(second_vector)/(exp(second_vector)+1)^2;
       third_function_derivative_hidden = exp(-1)/(exp(-1)+1)^2;
       
       %hidden error signal
       first_error_signal_hidden = tied_up_output_weights(1)*tied_up_output_error_signal*first_function_derivative_hidden;
       second_error_signal_hidden = tied_up_output_weights(2)*tied_up_output_error_signal*second_function_derivative_hidden;
       %3rd neuron is bias function
       third_error_signal_hidden = tied_up_output_weights(3)*tied_up_output_error_signal*third_function_derivative_hidden;
       
   end
end
