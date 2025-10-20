clc; clear; close all;

%% ---------- Parameters ----------
rng(42);
T = 200;
n = 3;  % state dimension
m = 3;  % sensor dimension

% System matrices
A_k = @(k) [1 0.1 0; 0 1 0.1; 0 0 1];
C = randn(m,n);
Q = 0.01*eye(n);
R_nominal = 0.05*eye(m);

% Initial state
x0 = [0;0;1];
P0 = eye(n);

%% ---------- Preallocate ----------
x_true = zeros(n,T);
y_meas = zeros(m,T);
gamma_seq = zeros(1,T);

x_est_nn = zeros(n,T);       % Estimated with NN
P_est_nn = zeros(n,n,T);

x_est_noNN = zeros(n,T);     % Estimated without NN
P_est_noNN = zeros(n,n,T);

% Attack parameters
attack_sensors = [2];
attack_bias = 2;

x_true(:,1) = x0;
x_est_nn(:,1) = x0;
P_est_nn(:,:,1) = P0;
x_est_noNN(:,1) = x0;
P_est_noNN(:,:,1) = P0;

%% ---------- Generate true state and measurements ----------
for k = 2:T
    w = mvnrnd(zeros(n,1),Q)';
    v = mvnrnd(zeros(m,1),R_nominal)';
    x_true(:,k) = A_k(k-1)*x_true(:,k-1) + w;
    y_meas(:,k) = C*x_true(:,k) + v;
    y_meas(attack_sensors,k) = y_meas(attack_sensors,k) + attack_bias;
end

%% ---------- Network for attack detection ----------
hiddenLayerSize = [20 10]; 
net = feedforwardnet(hiddenLayerSize,'trainlm');
for i=1:length(hiddenLayerSize)
    net.layers{i}.transferFcn = 'tansig';
end
net.layers{end}.transferFcn = 'logsig';

% Prepare training data
X_train = []; Y_train = [];
for k=2:T
    x_pred = A_k(k-1)*x_est_nn(:,k-1);
    r_tilde = y_meas(:,k) - C*x_pred;
    X_train = [X_train r_tilde];
    target = ones(m,1);
    target(attack_sensors) = 0;
    Y_train = [Y_train target];
end
net.trainParam.epochs = 500;
net.trainParam.showWindow = false;
net = train(net,X_train,Y_train);

%% ---------- Event-triggered MMSE Kalman with and without NN ----------
epsilon = 1e-6;
lambda = 0.3;  
r_EWMA = zeros(m,1);

for k = 2:T
    % --------- Prediction ----------
    x_pred_nn = A_k(k-1)*x_est_nn(:,k-1);
    P_pred_nn = A_k(k-1)*P_est_nn(:,:,k-1)*A_k(k-1)' + Q;

    x_pred_noNN = A_k(k-1)*x_est_noNN(:,k-1);
    P_pred_noNN = A_k(k-1)*P_est_noNN(:,:,k-1)*A_k(k-1)' + Q;

    % --------- Innovation ----------
    r_k = y_meas(:,k) - C*x_pred_nn;
    r_EWMA = lambda*r_EWMA + (1-lambda)*(r_k.^2);
    r_k_noNN = y_meas(:,k) - C*x_pred_noNN;

    % --------- Event-triggered ----------
    phi_k = exp(-0.5*r_k'*r_k); 
    if rand <= phi_k
        gamma_seq(k) = 0;
        x_est_nn(:,k) = x_pred_nn; 
        P_est_nn(:,:,k) = P_pred_nn;
        x_est_noNN(:,k) = x_pred_noNN;
        P_est_noNN(:,:,k) = P_pred_noNN;
        continue;
    else
        gamma_seq(k) = 1;
    end

    % --------- NN-based adaptive ----------
    attack_prob = net(r_k);
    attack_mask = 1 ./ (attack_prob + epsilon);
    R_adaptive = diag(R_nominal .* r_EWMA .* attack_mask);
    R_adaptive = max(min(R_adaptive, 10*R_nominal), 0.1*R_nominal);

    % --------- Kalman Gain ----------
    Theta_nn = C*P_pred_nn*C' + R_adaptive;
    K_nn = P_pred_nn*C'/Theta_nn;
    x_est_nn(:,k) = x_pred_nn + K_nn*r_k;
    P_est_nn(:,:,k) = (eye(n) - K_nn*C)*P_pred_nn;

    % --------- بدون NN ----------
    Theta_noNN = C*P_pred_noNN*C' + R_nominal;
    K_noNN = P_pred_noNN*C'/Theta_noNN;
    x_est_noNN(:,k) = x_pred_noNN + K_noNN*r_k_noNN;
    P_est_noNN(:,:,k) = (eye(n) - K_noNN*C)*P_pred_noNN;
end

%% ---------- Plot results ----------
figure;
for i=1:n
    subplot(n,1,i);
    plot(1:T,x_true(i,:),'k','LineWidth',2); hold on;
    plot(1:T,x_est_noNN(i,:),'r--','LineWidth',1.5);
    plot(1:T,x_est_nn(i,:),'b-.','LineWidth',1.5);
    legend('True State','Estimated No NN','Estimated NN');
    title(['State ' num2str(i)]);
    grid on;
end

figure;
stem(1:T,gamma_seq);
title('Event-Triggered Transmission Sequence');
xlabel('Time step'); ylabel('Transmission (1: yes, 0: no)');
ylim([-0.1 1.1]); grid on;
