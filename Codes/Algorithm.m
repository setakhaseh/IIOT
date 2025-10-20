clear; clc; close all;

%% Parameters
rng(0); % For reproducibility
T = 100; % Total time steps
h = 0.2;

% System matrices (time-varying)
A_k = @(k) [
exp(-h + sin(k*h) - sin((k-1)*h)), 0, 0;
2*sinh(h/2)*exp(-3*h/2 + sin(k*h) - sin((k-1)*h)), exp(-2*h + sin(k*h) - sin((k-1)*h)), 0;
0, 0, exp(-2*h + sin(k*h) - sin((k-1)*h))
];

G_k = [0.1, 0, 0.2; 0.2, 0.1, 0.3]'; % 3x2 matrix
H_k = [0.1, 0.2, 0.1; 0.3, 0.6, 0]'; % 3x2 matrix
C_k = @(k) [
cos(k*h), sin(k*h), 1.5;
1, sin(2*k*h), cos(2*k*h);
0, sin(3*k*h), 2
];

Q_k = [0.4, 0.2, 0.1; 0.2, 0.5, 0.3; 0.1, 0.3, 0.5];
R_k = [0.1, 0.03, 0.05; 0.03, 0.1, 0.02; 0.05, 0.02, 0.1];
Y_k = [0.05, 0.02, 0; 0.02, 0.05, 0.01; 0, 0.01, 0.1];

D_k = [1, 0]; % 1x2 matrix
E_k = [1, 0]; % 1x2 matrix

% Initial state
x0 = [0; 0; 0];
P0 = eye(3);

% Preallocate
x_true = zeros(3, T);
y_meas = zeros(3, T);
d_attack = zeros(2, T);
e_attack = zeros(2, T);
r_obs = zeros(1, T);
q_obs = zeros(1, T);
gamma_seq = zeros(1, T);

% Generate attacks
for k = 1:T
d_attack(:,k) = 0.5*sin(0.1*k) + 0.1*randn(2,1);
e_attack(:,k) = 0.3*cos(0.2*k) + 0.05*randn(2,1);
r_obs(k) = D_k * d_attack(:,k);
q_obs(k) = E_k * e_attack(:,k);
end

%% Generate true state and measurements
x_true(:,1) = x0 + sqrtm(P0)*randn(3,1);
for k = 1:T-1
w = mvnrnd(zeros(3,1), Q_k)';
x_true(:,k+1) = A_k(k)*x_true(:,k) + G_k*d_attack(:,k) + w;
end

for k = 1:T
v = mvnrnd(zeros(3,1), R_k)';
y_meas(:,k) = C_k(k)*x_true(:,k) + H_k*e_attack(:,k) + v;
end

%% Event-triggered transmission
for k = 1:T
phi_k = exp(-0.5 * y_meas(:,k)' * Y_k * y_meas(:,k));

if rand <= phi_k
gamma_seq(k) = 0;  % Do not transmit
else
gamma_seq(k) = 1;  % Transmit
end
end
comm_rate = mean(gamma_seq);
fprintf('Communication rate: %.2f\n', comm_rate);

%% Event-Triggered MMSE Estimation (Simplified version)
x_est = zeros(3, T);
P_est = zeros(3,3,T);

x_est(:,1) = x0;
P_est(:,:,1) = P0;

for k = 2:T
% --- Prediction Step ---
x_pred = A_k(k-1) * x_est(:,k-1);
P_pred = A_k(k-1) * P_est(:,:,k-1) * A_k(k-1)' + Q_k;

% --- Calculate Partially Observed Attacks ---
r_hat_prev = D_k * d_attack(:,k-1);  % \hat{r}_{k-1}
q_hat_k = E_k * e_attack(:,k);  % \hat{q}_k

% --- Event-triggered Update ---
if gamma_seq(k) == 1
% Measurement update when data is transmitted
% Calculate innovation covariance
S = C_k(k) * P_pred * C_k(k)' + R_k;

% Calculate Kalman gain
K = P_pred * C_k(k)' / S;

% Update state estimate (compensating for known attack)
innovation = y_meas(:,k) - C_k(k)*x_pred - H_k*e_attack(:,k);
x_est(:,k) = x_pred + K * innovation;

% Update covariance
P_est(:,:,k) = (eye(3) - K*C_k(k)) * P_pred;

% Add known attack information (simplified approach)
% This is a simplified way to incorporate r_hat_prev
if ~isempty(r_hat_prev) && ~isnan(r_hat_prev)
x_est(:,k) = x_est(:,k) + 0.1 * r_hat_prev * ones(3,1);
end

else
% No measurement update - keep prediction
x_est(:,k) = x_pred;
P_est(:,:,k) = P_pred;

% Still incorporate known process attack information
if ~isempty(r_hat_prev) && ~isnan(r_hat_prev)
x_est(:,k) = x_est(:,k) + 0.1 * r_hat_prev * ones(3,1);
end
end
end

%% Plot results
figure;
subplot(3,1,1);
plot(1:T, x_true(1,:), 'b', 'LineWidth', 1.5);
hold on;
plot(1:T, x_est(1,:), 'r--', 'LineWidth', 1.5);
legend('True', 'Estimated');
title('State 1 Estimation');
ylabel('Value');
grid on;

subplot(3,1,2);
plot(1:T, x_true(2,:), 'b', 'LineWidth', 1.5);
hold on;
plot(1:T, x_est(2,:), 'r--', 'LineWidth', 1.5);
legend('True', 'Estimated');
title('State 2 Estimation');
ylabel('Value');
grid on;

subplot(3,1,3);
plot(1:T, x_true(3,:), 'b', 'LineWidth', 1.5);
hold on;
plot(1:T, x_est(3,:), 'r--', 'LineWidth', 1.5);
legend('True', 'Estimated');
title('State 3 Estimation');
xlabel('Time step');
ylabel('Value');
grid on;

figure;
stem(1:T, gamma_seq);
title('Event-Triggered Transmission Sequence');
xlabel('Time step');
ylabel('Transmission (1: yes, 0: no)');
ylim([-0.1 1.1]);
grid on;

figure;
subplot(2,2,1);
plot(1:T, d_attack(1,:), 'k', 'LineWidth', 1.5);
title('Attack d_1');
xlabel('Time step');
ylabel('Value');
grid on;

subplot(2,2,2);
plot(1:T, d_attack(2,:), 'k', 'LineWidth', 1.5);

title('Attack d_2');
xlabel('Time step');
ylabel('Value');
grid on;

subplot(2,2,3);
plot(1:T, e_attack(1,:), 'k', 'LineWidth', 1.5);
title('Attack e_1');
xlabel('Time step');
ylabel('Value');
grid on;

subplot(2,2,4);
plot(1:T, e_attack(2,:), 'k', 'LineWidth', 1.5);
title('Attack e_2');
xlabel('Time step');
ylabel('Value');
grid on;

% Calculate and display estimation error
estimation_error = mean(sqrt(mean((x_true - x_est).^2, 1)));
fprintf('Average estimation error: %.4f\n', estimation_error);
fprintf('Final state values - True: [%.3f, %.3f, %.3f], Estimated: [%.3f, %.3f, %.3f]\n', ...
x_true(1,end), x_true(2,end), x_true(3,end), ...
x_est(1,end), x_est(2,end), x_est(3,end));clear; clc; close all;

%% Parameters
rng(0); % For reproducibility
T = 100; % Total time steps
h = 0.2;

% System matrices (time-varying)
A_k = @(k) [
exp(-h + sin(k*h) - sin((k-1)*h)), 0, 0;
2*sinh(h/2)*exp(-3*h/2 + sin(k*h) - sin((k-1)*h)), exp(-2*h + sin(k*h) - sin((k-1)*h)), 0;
0, 0, exp(-2*h + sin(k*h) - sin((k-1)*h))
];

G_k = [0.1, 0, 0.2; 0.2, 0.1, 0.3]'; % 3x2 matrix
H_k = [0.1, 0.2, 0.1; 0.3, 0.6, 0]'; % 3x2 matrix
C_k = @(k) [
cos(k*h), sin(k*h), 1.5;
1, sin(2*k*h), cos(2*k*h);
0, sin(3*k*h), 2
];

Q_k = [0.4, 0.2, 0.1; 0.2, 0.5, 0.3; 0.1, 0.3, 0.5];
R_k = [0.1, 0.03, 0.05; 0.03, 0.1, 0.02; 0.05, 0.02, 0.1];
Y_k = [0.05, 0.02, 0; 0.02, 0.05, 0.01; 0, 0.01, 0.1];

D_k = [1, 0]; % 1x2 matrix
E_k = [1, 0]; % 1x2 matrix

% Initial state
x0 = [0; 0; 0];
P0 = eye(3);

% Preallocate
x_true = zeros(3, T);
y_meas = zeros(3, T);
d_attack = zeros(2, T);
e_attack = zeros(2, T);
r_obs = zeros(1, T);
q_obs = zeros(1, T);
gamma_seq = zeros(1, T);

% Generate attacks
for k = 1:T
d_attack(:,k) = 0.5*sin(0.1*k) + 0.1*randn(2,1);
e_attack(:,k) = 0.3*cos(0.2*k) + 0.05*randn(2,1);
r_obs(k) = D_k * d_attack(:,k);
q_obs(k) = E_k * e_attack(:,k);
end

%% Generate true state and measurements
x_true(:,1) = x0 + sqrtm(P0)*randn(3,1);
for k = 1:T-1
w = mvnrnd(zeros(3,1), Q_k)';
x_true(:,k+1) = A_k(k)*x_true(:,k) + G_k*d_attack(:,k) + w;
end

for k = 1:T
v = mvnrnd(zeros(3,1), R_k)';
y_meas(:,k) = C_k(k)*x_true(:,k) + H_k*e_attack(:,k) + v;
end

%% Event-triggered transmission
for k = 1:T
phi_k = exp(-0.5 * y_meas(:,k)' * Y_k * y_meas(:,k));

if rand <= phi_k
gamma_seq(k) = 0;  % Do not transmit
else
gamma_seq(k) = 1;  % Transmit
end
end
comm_rate = mean(gamma_seq);
fprintf('Communication rate: %.2f\n', comm_rate);

%% Event-Triggered MMSE Estimation (Simplified version)
x_est = zeros(3, T);
P_est = zeros(3,3,T);

x_est(:,1) = x0;
P_est(:,:,1) = P0;

for k = 2:T
% --- Prediction Step ---
x_pred = A_k(k-1) * x_est(:,k-1);
P_pred = A_k(k-1) * P_est(:,:,k-1) * A_k(k-1)' + Q_k;

% --- Calculate Partially Observed Attacks ---
r_hat_prev = D_k * d_attack(:,k-1);  % \hat{r}_{k-1}
q_hat_k = E_k * e_attack(:,k);  % \hat{q}_k

% --- Event-triggered Update ---
if gamma_seq(k) == 1
% Measurement update when data is transmitted
% Calculate innovation covariance
S = C_k(k) * P_pred * C_k(k)' + R_k;

% Calculate Kalman gain
K = P_pred * C_k(k)' / S;

% Update state estimate (compensating for known attack)
innovation = y_meas(:,k) - C_k(k)*x_pred - H_k*e_attack(:,k);
x_est(:,k) = x_pred + K * innovation;

% Update covariance
P_est(:,:,k) = (eye(3) - K*C_k(k)) * P_pred;

% Add known attack information (simplified approach)
% This is a simplified way to incorporate r_hat_prev
if ~isempty(r_hat_prev) && ~isnan(r_hat_prev)
x_est(:,k) = x_est(:,k) + 0.1 * r_hat_prev * ones(3,1);
end

else
% No measurement update - keep prediction
x_est(:,k) = x_pred;
P_est(:,:,k) = P_pred;

% Still incorporate known process attack information
if ~isempty(r_hat_prev) && ~isnan(r_hat_prev)
x_est(:,k) = x_est(:,k) + 0.1 * r_hat_prev * ones(3,1);
end
end
end

%% Plot results
figure;
subplot(3,1,1);
plot(1:T, x_true(1,:), 'b', 'LineWidth', 1.5);
hold on;
plot(1:T, x_est(1,:), 'r--', 'LineWidth', 1.5);
legend('True', 'Estimated');
title('State 1 Estimation');
ylabel('Value');
grid on;

subplot(3,1,2);
plot(1:T, x_true(2,:), 'b', 'LineWidth', 1.5);
hold on;
plot(1:T, x_est(2,:), 'r--', 'LineWidth', 1.5);

legend('True', 'Estimated');
title('State 2 Estimation');
ylabel('Value');
grid on;

subplot(3,1,3);
plot(1:T, x_true(3,:), 'b', 'LineWidth', 1.5);
hold on;
plot(1:T, x_est(3,:), 'r--', 'LineWidth', 1.5);
legend('True', 'Estimated');
title('State 3 Estimation');
xlabel('Time step');
ylabel('Value');
grid on;

figure;
stem(1:T, gamma_seq);
title('Event-Triggered Transmission Sequence');
xlabel('Time step');
ylabel('Transmission (1: yes, 0: no)');
ylim([-0.1 1.1]);
grid on;

figure;
subplot(2,2,1);
plot(1:T, d_attack(1,:), 'k', 'LineWidth', 1.5);
title('Attack d_1');
xlabel('Time step');
ylabel('Value');
grid on;

subplot(2,2,2);
plot(1:T, d_attack(2,:), 'k', 'LineWidth', 1.5);
title('Attack d_2');
xlabel('Time step');
ylabel('Value');
grid on;

subplot(2,2,3);
plot(1:T, e_attack(1,:), 'k', 'LineWidth', 1.5);
title('Attack e_1');
xlabel('Time step');
ylabel('Value');
grid on;

subplot(2,2,4);
plot(1:T, e_attack(2,:), 'k', 'LineWidth', 1.5);
title('Attack e_2');
xlabel('Time step');
ylabel('Value');
grid on;

% Calculate and display estimation error
estimation_error = mean(sqrt(mean((x_true - x_est).^2, 1)));
fprintf('Average estimation error: %.4f\n', estimation_error);
fprintf('Final state values - True: [%.3f, %.3f, %.3f], Estimated: [%.3f, %.3f, %.3f]\n', ...
x_true(1,end), x_true(2,end), x_true(3,end), ...
x_est(1,end), x_est(2,end), x_est(3,end));