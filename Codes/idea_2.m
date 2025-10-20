clc; clear; close all;

%% -----------------------------
% تنظیمات
%% -----------------------------
num_sensors = 10;
num_features = 3;   % سه ویژگی
num_attacked = 4;   % تعداد سنسورهای تحت حمله
num_clean = num_sensors - num_attacked;

rng(42); % بازتولید پذیری

%% -----------------------------
% داده‌های سالم (چگال، نزدیک هم)
%% -----------------------------
data_clean = normrnd(0, 0.05, [num_clean, num_features]);

%% -----------------------------
% داده‌های حمله شده (پرت و متفاوت)
%% -----------------------------
data_attacked = [];
for i = 1:num_attacked
    r = rand;
    if r < 0.4
        sample = normrnd(0.3, 0.05, [1, num_features]);  % نزدیک به سالم
    elseif r < 0.7
        sample = normrnd(1.0, 0.2, [1, num_features]);    % متوسط
    else
        sample = normrnd(2.0, 0.3, [1, num_features]);    % پرت
    end
    data_attacked = [data_attacked; sample];
end

%% -----------------------------
% ترکیب داده‌ها
%% -----------------------------
data = [data_clean; data_attacked];

%% -----------------------------
% نرمال‌سازی داده‌ها
%% -----------------------------
data_scaled = normalize(data);

%% -----------------------------
% خوشه‌بندی با K-means
%% -----------------------------
k = 3; % تعداد خوشه‌ها
[idx, C] = kmeans(data_scaled, k);

%% -----------------------------
% شناسایی خوشه تمیز (بزرگترین خوشه)
%% -----------------------------
cluster_sizes = histcounts(idx, 1:k+1);
[~, clean_cluster] = max(cluster_sizes);
clean_mask = (idx == clean_cluster);

%% -----------------------------
% رتبه‌بندی خوشه‌ها بر اساس فاصله تا خوشه تمیز
%% -----------------------------
dist_to_clean = sum((C - mean(data_scaled(clean_mask,:))).^2, 2);
[~, sorted_idx] = sort(dist_to_clean, 'descend');

cluster_names = cell(k,1);
for i = 1:k
    if sorted_idx(i) == clean_cluster
        cluster_names{sorted_idx(i)} = 'Clean / Low Attack';
    elseif i == 1
        cluster_names{sorted_idx(i)} = 'High Attack';
    else
        cluster_names{sorted_idx(i)} = 'Medium Attack';
    end
end

% نمایش نام خوشه‌ها
for i = 1:k
    fprintf('Cluster %d: %s\n', i, cluster_names{i});
end

%% -----------------------------
% استخراج داده‌های خوشه‌ها
%% -----------------------------
for i = 1:k
    fprintf('Cluster %d size: %d\n', i, sum(idx==i));
end

%% -----------------------------
% رسم نمودار
%% -----------------------------
figure; hold on;
colors = lines(k);
for i = 1:k
    scatter3(data_scaled(idx==i,1), data_scaled(idx==i,2), data_scaled(idx==i,3), 100, 'MarkerFaceColor', colors(i,:), 'DisplayName', cluster_names{i});
end
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Feature 3');
title('Sensor Data Clustering with Attack Ranking (3 Features)');
legend('Location','best');
grid on;
hold off;
