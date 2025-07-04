% Pfad zur .mat-Datei
matfile_path = 'C:\Users\anton\Documents\Studium\Bachelorarbeit\GPR_Daten_mat\radargrams.mat';

% .mat-Datei laden und in single konvertieren
load(matfile_path, 'radargrams');
data = single(radargrams{30});
clear radargrams

%data = data(:, 4000:6000);
[nt, nx] = size(data);
dt = single(1e-9);

% Parameter
env_size = 7;
window_size = 2 * env_size + 1;
block_size = 50;  % Kleinere Blöcke
num_blocks = ceil(nx/block_size);
n_features = 8;  % Nur 8 Features

% Speicher vorallokieren für Endergebnis
X = zeros(nt * nx, n_features, 'single');

% Blockweise Verarbeitung
for b = 1:num_blocks
    start_idx = (b-1)*block_size + 1;
    end_idx = min(b*block_size, nx);
    block_width = end_idx - start_idx + 1;

    current_block = data(:, start_idx:end_idx);

    % Feature-Berechnung für die 8 gewünschten Features
    temp_features = zeros(nt, block_width, n_features, 'single');

    % 1. Envelope
    for ix = 1:block_width
        analytic_signal = hilbert(current_block(:, ix));
        temp_features(:,ix,1) = abs(analytic_signal);
        temp_features(:,ix,3) = [0; diff(unwrap(angle(analytic_signal))) / dt] / (2*pi); % Inst. Freq
        temp_features(:,ix,2) = 0; % Platzhalter für Abs Grad Phase (wird unten berechnet)
    end

    % 2. Abs Grad Phase
    phase_block = angle(hilbert(current_block));
    [Gx_p, Gy_p] = gradient(phase_block);
    temp_features(:,:,2) = sqrt(Gx_p.^2 + Gy_p.^2);

    % 4. Mean
    kernel = ones(window_size, 'single') / window_size^2;
    temp_features(:,:,4) = conv2(current_block, kernel, 'same');

    % 5. Entropy
    temp_features(:,:,5) = entropyfilt(current_block, true(window_size));

    % 6. Skewness & 7. Kurtosis
    mean_local = temp_features(:,:,4);
    diff_local = current_block - mean_local;
    var_local = conv2(diff_local.^2, kernel, 'same');
    std_local = sqrt(var_local);
    norm_diff = diff_local ./ std_local;
    temp_features(:,:,6) = conv2(norm_diff.^3, kernel, 'same');      % Skewness
    temp_features(:,:,7) = conv2(norm_diff.^4, kernel, 'same') - 3;  % Kurtosis

    % 8. Abs Grad Skewness
    [Gx_s, Gy_s] = gradient(temp_features(:,:,6));
    temp_features(:,:,8) = sqrt(Gx_s.^2 + Gy_s.^2);

    % In Ergebnismatrix einfügen
    idx_range = (start_idx-1)*nt + 1 : end_idx*nt;
    X(idx_range, :) = reshape(temp_features, [], n_features);

    clear temp_features current_block Gx_p Gy_p Gx_s Gy_s diff_local var_local std_local norm_diff mean_local phase_block;
end

% Nachbearbeitung
X(~isfinite(X)) = 0;
X = normalize(X, 'range');

% SOM Training mit doppelter Batchsize
batch_size = 20000; % vorher 10000
n_samples = size(X,1);
num_batches = ceil(n_samples/batch_size);

% Zufällige Permutation aller Indizes
all_indices = randperm(n_samples);

% Feature Namen definieren (vor dem Training)
feature_names = {'Envelope', 'Abs Grad Phase', 'Inst. Freq', 'Mean', ...
                 'Entropy', 'Skewness', 'Kurtosis', 'Abs Grad Skewness'};

% SOM Netzwerk initialisieren mit Input Labels
dimension1 = 10;
dimension2 = 10;
net = selforgmap([dimension1 dimension2]);
net.trainParam.showWindow = false;
net.trainParam.epochs = 1;  % Ein Epoch pro Batch

% Training mit allen Batches
for i = 1:num_batches
    start_idx = (i-1)*batch_size + 1;
    end_idx = min(i*batch_size, n_samples);
    batch_indices = all_indices(start_idx:end_idx);
    current_batch = X(batch_indices,:);
    net = train(net, current_batch');
end

% Visualisierung in separaten Fenstern
figure('Name', 'SOM Hits');
plotsomhits(net, X');
title('SOM Hits');

figure('Name', 'SOM Neighbor Distances');
plotsomnd(net);
title('SOM Neighbor Distances');

% Visualisierung der Feature Planes mit manuellen Namen
figure('Name', 'SOM Feature Planes');
plotsomplanes(net);

% Feature-Namen als Titel setzen
for i = 1:n_features
    subplot(2,4,i); % 2x4 für 8 Features
    title(feature_names{i}, 'Interpreter', 'none');
end

%figure('Name', 'SOM Unit Positions');
%plotsompos(net);
%title('SOM Unit Positions');

% BMU für jeden Datenpunkt bestimmen
bmu_indices = vec2ind(net(X'))'; % Länge: nt*nx

% KMeans-Clustering der BMUs (K=10)
K = 10;
bmu_cluster = kmeans(double(bmu_indices), K, 'Replicates', 5);

% In Bildform bringen (Cluster-Labels)
bmu_cluster_img = reshape(bmu_cluster, nt, nx);

% Plotten
figure('Name', 'Radargramm BMU-KMeans-Cluster');
imagesc(bmu_cluster_img);
xlabel('Spur'); ylabel('Zeit (Samples)');
title('BMU-Cluster (KMeans, K=10)');
colormap(jet);
colorbar;