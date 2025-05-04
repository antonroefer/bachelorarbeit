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
        temp_features(:,ix,2) = [0; diff(unwrap(angle(analytic_signal))) / dt] / (2*pi); % Inst. Freq
        temp_features(:,ix,3) = cos(angle(analytic_signal)); % Real Inst. Phase
        temp_features(:,ix,4) = sin(angle(analytic_signal)); % Imag Inst. Phase
    end

    for it = 1:nt
        temp_features(it,:,6) = skewness()
    end
    
    kernel = ones(window_size, 'single') / window_size^2;

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

    % 8. Mean
    temp_features(:,:,8) = conv2(current_block, kernel, 'same');


    % In Ergebnismatrix einfügen
    idx_range = (start_idx-1)*nt + 1 : end_idx*nt;
    X(idx_range, :) = reshape(temp_features, [], n_features);

    clear temp_features current_block Gx_p Gy_p Gx_s Gy_s diff_local var_local std_local norm_diff mean_local phase_block;
end

% Nachbearbeitung
X(~isfinite(X)) = 0;
X = normalize(X, 'range');

% SOM Training mit allen Daten auf einmal

% Feature Namen definieren (vor dem Training)
feature_names = {'Envelope', 'Inst. Frequency', 'Real Inst. Phase', 'Imag. Inst. Phase', ...
                 'Entropy', 'Skewness', 'Kurtosis', 'Mean'};

% SOM Netzwerk initialisieren mit Input Labels
dimension1 = 10;
dimension2 = 10;
net = selforgmap([dimension1 dimension2]);
net.trainParam.showWindow = false;
net.trainParam.epochs = 10;  % Anzahl der Epochen für das Training

% Training mit allen Daten
net = train(net, X');

% Visualisierung in separaten Fenstern
figure('Name', 'SOM Hits');
plotsomhits(net, X');
title('SOM Hits');

figure('Name', 'SOM Neighbor Distances');
plotsomnd(net);
title('SOM Neighbor Distances');

figure('Name', 'SOM Feature Planes');
plotsomplanes(net);

% Achsen finden und Titel setzen
ax = findall(gcf, 'Type', 'axes');
ax = flipud(ax); % MATLAB sortiert von zuletzt gezeichnet nach zuerst → umdrehen

for i = 1:n_features
    title(ax(i), feature_names{i}, 'Interpreter', 'none');
end

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

clear