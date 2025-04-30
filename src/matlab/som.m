% Pfad zur .mat-Datei
matfile_path = 'C:\Users\anton\Documents\Studium\Bachelorarbeit\GPR_Daten_mat\radargrams.mat';

% .mat-Datei laden und in single konvertieren
load(matfile_path, 'radargrams');
data = single(radargrams{9});
clear radargrams

[nt, nx] = size(data);
dt = single(1e-9);

% Parameter
env_size = 7;
window_size = 2 * env_size + 1;
block_size = 50;  % Kleinere Blöcke
num_blocks = ceil(nx/block_size);
n_features = 16;  % Anzahl Features beibehalten

% Speicher vorallokieren für Endergebnis
X = zeros(nt * nx, n_features, 'single');

% Blockweise Verarbeitung
for b = 1:num_blocks
    start_idx = (b-1)*block_size + 1;
    end_idx = min(b*block_size, nx);
    block_width = end_idx - start_idx + 1;
    
    % Aktueller Block
    current_block = data(:, start_idx:end_idx);
    
    % Hilbert Transform
    temp_features = zeros(nt, block_width, n_features, 'single');
    
    for ix = 1:block_width
        trace = current_block(:, ix);
        analytic_signal = hilbert(trace);
        temp_features(:,ix,1) = abs(analytic_signal);  % amplitude_envelope
        temp_features(:,ix,2) = angle(analytic_signal); % instantaneous_phase
        temp_features(:,ix,3) = [0; diff(unwrap(angle(analytic_signal))) / dt] / (2*pi); % inst_freq
    end
    
    % Gradient Features
    [Gx, Gy] = gradient(current_block);
    temp_features(:,:,4) = sqrt(Gx.^2 + Gy.^2);  % abs_gradient
    
    % Statistische Features
    kernel = ones(window_size, 'single') / window_size^2;
    temp_features(:,:,5) = conv2(current_block, kernel, 'same');  % mean_env
    temp_features(:,:,6) = medfilt2(current_block, [window_size, window_size], 'symmetric');  % median_env
    temp_features(:,:,7) = stdfilt(current_block, true(window_size));  % std_env
    temp_features(:,:,8) = entropyfilt(current_block, true(window_size));  % entropy_env
    
    % Ordnungsstatistiken
    num_elements = window_size^2;
    temp_features(:,:,9) = ordfilt2(current_block, num_elements, true(window_size));  % max
    temp_features(:,:,10) = ordfilt2(current_block, 1, true(window_size));  % min
    temp_features(:,:,11) = temp_features(:,:,9) - temp_features(:,:,10);  % range
    
    % Schiefe und Wölbung
    mean_local = temp_features(:,:,5);
    diff_local = current_block - mean_local;
    var_local = conv2(diff_local.^2, kernel, 'same');
    std_local = sqrt(var_local);
    norm_diff = diff_local ./ std_local;
    
    temp_features(:,:,12) = conv2(norm_diff.^3, kernel, 'same');  % skewness
    temp_features(:,:,13) = conv2(norm_diff.^4, kernel, 'same') - 3;  % kurtosis
    
    % Zusätzliche Features
    temp_features(:,:,14) = conv2(temp_features(:,:,3), kernel, 'same');  % mean_inst_freq
    [Gx_p, Gy_p] = gradient(temp_features(:,:,2));
    temp_features(:,:,15) = sqrt(Gx_p.^2 + Gy_p.^2);  % abs_grad_phase
    [Gx_s, Gy_s] = gradient(temp_features(:,:,12));
    temp_features(:,:,16) = sqrt(Gx_s.^2 + Gy_s.^2);  % abs_grad_skew
    
    % In Ergebnismatrix einfügen
    idx_range = (start_idx-1)*nt + 1 : end_idx*nt;
    X(idx_range, :) = reshape(temp_features, [], n_features);
    
    % Zwischenspeicher freigeben
    clear temp_features current_block
end

% Nachbearbeitung
X(~isfinite(X)) = 0;
X = normalize(X, 'zscore');

% SOM Training mit zufälligen Batches
batch_size = 10000;
n_samples = size(X,1);
num_batches = ceil(n_samples/batch_size);

% Zufällige Permutation aller Indizes
all_indices = randperm(n_samples);

% SOM Netzwerk initialisieren
dimension1 = 10;
dimension2 = 10;
net = selforgmap([dimension1 dimension2]);
net.trainParam.showWindow = false;

% Initiales Training mit erstem Batch
batch_indices = all_indices(1:batch_size);
initial_batch = X(batch_indices,:);
net = train(net, initial_batch');

% Weiteres Training mit restlichen Batches
for i = 2:num_batches
    start_idx = (i-1)*batch_size + 1;
    end_idx = min(i*batch_size, n_samples);
    batch_indices = all_indices(start_idx:end_idx);
    current_batch = X(batch_indices,:);
    net = adapt(net, current_batch');
end

% Visualisierung
figure('Name', 'SOM Results');
subplot(2,2,1);
plotsomhits(net, X'); % Visualisierung mit reduziertem Datensatz
title('SOM Hits');

subplot(2,2,2);
plotsomnd(net);
title('SOM Neighbor Distances');

subplot(2,2,3);
plotsomplanes(net);
title('SOM Planes');