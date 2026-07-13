clear; clc;

% --- BER ---offset=0---
x = [10, 11, 12, 13, 14, 15];
y0 = [0.00287619048 0.00630476190 0.00995714286 0.0148857143 0.0190857143 0.0252571429]; %% without
y1 = [5.054187192118227e-4 1.306493506493507e-3 2.954285714285714e-3 7.447619047619048e-3 1.110000000000000e-2 1.771428571428571e-2]; %% sparse code
% y2 = [5.714e-05 1.429e-04 7.571e-04 1.871e-03 4.843e-03 8.543e-03]; %% only BCH
y3 = [0.00019210526315789473 0.0003435374149659864 0.0009942857142857143 0.0020653061224489796 0.004885714285714286 0.008942857142857143]; %% SLNN

%------------FER---------------

% x = [10, 11, 12, 13, 14, 15];
y00 = [0.0215154639 0.0399114286 0.0650571429 0.0930200000 0.126180000 0.159560000]; %% without
y11 = [0.0010482758620689655 0.0027454545454545453 0.00664 0.015133333333333334 0.0232 0.0386]; %% sparse code
% y2 = [2.000e-04 6.000e-04 2.800e-03 6.900e-03 1.790e-02 3.090e-02]; %% only BCH
y33 = [0.0004105263157894737 0.0007523809523809524 0.0022266666666666667 0.004914285714285714 0.01035 0.0188]; %% SLNN


% --- plot ---
figure; hold on; grid on; box on;
semilogy(x, y0, 'v-','Color', [0 0.4470 0.7410], 'LineWidth', 2, 'MarkerSize', 8);
semilogy(x, y1, 's-','Color', [0.8500 0.3250 0.0980], 'LineWidth', 2, 'MarkerSize', 8);
% semilogy(x, y2, 'o--', 'LineWidth', 2, 'MarkerSize', 8);
semilogy(x, y3, 'd-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2, 'MarkerSize', 8);
% semilogy(x, y4, 'x--', 'LineWidth', 2, 'MarkerSize', 8);

semilogy(x, y00, 'v-.','Color', [0 0.4470 0.7410], 'LineWidth', 2, 'MarkerSize', 8);
semilogy(x, y11, 's-.', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2, 'MarkerSize', 8);
semilogy(x, y33, 'd-.', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2, 'MarkerSize', 8);


xlabel('\sigma_0 / \mu_0', 'FontSize', 12);
ylabel('BER & FER', 'FontSize', 12);
legend({'BER - without coding','BER - 7/9 sparse code with ML decoding','BER - 7/9 sparse code with SLNN-based decoding', 'FER - without coding','FER - 7/9 sparse code with ML decoding','FER - 7/9 sparse code with SLNN-based decoding'}, 'Location', 'northwest', 'FontSize', 11);


grid on;
ax = gca;
xticks(x);
xtickformat('%.0f');   % nếu x là số nguyên như 5,6,...15
ax.YScale = 'log';
ax.YTick = [1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0];
ylim([1e-4 1e-0]); 


