clear; clc;

% --- data BER---
x = [2.00e-08,2.00e-07,2.00e-06,2.00e-05,2.00e-04,2.00e-03];
y0 = [3.104981e-03, 3.105027e-03, 3.105488e-03, 3.110094e-03, 3.156156e-03, 3.616768e-03]; %% without code
y1 = [3.905000e-04, 3.872143e-04, 3.966429e-04, 3.952857e-04, 5.239286e-04, 1.696857e-03]; %% 7/9 sparse code
y3 = [4.578754578754579e-05, 4.528301886792453e-05, 4.63594470046083e-05, 5.083884087442806e-05, 0.0001554531490015361, 0.0016825396825396826];


% --- data FER---
y00 = [2.153345e-02, 2.153377e-02, 2.153693e-02, 2.156858e-02, 2.188500e-02, 2.504432e-02]; %% without code
y11 = [8.405000e-04, 8.315000e-04, 8.460000e-04, 8.485000e-04, 1.133500e-03, 3.678500e-03];
y33 = [9.615384615384615e-05, 9.937106918238994e-05, 0.00010193548387096774, 0.00011601423487544484, 0.00033978494623655914, 0.003488888888888889];

figure; hold on; grid on; box on;
semilogy(x, y0, 'v-','Color', [0 0.4470 0.7410], 'LineWidth', 2, 'MarkerSize', 8);
semilogy(x, y1, 's-','Color', [0.8500 0.3250 0.0980], 'LineWidth', 2, 'MarkerSize', 8);
% semilogy(x, y2, 'o--', 'LineWidth', 2, 'MarkerSize', 8);
semilogy(x, y3, 'd-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2, 'MarkerSize', 8);
% semilogy(x, y4, 'x--', 'LineWidth', 2, 'MarkerSize', 8);

semilogy(x, y00, 'v-.','Color', [0 0.4470 0.7410], 'LineWidth', 2, 'MarkerSize', 8);
semilogy(x, y11, 's-.', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2, 'MarkerSize', 8);
semilogy(x, y33, 'd-.', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2, 'MarkerSize', 8);


xlabel('P_1', 'FontSize', 12);
ylabel('BER & FER', 'FontSize', 12);
% legend({'without coding','7/9 sparse code','BCH (15,7,2)'}, 'Location', 'northwest', 'FontSize', 11);
% legend({'BCH (15,7,2)'}, 'Location', 'northwest', 'FontSize', 11);
legend({'BER - without coding','BER - 7/9 sparse code with ML decoding','BER - 7/9 sparse code with SLNN-based decoding', 'FER - without coding','FER - 7/9 sparse code with ML decoding','FER - 7/9 sparse code with SLNN-based decoding'}, 'Location', 'northwest', 'FontSize', 11);

grid on;
ax = gca;
ax.XScale = 'log';
ax.YScale = 'log';
ax.YTick = [3e-3 1e-2 3e-2];
ylim([1e-5 1e-0]);
ax.TickLabelInterpreter = 'latex';
yticks([1e-5 1e-4 1e-3 1e-2 1e-1 1e-0]);
yticklabels({'$10^{-5}$','$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'});


ax.XTick = [2.00e-08,2.00e-07,2.00e-06,2.00e-05,2.00e-04,2.00e-03];
xlim([2.00e-08 2.00e-03]); 
ax.XTickLabel = {'$2\times10^{-8}$', '$2\times10^{-7}$','$2\times10^{-6}$','$2\times10^{-5}$','$2\times10^{-4}$','$2\times10^{-3}$'};
