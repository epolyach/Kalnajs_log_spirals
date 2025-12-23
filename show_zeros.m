DII = D_global(I,I);

tol = 1e-12 * norm(DII, 1);       % adjust as needed
I = 1:NJ;
Aabs = abs(DII);
amax = max(Aabs(:));
if amax == 0, error('DII is all zeros'); end
Alog = log10(Aabs / amax + 1e-16);     % normalized log10 magnitude

% pcolor drops last row/col; pad by replication for proper coverage
AlogP = padarray(Alog, [1 1], 'replicate', 'post');

figure; clf
h1 = pcolor(AlogP); shading flat
axis equal tight
colormap(parula); colorbar
caxis([-12 0])                       % show 12 orders of magnitude
title('log_{10} |D_{global}| (normalized) with near-zero overlay')

hold on
% Near-zero mask and overlay as red points
mask = (Aabs <= tol);
[rowZ, colZ] = find(mask);
scatter(colZ, rowZ, 8, 'r', 'filled', 'MarkerFaceAlpha', 0.7);

% Optional: draw l-block boundaries if block structure is (l, alpha)
if exist('N_alpha','var') && exist('n_l','var') && ~isempty(N_alpha) && ~isempty(n_l)
    for k = 1:n_l-1
        x = k*N_alpha + 0.5;  % boundary between blocks
        y = x;
        plot([x x], [0.5 size(DII,1)+0.5], 'k:', 'LineWidth', 0.5)
        plot([0.5 size(DII,2)+0.5], [y y], 'k:', 'LineWidth', 0.5)
    end
end
hold off