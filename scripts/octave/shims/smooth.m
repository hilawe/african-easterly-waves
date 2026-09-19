function out = smooth(y, span)
% SMOOTH  MATLAB's moving average with shrinking end windows.
%
% Version 1 applies `smooth(x, 5)` to the mean latitude and longitude of every surviving
% track, so this decides the coordinates the published record carries. MATLAB's 'moving'
% method shrinks the window symmetrically at the ends, so the first and last values come
% back unchanged and the second and second-last are three-point means.
%
% The port implements the same rule in association._moving_average_5, and this shim is
% written from MATLAB's documented behavior rather than from the port, so the two are
% independent statements of it.
  y = y(:);
  n = numel(y);
  half = (span - 1) / 2;
  out = zeros(n, 1);
  for i = 1:n
    h = min([half, i - 1, n - i]);
    out(i) = mean(y(i - h : i + h));
  end
end
