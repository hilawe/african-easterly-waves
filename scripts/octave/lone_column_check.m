% The Sahara case's divergence step in isolation, on the version 1 side: a five by five
% field that is missing everywhere except one column whose values change sign twice.
% Version 1 contours its masked advection field with `contours` (the shim forwards to
% Octave's contourc), and the question is whether a lone unmasked column bounded by
% missing cells yields zero-level contours. Prints one line per contour found, in the
% form the trace script records. A count of contours here is OCTAVE'S; whether MATLAB's
% contouring agrees on this input is not established by running this.
addpath(fullfile(fileparts(mfilename('fullpath')), 'shims'));
lat = [22 20 18 16 14]; lon = [-13 -11 -9 -7 -5];
[NG, LG] = meshgrid(lon, lat);
f = nan(5, 5); f(2, 3) = 1.3e-11; f(3, 3) = -3.5e-11; f(4, 3) = 7.4e-11;
g = f; g(2, 4) = 1.0e-11; g(3, 4) = -2.0e-11; g(4, 4) = 5.0e-11;
% and the eastern Pacific shape: one finite ROW, plus plus minus, masked above and below
r = nan(5, 5); r(2, 2) = 2.0e-11; r(2, 3) = 1.0e-11; r(2, 4) = -3.0e-11;
rlat = [-4 -6 -8 -10 -12]; rlon = [-141 -139 -137 -135 -133];
% and the South American shape: one finite ROW of five cells whose sign changes in the
% INTERIOR of the row, masked above, with two cells of one sign below its eastern end.
% The values are the port's own at that step, so the crossing here is that step's.
q = nan(5, 6);
q(3, 1) = 4.071516838295352e-10; q(3, 2) = 3.615261478456151e-10;
q(3, 3) = 1.202971274689068e-10; q(3, 4) = -2.3182610628854524e-10;
q(3, 5) = -1.3041326376378817e-10; q(4, 5) = -8.94e-10; q(4, 6) = -1.46e-09;
qlat = [-28 -30 -32 -34 -36]; qlon = [-65 -63 -61 -59 -57 -55];
for which = {'lone_column', 'two_columns', 'lone_row', 'interior_change_row'}
  if strcmp(which{1}, 'lone_column'); z = f; [NG, LG] = meshgrid(lon, lat);
  elseif strcmp(which{1}, 'two_columns'); z = g; [NG, LG] = meshgrid(lon, lat);
  elseif strcmp(which{1}, 'lone_row'); z = r; [NG, LG] = meshgrid(rlon, rlat);
  else; z = q; [NG, LG] = meshgrid(qlon, qlat); end
  c = contours(NG, LG, z, [0 0]);
  k = 1; n = 0;
  while k <= size(c, 2)
    npts = c(2, k); n = n + 1;
    printf('%s contour %d: %d points, lon %.4f..%.4f, lat %.4f..%.4f\n', which{1}, n, npts, ...
           min(c(1, k+1:k+npts)), max(c(1, k+1:k+npts)), min(c(2, k+1:k+npts)), max(c(2, k+1:k+npts)));
    k = k + npts + 1;
  end
  printf('%s: %d contours\n', which{1}, n);
end
