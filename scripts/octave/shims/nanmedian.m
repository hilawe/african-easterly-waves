function m = nanmedian(x)
% NANMEDIAN  Median ignoring NaN, as MATLAB's Statistics Toolbox provides it.
%
% Version 1 calls this on a vector of wind values taken over one wave's trough cells,
% `nanmedian(ut2(pot_wv(id).wave_points))`. The NaN it is guarding against is real and
% deliberate: smth9_f propagates missing values on purpose, so a plain median would
% return NaN for any wave touching masked ground and silently end its track.
%
% An all-NaN input returns NaN, which is what the toolbox does.
  x = x(:);
  x = x(~isnan(x));
  if isempty(x)
    m = NaN;
  else
    m = median(x);
  end
end
