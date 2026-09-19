function ch = contours(x, y, z, levels)
% CONTOURS  The contour matrix, as version 1's MATLAB provided it.
%
% `contours` is an old MATLAB name for what is now `contourc`. Octave carries contourc
% and not the alias, so this forwards. The RETURN FORMAT is the documented contour
% matrix both share: each contour is a header column [level; point_count] followed by
% that many [x; y] vertex columns, which is the format find_ews_f.m parses directly.
%
% WHERE THIS COULD DIFFER FROM MATLAB. Only in the contouring ALGORITHM, which decides
% where vertices fall and in what order contours appear. It cannot change the matrix
% LAYOUT, which is what the parser reads. Any conclusion drawn about the parser
% transfers; a count of contours is Octave's.
%
% find_ews_f calls this with meshgrid outputs rather than vectors, so the grids are
% reduced to the axes contourc expects.
  if ~isvector(x); x = x(1, :); end
  if ~isvector(y); y = y(:, 1)'; end
  ch = contourc(x, y, z, levels);
end
