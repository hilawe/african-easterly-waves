% Run version 1's tracker with its intermediate positions dumped at chosen timesteps.
%
% WHAT THIS ANSWERS. The port detects the eastern Africa wave at nine of ten timesteps and
% its association still fragments the feature into three tracks, which the prune then
% discards (THE_EASTERN_AFRICA_WAVE.md). Two candidates were left open there: version 1's
% trough axes may differ from the port's, since the port's contour handling is a recorded
% deliberate divergence, or version 1's association may bridge a gap the port's does not.
% Both are questions about version 1's INTERNAL state, which no oracle has reached, because
% find_ews_f.m returns finished tracks and nothing else.
%
% THE ARCHIVED SOURCE STAYS BYTE-IDENTICAL. The instrumented copy lives in
% $AEW_ORACLE_DIR/v1_instrumented, built by scripts/make_instrumented_v1.py, and is not
% committed for the same reason the patched merge copy was not: it is a modified copy of
% the archived record. It adds four printf blocks and one global, all inert when
% AEWDBG_TIMES is empty, and changes no arithmetic.
%
% WHAT IT PRINTS, one line per item per selected timestep:
%   AXIS   time npoints lat_mean lon_mean lat_span lon_span   (raw contour candidates)
%   COARSE time lat_mean lon_mean                             (after the coarse merge)
%   FINE   time lat_mean lon_mean                             (after the fine merge)
%   TRACK  time index nobs last_lat last_lon last_time        (live tracks, end of step)
warning('off','all');
max_recursion_depth(500);
% octave-cli, never `octave --no-gui`; see run_tracker.m for why.
repo = getenv('AEW_REPO');
S = getenv('AEW_ORACLE_DIR');
if isempty(repo) || isempty(S)
  error('set AEW_REPO and AEW_ORACLE_DIR');
end
src = fullfile(S, 'v1_instrumented');
if exist(fullfile(src, 'find_ews_f.m'), 'file') ~= 2
  error('no instrumented find_ews_f.m under %s; run make_instrumented_v1.py first', src);
end
addpath(fullfile(repo, 'scripts/octave/shims'));
addpath(src);
d = load(fullfile(S, 'tracker_case.mat'));

global AEWDBG_TIMES
% the ten timesteps of version 1's track 41, the feature first investigated; the
% environment variable AEW_DUMP_TIMES (space separated) overrides the list so another
% feature's lifetime can be dumped without editing this file, and the list used is
% recorded in the output's producer record either way
AEWDBG_TIMES = [33027.00 33027.50 33027.75 33028.25 33028.50 33028.75 ...
                33029.00 33029.25 33029.75 33030.25];
dump_env = getenv('AEW_DUMP_TIMES');
if ~isempty(dump_env)
  AEWDBG_TIMES = str2num(dump_env);  %#ok<ST2NM>
end
printf('dumping intermediates at %d timesteps\n', numel(AEWDBG_TIMES));
fflush(stdout);

% PRODUCING PROVENANCE, hashed HERE at run time from the files that execute, and
% carried inside the output. Which source produced the saved tracks is then a fact the
% output states, not something inferred later from a file name or from whatever tree
% sits beside the output when a comparison runs. The repair mode is DETECTED from the
% executed find_ews_f.m: the archived call convhull(...,'simplify',true) is what Octave
% refuses, so its absence in the executed copy is what makes this run the faithful one.
function h = file_sha256(path)
  fid = fopen(path, 'rb');
  if fid < 0
    error('cannot read %s for hashing', path);
  end
  b = fread(fid, Inf, 'uint8=>uint8');
  fclose(fid);
  h = hash('sha256', char(b'));
end
executed = struct();
srcs = dir(fullfile(src, '*.m'));
for k = 1:numel(srcs)
  key = strrep(strrep(['v1_instrumented/' srcs(k).name], '/', '__'), '.', '_');
  executed.(key) = file_sha256(fullfile(src, srcs(k).name));
end
shims = dir(fullfile(repo, 'scripts/octave/shims', '*.m'));
for k = 1:numel(shims)
  key = strrep(strrep(['shims/' shims(k).name], '/', '__'), '.', '_');
  executed.(key) = file_sha256(fullfile(repo, 'scripts/octave/shims', shims(k).name));
end
executed_src = fileread(fullfile(src, 'find_ews_f.m'));
n_unrepaired = numel(strfind(executed_src, "'simplify',true"));
producer = struct();
producer.producer = 'scripts/octave/run_tracker_instrumented.m';
producer.case_id = d.case_id;
producer.instrumented = true;
producer.repaired_convhull = (n_unrepaired == 0);
producer.unrepaired_convhull_sites = n_unrepaired;
producer.octave_version = version();
producer.dump_times = AEWDBG_TIMES(:)';
producer.executed_source_sha256 = executed;
producer.runner_sha256 = file_sha256(fullfile(repo, 'scripts/octave/run_tracker_instrumented.m'));
producer_json = jsonencode(producer);
% THE RUN'S OWN DIGEST, printed into the log before any dump, so the log can be bound
% to the output that carries this same producer record: a log with the right dump
% schedule and track count could still belong to another run, and a review showed the
% trace accepting one. The trace requires this line to equal the digest of the
% producer_json string it reads from the output.
printf('RUN %s\n', hash('sha256', producer_json));
fflush(stdout);

t0 = tic;
ews = find_ews_f(d.u_c, d.v_c, d.currv_anom_c, d.advcurrv_anom_c, d.lat_c, d.lon_c, ...
                 d.time, d.latgrid, d.longrid, d.u, d.v, d.currv_anom, d.rean, d.level);
printf('find_ews_f returned %d tracks in %.0f s\n', numel(ews), toc(t0));
% The instrumented copy must not change the answer. The caller compares this count and
% the saved tracks against the uninstrumented run's tracker_octave.mat.
out = struct(); out.n = numel(ews); out.case_id = d.case_id;
out.producer_json = producer_json;
for i = 1:numel(ews)
  out.(sprintf('lat%d', i-1))  = ews(i).meanlat(:);
  out.(sprintf('lon%d', i-1))  = ews(i).meanlon(:);
  out.(sprintf('time%d', i-1)) = ews(i).time(:);
end
save('-v7', fullfile(S, 'tracker_octave_instrumented.mat'), '-struct', 'out');
printf('saved\n');
