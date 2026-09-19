% Run version 1's WHOLE tracker on the exported window and record its tracks.
%
% find_ews_f.m does its own smoothing, masking, contouring, merging, association, prune
% and speed filter from raw fields, so this covers the association and prune stages that
% no oracle has reached before.
%
% THE SHIM DIRECTORY GOES FIRST ON THE PATH. It supplies three functions version 1's
% MATLAB environment had and Octave does not (contours, nanmedian, smooth). The archived
% source is left byte-identical.
warning('off','all');
% MATLAB'S OWN DEFAULT, 500, which is what version 1 ran under. An earlier version set
% 5000, which is simply unfaithful. The merge comparison separately measured that limits
% of 256, 500 and 5000 give identical results on real timesteps, so this is a fidelity
% choice and not a safety one.
max_recursion_depth(500);
% RUN THIS UNDER octave-cli, NEVER `octave --no-gui`. The latter suppresses the interface
% but still launches the GUI binary, which runs the interpreter on a Qt worker thread
% with a 544 KB stack. This tracker's region growth exhausts that at around 598 nested
% interpreter frames, well before Octave's own recursion counter would refuse, and a
% native stack overflow is not a catchable error: two runs stopped at timestep 41 of 60
% with no message, no partial output and nothing the harness could detect but a missing
% process. octave-cli runs the interpreter on the main thread and its 8 MB stack.
% VALIDATE BOTH PATHS BEFORE ADDING THEM. `addpath` accepts a directory that does not
% exist without complaint, and warnings are suppressed above, so a wrong or unset
% AEW_REPO would leave the archived source off the path and let any other `find_ews_f`
% already there be called instead. That failure would look like a successful run.
repo = getenv('AEW_REPO');
S = getenv('AEW_ORACLE_DIR');
if isempty(repo) || isempty(S)
  error('set AEW_REPO and AEW_ORACLE_DIR');
end
v1_src = fullfile(repo, 'data/aewc_v2_pilot/v1_src');
if exist(fullfile(v1_src, 'find_ews_f.m'), 'file') ~= 2
  error('no find_ews_f.m under %s; AEW_REPO is wrong', v1_src);
end
addpath(fullfile(repo, 'scripts/octave/shims'));
addpath(v1_src);
if exist(fullfile(S, 'tracker_case.mat'), 'file') ~= 2
  error('no tracker_case.mat in %s; run export_tracker_case.py first', S);
end
d = load(fullfile(S, 'tracker_case.mat'));
% REMOVE THE PREVIOUS ANSWER BEFORE COMPUTING A NEW ONE, so a run that dies leaves no
% file rather than a stale one. The case-id check downstream is the real guard; this
% makes the common failure loud instead of relying on it.
prev = fullfile(S, 'tracker_octave.mat');
if exist(prev, 'file'); delete(prev); end
printf('window of %d timesteps, coarse %dx%d, fine %dx%d\n', numel(d.time), ...
       numel(d.lat_c), numel(d.lon_c), size(d.latgrid,1), size(d.latgrid,2));
fflush(stdout);
t0 = tic;
ews = find_ews_f(d.u_c, d.v_c, d.currv_anom_c, d.advcurrv_anom_c, d.lat_c, d.lon_c, ...
                 d.time, d.latgrid, d.longrid, d.u, d.v, d.currv_anom, d.rean, d.level);
printf('find_ews_f returned %d tracks in %.0f s\n', numel(ews), toc(t0));
% CARRY THE CASE ID THROUGH. This script writes to a fixed path, so a run that dies
% part-way leaves the previous run's tracks in place and the comparison would read them
% against the new port output without noticing. Stamping the id the exporter computed
% lets the comparison refuse that.
out = struct(); out.n = numel(ews); out.case_id = d.case_id;
for i = 1:numel(ews)
  out.(sprintf('lat%d', i-1))  = ews(i).meanlat(:);
  out.(sprintf('lon%d', i-1))  = ews(i).meanlon(:);
  out.(sprintf('time%d', i-1)) = ews(i).time(:);
end
save('-v7', fullfile(S, 'tracker_octave.mat'), '-struct', 'out');
printf('saved\n');
