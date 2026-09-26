#!/usr/bin/env bash
# The protocol tracking campaign: every year of both datasets through the shared entry
# point, four runs at a time, one durable parent process, one log and one completion
# record per run, nothing overwritten, a run skipped when its record already exists.
#
# WHY FOUR. The machine has eight cores and 16 GB. A run holds a year's fields at about
# 1.5 to 2 GB at peak beside the 225 MB climatology, so four runs fit with room and six
# would risk swapping. Runs land outside git, under data/protocol_runs/campaign, because
# each retained case is 387 MB; the small files (tracks, records, logs) are collected
# into the evidence directory afterwards by scripts/collect_protocol_campaign.py.
#
#     nohup scripts/run_protocol_campaign.sh > data/protocol_runs/campaign/campaign.log 2>&1 &
#
# EXIT STATUS. A failed run is logged AND counted, and the command exits 1 when any run
# failed, so a caller (or a restart) cannot read a partial campaign as a complete one.
# The first campaign logged failures without failing, which a review named; no run
# failed then, so nothing was misread, and the repair is for restarts. A restart skips
# every run whose record exists and reruns only the rest, never rewriting a record.
# COMPLETION IS RECONCILED, not inferred from the failure count: the command exits 1
# unless every expected record exists, so a worker that ends without reporting (a
# signal, a lost shell, a scheduler failure) cannot produce a successful campaign, and
# the year range is validated before anything runs. EVERY ATTEMPT RUNS IN ITS OWN
# DIRECTORY, `<dataset>_<year>.attempt-<time>-<pid>/<dataset>_<year>/`, and only an
# attempt whose completion record exists is promoted, by one rename, to the canonical
# `<dataset>_<year>/`. A partial attempt stays where it is, never deleted, and a process
# surviving the death of its parent keeps publishing into its own attempt directory and
# can never land a record in a promoted run, which a review reproduced under the earlier
# scheme of setting a partial directory aside and recreating the same pathname. The
# first campaign's runs sit at the canonical paths already and are skipped by record.
# PROMOTION IS EXCLUSIVE: the promoting worker first creates `<dataset>_<year>.promoting`
# with mkdir, which is atomic and fails when it exists, so two campaign commands started
# against one root cannot both pass the destination check and nest one run inside the
# other (a review's scenario). A worker that cannot take the lock fails its run and says
# so; a lock left by a crash is reported by name and removed by hand.
#
# OVERRIDES, for the test and for a scratch campaign: OUT (the campaign root), YEARS
# (first-last), ENTRY (the command that runs one year, given the entry point's
# arguments), MANIFEST, and CAL_<dataset> / CACHE_<dataset> paths.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-data/protocol_runs/campaign}"
MANIFEST="${MANIFEST:-docs/aewc_v2/protocol/manifest_2026-09-25.json}"
YEARS="${YEARS:-1979-2010}"
ENTRY="${ENTRY:-.venv/bin/python3 scripts/export_protocol_case.py}"
CAL_eraint="${CAL_eraint:-docs/aewc_v2/artifacts/thresholds_protocol_eraint_1979_2010.json}"
CAL_era5="${CAL_era5:-docs/aewc_v2/artifacts/thresholds_protocol_era5_1979_2010.json}"
CACHE_eraint="${CACHE_eraint:-data/climo/climo_eraint_1979_2010.npz}"
CACHE_era5="${CACHE_era5:-data/climo/climo_era5_1979_2010.npz}"
case "$YEARS" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]) ;;
  *) echo "REFUSED: YEARS must be first-last, four digits each, got '$YEARS'"; exit 1 ;;
esac
Y0="${YEARS%-*}"; Y1="${YEARS#*-}"
if [ "$Y0" -gt "$Y1" ]; then echo "REFUSED: YEARS $YEARS is reversed"; exit 1; fi
mkdir -p "$OUT"
FAILURES="$OUT/.failures.$$"
: > "$FAILURES"
echo "campaign started $(date -u +%FT%TZ) at $(git rev-parse --short HEAD), tree $(git status --porcelain | wc -l | tr -d ' ') dirty paths"

run_one() {
  dataset="$1"; year="$2"
  case "$dataset" in
    eraint) cal="$CAL_eraint"; cache="$CACHE_eraint" ;;
    era5)   cal="$CAL_era5";   cache="$CACHE_era5" ;;
    *) echo "unknown dataset $dataset"; echo "$dataset $year" >> "$FAILURES"; return 1 ;;
  esac
  final="$OUT/${dataset}_${year}"
  record="$final/tracking_${dataset}_${year}.json"
  if [ -f "$record" ]; then echo "skip $dataset $year, record exists"; return 0; fi
  if [ -e "$final" ]; then
    echo "FAILED $dataset $year: $final exists without a completion record and is never reused or removed, move it aside by hand"
    echo "$dataset $year" >> "$FAILURES"; return 1
  fi
  attempt="$OUT/${dataset}_${year}.attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  mkdir "$attempt" || { echo "FAILED $dataset $year: cannot create $attempt"; echo "$dataset $year" >> "$FAILURES"; return 1; }
  mkdir "$attempt/${dataset}_${year}"
  started=$(date +%s)
  if $ENTRY --manifest "$MANIFEST" --dataset "$dataset" --year "$year" \
       --stage tracking --calibration "$cal" --climo-cache "$cache" --out-dir "$attempt" \
       > "$attempt/${dataset}_${year}/run.log" 2>&1 \
     && [ -f "$attempt/${dataset}_${year}/tracking_${dataset}_${year}.json" ] \
     && promote "$attempt/${dataset}_${year}" "$final"; then
    rmdir "$attempt" 2>/dev/null || true
    echo "done $dataset $year in $(( $(date +%s) - started )) s"
  else
    echo "FAILED $dataset $year in $(( $(date +%s) - started )) s, attempt kept at $attempt"
    echo "$dataset $year" >> "$FAILURES"
  fi
}

promote() {
  src="$1"; dst="$2"; lock="$dst.promoting"
  if ! mkdir "$lock" 2>/dev/null; then
    echo "promotion of $dst refused: $lock is held, by another campaign command or left by a crash"
    return 1
  fi
  if [ -e "$dst" ]; then
    echo "promotion of $dst refused: it exists"
    rmdir "$lock"; return 1
  fi
  mv "$src" "$dst"; status=$?
  rmdir "$lock"
  return $status
}
export -f promote
export -f run_one
export OUT MANIFEST ENTRY FAILURES CAL_eraint CAL_era5 CACHE_eraint CACHE_era5

for year in $(seq "$Y0" "$Y1"); do for dataset in eraint era5; do echo "$dataset $year"; done; done \
  | xargs -P 4 -L 1 bash -c 'run_one "$0" "$1"'
scheduler_status=$?

failed=$(wc -l < "$FAILURES" | tr -d ' ')
rm -f "$FAILURES"
expected=$(( 2 * (Y1 - Y0 + 1) ))
present=0
missing=""
for year in $(seq "$Y0" "$Y1"); do for dataset in eraint era5; do
  if [ -f "$OUT/${dataset}_${year}/tracking_${dataset}_${year}.json" ]; then present=$((present + 1)); else missing="$missing $dataset/$year"; fi
done; done
echo "campaign ended $(date -u +%FT%TZ): $present of $expected records, $failed failed, scheduler status $scheduler_status"
if [ "$failed" -gt 0 ] || [ "$present" -ne "$expected" ] || [ "$scheduler_status" -ne 0 ]; then
  echo "CAMPAIGN INCOMPLETE: missing records for:$missing. Rerun to retry them, records are never rewritten"
  exit 1
fi
