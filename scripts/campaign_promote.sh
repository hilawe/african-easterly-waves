# The campaign driver's promotion step, in its own file so it can be sourced by the
# driver and by its test alike. promote SRC DST moves a completed attempt's run
# directory to its canonical path under an exclusive lock: the lock is a directory
# created with mkdir, which is atomic and fails when it exists, so two campaign
# commands on one root cannot both pass the destination check and nest one run
# inside the other. A lock left by a crash is reported by name and removed by hand.
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
