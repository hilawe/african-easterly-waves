# The campaign driver's promotion step, in its own file so it can be sourced by the
# driver and by its test alike. promote SRC DST moves a completed attempt's run
# directory to its canonical path under an exclusive lock, so two campaign commands on
# one root cannot both pass the destination check and nest one run inside the other.
# A lock left by a crash is reported by name and removed by hand.
#
# The lock and the move are made by Python's os.mkdir and os.rename, which are the
# mkdir and rename system calls. The mkdir command is not used, because the one that
# ships with Ubuntu 26.04 (uutils coreutils 0.10.0) reports success to both of two
# processes creating the same directory at once (1,111 of 3,000 trials, against 0 of
# 3,000 for GNU coreutils 9.4 on Ubuntu 24.04). The mv command is not used, because it
# moves a source into a destination directory that already exists, where rename fails
# or replaces only an empty one.
promote() {
  "${PYTHON:-python3}" - "$1" "$2" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
lock = dst + ".promoting"
try:
    os.mkdir(lock)
except FileExistsError:
    print(f"promotion of {dst} refused: {lock} is held, by another campaign command or left by a crash")
    sys.exit(1)
try:
    if os.path.lexists(dst):
        print(f"promotion of {dst} refused: it exists")
        sys.exit(1)
    os.rename(src, dst)
finally:
    os.rmdir(lock)
PY
}
