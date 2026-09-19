# Shims for running version 1's tracker under Octave

Three functions version 1's MATLAB environment provided and Octave does not. They are
kept HERE rather than as edits to `data/aewc_v2_pilot/v1_src`, so the archived source
stays byte-identical to what NCEI published and this directory carries everything that
is not version 1's own code. Put this directory FIRST on the Octave path.

Each is a reimplementation of documented behavior, not a guess, and each says what it
reproduces and where it could differ from MATLAB's.
