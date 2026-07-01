# African easterly waves and thunderstorm clusters: a plain-language summary

This note explains, without heavy jargon, what the reproduced and extended analyses show
and where they still fall short. Technical detail lives in the methods draft. This note is the
readable overview.

## The question

Every few days in summer, a ripple in the winds moves westward across West Africa. These
ripples are African easterly waves (AEWs). Riding along with them are large clusters of
thunderstorms, called mesoscale convective systems (MCS), which produce most of the Sahel's
rain and sometimes seed Atlantic hurricanes. The long-standing question is how tightly the
waves and the storms travel together, and where the storms sit relative to the wave.

## The old view and the new view

The original study answered a fixed-location version of the question. It sat at one point
(10 N, 0 E), waited for the wave to peak there, and averaged the storms around those dates.
That works, but it ties the answer to one spot and to strong events at that spot.

The new analysis follows the moving wave instead. Using an independent catalogue of wave
troughs (the low-pressure axis of each wave), it measures where the storms are relative to
the trough wherever the trough happens to be, then averages over thousands of trough
sightings. This trough-following view is closer to the physics, because the wave is a moving
thing, not a fixed location.

## What the storms do

Storms concentrate in and just west of the moving trough, which is the direction the wave
is heading. To check this is real and not just the fact that the Sahel is stormy anyway,
the analysis compares against a scrambled control in which the trough longitudes are
randomized. The real storm excess near the trough is about six times larger than the spread
of that control, so the clustering is tied to the trough, not to geography.

Two refinements sharpen the picture:

- Stronger waves organize storms more tightly. Splitting the waves by strength, the
  strongest third produce a sharper, higher storm peak at the trough than the weakest third.
- Birth and maturity differ. Where storm systems first appear (genesis) is spread broadly
  around the wave, but where mature systems pile up is narrowly focused in and just west of
  the trough. In plain terms, storms are born over a wide area near the wave, then grow up
  concentrated in the part of the wave that favors them.

## The chicken-and-egg problem, and an honest result

There is a real objection that any reviewer will raise. Waves and storms feed each other.
The storms release heat that helps maintain the wave, and the wave catalogue is built from a
weather model that has absorbed satellite data about those same storms. So finding storms at
the trough could partly be circular.

A first test of this looked at whether a wave that was already large two days earlier goes
on to produce more storms now. If earlier wave strength predicts later storms, the wave is
leading. The result was weak. In the averaged composite the strong-precursor waves have a
slightly higher storm peak (77 versus 64 in the same units as above), but a direct
point-by-point relationship between wave strength two days earlier and storm count now is
essentially flat. That is an honest negative. Wave strength changes slowly, so a two-day
lead mostly washes out, and the storm count in a small box at a single time is very noisy.

The takeaway is measured. The trough-following clustering itself is solid and repeatable.
The claim that the wave clearly leads and forces the storms is not settled by this test. The
next step is a timing test that asks whether the dry trough arrives before the storms build,
rather than whether the wave was merely large earlier. The wave catalogue is built from
curvature of the flow, which is less affected by the storms' own winds than plain vorticity
is, and that helps, but the timing test is what would answer the objection.

## Independence and repeatability

A separate goal was to remove any dependence on old or private data. The wind side now runs
on free public reanalysis (ERA5) through a filter that reproduces the original wave series
closely (correlation 0.92 with the original). The storm side can run on a storm catalogue
built in-house from free satellite infrared imagery (GridSat-B1). That in-house catalogue
matches the original ISCCP storm pattern about as well as, and slightly better than, a
published competitor. The original headline numbers reproduce exactly, including 272
composite dates at the published threshold. Everything is a tested Python package, and each
numerical step was independently cross-checked against the original code.

## What it adds up to

The strongest new result is the trough-following map of storms, with the split between where
storms are born and where they mature, and the finding that stronger waves organize storms
more sharply. The reproduction and the in-house storm tracker are the trustworthy foundation
underneath it. The open question is causality, and the practical next step is a timing-based
lead-lag test plus storm rainfall and environment, which would turn a strong description into
a process story.
