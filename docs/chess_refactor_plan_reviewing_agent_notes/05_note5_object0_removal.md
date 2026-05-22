# Note 5: Remove Visible object0 Cube

## Problem
`object0` is the legacy Fetch pick-and-place cube. In chess mode it is teleported to
`[2.0, 2.0, 0.015]` but remains visually visible as a dark grey box (`block_mat`, rgba 0.2 0.2 0.2 1).

## Fix
Change `block_mat` in `chess_env/assets/shared.xml` to alpha=0 (fully transparent).
Also change `puck_mat` to transparent (same issue).

```xml
<material name="block_mat" ... rgba="0.2 0.2 0.2 0"/>
<material name="puck_mat"  ... rgba="0.2 0.2 0.2 0"/>
```

The object0 body and joint remain in the XML (required by the Fetch env base class internals);
only the visual is made invisible.

## Validation
`python scripts/visualize.py` — no dark cube visible at [2, 2, 0.015].
