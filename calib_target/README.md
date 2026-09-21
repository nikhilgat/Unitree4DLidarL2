# calib_target - checkerboard extraction from Unitree L2 rosbags

Finds the checkerboard in each rosbag (`/unilidar/cloud`, mcap) and estimates `T_lidar_target`
(target -> LiDAR, 4x4).

```
python -m calib_target.extract C:\path\to\rosbags --out results          # all bags
python -m calib_target.extract rosbags\my_lidar_bag3 --out results       # one bag
  --cols 8 --rows 6 --square 0.095 --frames 200
```

Per bag it writes `results/<bag>/target.json` (transform, its inverse, quality metrics, walk curve),
`board.ply` (on-board points, walk-corrected, intensity as grey) and `detection.png`; plus
`results/summary.csv`. Check `accepted` and `detection.png` before trusting a bag.

## Method
1. **Accumulate** all frames of the (static) bag; the L2 scan is non-repeating so density grows.
2. **Propose** planar segments by normal-consistent region growing (`segment.py`) that could be the sheet.
3. **Verify** each proposal
   - reference plane from the saturated (intensity 255 = white) returns;
   - **range-walk correction**: on this data dark returns are measured up to ~9 cm *nearer* than
     white ones on the same plane, as a smooth function of intensity. The curve is estimated per
     bag from the board itself and removed along each ray;
   - masked NCC of the checker template on the plane-frame intensity image over rotation x
     translation x polarity x scale, then continuous refinement on the raw points.
4. **Finalise**: plane refit on corrected on-board points; `T_lidar_target` with origin at the
   pattern centre, x along the 8-square side, y along the 6-square side, z = board normal facing
   the sensor.
5. **Plausibility gate** (`plausible`): connected on-plane region must fit the board, be filled,
   and have 20-70 % dark points; `accepted` also needs NCC >= 0.4.

## Known limitations / open points
- **180 deg ambiguity**: an even x even checker looks identical rotated 180 deg about its normal.
  `T_lidar_target_alt` is the other solution; pick with an external cue (e.g. which board edge is up).
- **Square size**: the fitted square size is ~106-123 mm, not the stated 95 mm, and grows as the
  board gets closer (see `scale`/`square_mm` in the summary). Origin of this is unresolved
  (different board vs. distance-dependent range scale) - measure a printed square with a ruler.
- Which plane is "true" (white vs dark returns) is not known without an external reference; the
  white (saturated) surface is used.
- Boards near the sensor's zenith, far away (>2 m) or sparsely sampled can score low (bag 9).
