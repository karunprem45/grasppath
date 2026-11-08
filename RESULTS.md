# GraspPath results log

Fill this in from your own runs. Only numbers in this file go on the resume.

## 1. FK check (`python -m scripts.check_fk_vs_sim`)
- Date run:
- Position error (mm): mean ____ , max ____
- Rotation error (deg): mean ____ , max ____

## 2. Tests (`pytest -v`)
- Tests passed: ____ / ____

## 3. Pick-and-place trials (`python -m scripts.run_trials --trials 50 --seed 0`)
- Trials:
- Success rate: ____ / ____
- Placement error on successes (mm): median ____ , max ____
- Mean planning time per trial (s):
- Shortcutting cut joint-space path length by ____ %
- Failures by state:

## 4. One thing that broke and how I fixed it
(This is your interview story. Write it in your own words.)

## 5. Plot
- `python -m scripts.plot_trajectory` saved results/trajectory.png: yes / no
