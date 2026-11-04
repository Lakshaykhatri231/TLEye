# Training loss by learning-rate range (script-generated)

Constant-guess BCE at the training TLE rate 0.395: 0.671. Values clearly below this mean the model learned something.

| run                                | lr_bin     |   logged_points |   mean_loss |   last_step |
|:-----------------------------------|:-----------|----------------:|------------:|------------:|
| run1 (no prefix)                   | <3e-5      |              31 |      0.6411 |        5000 |
| run1 (no prefix)                   | 3-6e-5     |              33 |      0.6662 |        4225 |
| run1 (no prefix)                   | 6e-5-1e-4  |              31 |      0.6692 |        3425 |
| run1 (no prefix)                   | 1-1.5e-4   |              38 |      0.6758 |        2675 |
| run1 (no prefix)                   | 1.5-2.1e-4 |              67 |      0.6804 |        1750 |
| run2 (prefix, aborted at step 750) | 3-6e-5     |               1 |      0.6788 |          25 |
| run2 (prefix, aborted at step 750) | 6e-5-1e-4  |               1 |      0.6969 |          50 |
| run2 (prefix, aborted at step 750) | 1-1.5e-4   |               1 |      0.6968 |          75 |
| run2 (prefix, aborted at step 750) | 1.5-2.1e-4 |              27 |      0.6885 |         750 |

Note: within one run lr and training progress are confounded (lr decays as training proceeds), so this table is suggestive, not proof of an lr effect.
