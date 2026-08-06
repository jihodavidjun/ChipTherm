# Split Summary

| Partition | Families | Workloads per family | Samples |
|---|---:|---:|---:|
| Familiar training | 35 | 160 (1-160) | 5,600 |
| Familiar internal validation | 35 | 20 (161-180) | 700 |
| Familiar test | 35 | 20 (181-200) | 700 |
| Held-out validation | 5 | 200 | 1,000 |
| Held-out final test | 10 | 200 | 2,000 |

## Families

- Train: `f003 f004 f005 f006 f007 f009 f010 f011 f012 f013 f014 f020 f021 f022 f023 f024 f026 f027 f028 f029 f030 f032 f034 f036 f037 f039 f040 f041 f042 f045 f046 f047 f048 f049 f050`
- Validation: `f002 f008 f018 f025 f038`
- Final test: `f001 f015 f016 f017 f019 f031 f033 f035 f043 f044`

Held-out validation is not used by the source-response or package checkpoint
selectors. Checkpoint selection uses workload ordinals 161-180 from the 35
training families for package models. Held-out validation may be used only for
the subsequent, frozen post-training model comparison. Because isolated-source
data are representative packages rather than a 200-workload grid,
source-response uses a deterministic 28-family fit / 7-family
internal-selection split wholly inside the same 35 training-family set.
Final-test families are evaluation-only after model and checkpoint choices are
frozen.
