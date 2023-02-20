### Extended Hydra([Sehweg et al.](https://proceedings.neurips.cc/paper/2020/file/e3a72c791a69f87b05ea7742e04430ed-Paper.pdf)) implementation in Heracles experiments:

We fork the code from [Hydra](https://github.com/inspire-group/hydra) and extend with following additional supports to focus on the non-unform adversarially robust pruning:

1. Impose PGD adversaial training by adding `pgd_loss` in `./utils/adv.py`
2. Extend pruning with channel granularity in `./models/layers.py`.
3. Add reader function `parse_prune_stg` in `./utils/logging.py` to fetch a strategy from `strategies/$dataset$.json`.
4. Apply non-uniform compression strategy via the function `set_prune_rate_model` in `./utils/model.py`.
