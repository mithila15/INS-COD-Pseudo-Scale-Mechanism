# Fast smoke-test config. Use this before full training.

has_test = False
deterministic = True
use_custom_worker_init = True
log_interval = 1
base_seed = 112358

# Debug run does not evaluate by default.
eval_interval = 0
save_best = False
monitor = "cod10k_te.wF"
monitor_mode = "max"
save_epoch_interval = 0

model = dict(
    mid_dim=32,
    hmu_groups=2,
    use_rfrm=True,
    use_gmffm=False,
    ual_weight=1.0,
    edge_weight=1.0,
    region_weight=0.5,
    deep_weight=0.25,
    kd_weight=0.0,
)

train = dict(
    batch_size=1,
    num_workers=0,
    use_amp=False,
    num_epochs=1,
    epoch_based=True,
    num_iters=None,
    lr=0.0001,
    grad_acc_step=1,
    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(weight_decay=0, diff_factor=0.1),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(num_iters=0, initial_coef=0.01, mode="linear"),
        mode="step",
        cfg=dict(milestones=100, gamma=0.1),
    ),
    bn=dict(freeze_status=True, freeze_affine=True, freeze_encoder=False),
    data=dict(shape=dict(h=384, w=384), names=["cod10k_tr", "camo_tr"]),
)

test = dict(batch_size=1, num_workers=0, clip_range=None, data=dict(shape=dict(h=384, w=384), names=[]))
