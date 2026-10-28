# Iso-FLOPs config for FSA-ZoomNeXt.
# Adjust __INPUT_SIZE after running scripts/search_iso_flops_fsa.sh.

has_test = True
deterministic = True
use_custom_worker_init = True
log_interval = 20
base_seed = 112358

eval_interval = 5
save_best = True
monitor = "cod10k_te.wF"
monitor_mode = "max"
save_epoch_interval = 0

__INPUT_SIZE = 640
__BATCHSIZE = 2
__NUM_EPOCHS = 150
__NUM_TR_SAMPLES = 3040 + 1000
__ITER_PER_EPOCH = __NUM_TR_SAMPLES // __BATCHSIZE
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH

model = dict(
    mid_dim=48,
    use_compensation=True,
    residual_refine_weight=0.5,
    ual_weight=1.0,
    edge_weight=1.0,
    region_weight=0.5,
    deep_weight=0.25,
    kd_weight=0.0,
)

train = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    use_amp=False,
    num_epochs=__NUM_EPOCHS,
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
        cfg=dict(milestones=int(__NUM_ITERS * 2 / 3), gamma=0.1),
    ),
    bn=dict(freeze_status=True, freeze_affine=True, freeze_encoder=False),
    data=dict(shape=dict(h=__INPUT_SIZE, w=__INPUT_SIZE), names=["cod10k_tr", "camo_tr"]),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    clip_range=None,
    data=dict(shape=dict(h=__INPUT_SIZE, w=__INPUT_SIZE), names=["chameleon", "camo_te", "cod10k_te", "nc4k"]),
)
