# Experiment leaderboard

Lowest mean validation MAE among variants with identical training protocol and seed set. Test results never select a checkpoint or winner.

Separate ranks for every dataset, metric, protocol and observed seed set; exact ties share a competition rank. No composite accuracy score and no mixing hardware for inference costs.

Descriptive seed mean and sample standard deviation; not a significance test. A single seed is provisional. Without a plan, target seed completeness is unknown.

Contraction FLOPs = 2 x MACs. Estimated total FLOPs additionally approximate supported noncontraction operators; consult each profile's coverage. Different hardware/input/timing protocols have separate ranks.

The standalone `leaderboard.html` includes every metric and protocol. This Markdown view shows validation MAE, the selected test metric, and principal inference costs.

## Progress

total_runs: 8 | planned_runs: 8 | training_complete: 8 | evaluation_complete: 8 | profile_complete: 8 | not_started: 0 | training_incomplete: 0 | all_stages_complete: 8

## validation · MAE ↓

Cohort `b84e76d2cdd4` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | dysample__asff | 0.027864475 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__mean | 0.028252939 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__asff | 0.028549269 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | projection__mean | 0.02857957 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | dysample__sk | 0.028701664 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | dysample__mean | 0.028709384 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__sk | 0.02909378 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## validation · MAE ↓

Cohort `da7bd9e35ee5` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 0.028608166 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## camo · S ↑

Cohort `4d01ee2852c0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | projection__mean | 0.8412583 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | dysample__mean | 0.83775653 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__sk | 0.83609518 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | dysample__asff | 0.83607645 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | native_conv__asff | 0.83498732 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | native_conv__sk | 0.83441304 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__mean | 0.8334611 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "evaluation_count": 250,
  "evaluation_packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "evaluation_source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  },
  "geometry_protocol": {
    "alignment": "full-image normalized coordinates; registration is not verified",
    "ground_truth": "original bytes and resolution; no crop, resize or sample exclusion",
    "policy": "resize-prediction-to-gt",
    "prediction_grid": "original GT mask height/width",
    "prediction_resize": "bilinear; align_corners=False; after model-resolution min-max",
    "version": 1
  },
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "metric_protocol": "official_py_sod_metrics_uint8_v2",
  "metric_protocol_details": {
    "Fbeta": {
      "adaptive_threshold": "min(2 * mean(prepared prediction), 1)",
      "beta_squared": 0.3,
      "empty_masks": "unaltered PySODMetrics FmeasureV2 conventions (zero denominator -> zero score)",
      "threshold_curve": "256 integer thresholds [0,255], >= comparison after library float-to-uint8 conversion"
    },
    "aggregation": "equal weight per image within each dataset; E/Fbeta mean and max taken AFTER averaging threshold curves across images",
    "binary_empty_masks": {
      "BER": "0.5 * (FNR + FPR), range [0,1]; an absent GT class contributes zero error",
      "BalancedAccuracy": "1 - BER",
      "overlap": "IoU, Dice/F1 and BoundaryIoU are 1 when both masks are empty; 0 when only one is empty",
      "precision": "no predicted foreground -> 1 if GT also empty, otherwise 0",
      "recall": "no GT foreground -> 1; FNR -> 0",
      "specificity": "no GT background -> 1; FPR -> 0"
    },
    "binary_threshold": {
      "comparison": ">=",
      "input": "PySODMetrics prepared float64 map",
      "value": 0.5
    },
    "boundary": {
      "aggregation": "image macro mean; semantic foreground mask, not COCO instance AP",
      "dilation_ratio": 0.02,
      "empty_union": 1.0,
      "operation": "inner boundary from 3x3 erosion repeated radius times; zero padding includes image borders",
      "radius": "max(1, round(.02 * sqrt(height**2 + width**2)))"
    },
    "input": "saved uint8 probability map: floor(255 * resized prediction); library prepare_data maps nonconstant uint8 maps to [0,1], constant maps divided by 255; GT is input mask>0",
    "per_image_maxima": "per-image E_max/Fbeta_max are diagnostic; their mean is NOT dataset E_max/Fbeta_max",
    "pysodmetrics_version": "1.6.2",
    "schema_version": 2,
    "sources": {
      "Boundary_IoU": "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
      "Boundary_IoU_author_code": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py",
      "Emeasure": "https://www.ijcai.org/proceedings/2018/97",
      "Fmeasure": "https://doi.org/10.1109/CVPR.2009.5206596",
      "PySODMetrics": "https://github.com/lartpang/PySODMetrics",
      "Smeasure": "https://openaccess.thecvf.com/content_ICCV_2017/html/Fan_Structure-Measure_A_New_ICCV_2017_paper.html",
      "weighted_Fmeasure": "https://openaccess.thecvf.com/content_cvpr_2014/html/Margolin_How_to_Evaluate_2014_CVPR_paper.html"
    }
  },
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "prediction_protocol": "sigmoid; per-image min-max at model resolution; bilinear resize to original mask; mask>0",
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## camo · S ↑

Cohort `9bb23dc60be8` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 0.83272737 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "evaluation_count": 250,
  "evaluation_packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "evaluation_source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  },
  "geometry_protocol": {
    "alignment": "full-image normalized coordinates; registration is not verified",
    "ground_truth": "original bytes and resolution; no crop, resize or sample exclusion",
    "policy": "resize-prediction-to-gt",
    "prediction_grid": "original GT mask height/width",
    "prediction_resize": "bilinear; align_corners=False; after model-resolution min-max",
    "version": 1
  },
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "metric_protocol": "official_py_sod_metrics_uint8_v2",
  "metric_protocol_details": {
    "Fbeta": {
      "adaptive_threshold": "min(2 * mean(prepared prediction), 1)",
      "beta_squared": 0.3,
      "empty_masks": "unaltered PySODMetrics FmeasureV2 conventions (zero denominator -> zero score)",
      "threshold_curve": "256 integer thresholds [0,255], >= comparison after library float-to-uint8 conversion"
    },
    "aggregation": "equal weight per image within each dataset; E/Fbeta mean and max taken AFTER averaging threshold curves across images",
    "binary_empty_masks": {
      "BER": "0.5 * (FNR + FPR), range [0,1]; an absent GT class contributes zero error",
      "BalancedAccuracy": "1 - BER",
      "overlap": "IoU, Dice/F1 and BoundaryIoU are 1 when both masks are empty; 0 when only one is empty",
      "precision": "no predicted foreground -> 1 if GT also empty, otherwise 0",
      "recall": "no GT foreground -> 1; FNR -> 0",
      "specificity": "no GT background -> 1; FPR -> 0"
    },
    "binary_threshold": {
      "comparison": ">=",
      "input": "PySODMetrics prepared float64 map",
      "value": 0.5
    },
    "boundary": {
      "aggregation": "image macro mean; semantic foreground mask, not COCO instance AP",
      "dilation_ratio": 0.02,
      "empty_union": 1.0,
      "operation": "inner boundary from 3x3 erosion repeated radius times; zero padding includes image borders",
      "radius": "max(1, round(.02 * sqrt(height**2 + width**2)))"
    },
    "input": "saved uint8 probability map: floor(255 * resized prediction); library prepare_data maps nonconstant uint8 maps to [0,1], constant maps divided by 255; GT is input mask>0",
    "per_image_maxima": "per-image E_max/Fbeta_max are diagnostic; their mean is NOT dataset E_max/Fbeta_max",
    "pysodmetrics_version": "1.6.2",
    "schema_version": 2,
    "sources": {
      "Boundary_IoU": "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
      "Boundary_IoU_author_code": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py",
      "Emeasure": "https://www.ijcai.org/proceedings/2018/97",
      "Fmeasure": "https://doi.org/10.1109/CVPR.2009.5206596",
      "PySODMetrics": "https://github.com/lartpang/PySODMetrics",
      "Smeasure": "https://openaccess.thecvf.com/content_ICCV_2017/html/Fan_Structure-Measure_A_New_ICCV_2017_paper.html",
      "weighted_Fmeasure": "https://openaccess.thecvf.com/content_cvpr_2014/html/Margolin_How_to_Evaluate_2014_CVPR_paper.html"
    }
  },
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "prediction_protocol": "sigmoid; per-image min-max at model resolution; bilinear resize to original mask; mask>0",
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## chameleon · S ↑

Cohort `20c2614abd95` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 0.90048986 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "evaluation_count": 76,
  "evaluation_packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "evaluation_source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  },
  "geometry_protocol": {
    "alignment": "full-image normalized coordinates; registration is not verified",
    "ground_truth": "original bytes and resolution; no crop, resize or sample exclusion",
    "policy": "resize-prediction-to-gt",
    "prediction_grid": "original GT mask height/width",
    "prediction_resize": "bilinear; align_corners=False; after model-resolution min-max",
    "version": 1
  },
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "metric_protocol": "official_py_sod_metrics_uint8_v2",
  "metric_protocol_details": {
    "Fbeta": {
      "adaptive_threshold": "min(2 * mean(prepared prediction), 1)",
      "beta_squared": 0.3,
      "empty_masks": "unaltered PySODMetrics FmeasureV2 conventions (zero denominator -> zero score)",
      "threshold_curve": "256 integer thresholds [0,255], >= comparison after library float-to-uint8 conversion"
    },
    "aggregation": "equal weight per image within each dataset; E/Fbeta mean and max taken AFTER averaging threshold curves across images",
    "binary_empty_masks": {
      "BER": "0.5 * (FNR + FPR), range [0,1]; an absent GT class contributes zero error",
      "BalancedAccuracy": "1 - BER",
      "overlap": "IoU, Dice/F1 and BoundaryIoU are 1 when both masks are empty; 0 when only one is empty",
      "precision": "no predicted foreground -> 1 if GT also empty, otherwise 0",
      "recall": "no GT foreground -> 1; FNR -> 0",
      "specificity": "no GT background -> 1; FPR -> 0"
    },
    "binary_threshold": {
      "comparison": ">=",
      "input": "PySODMetrics prepared float64 map",
      "value": 0.5
    },
    "boundary": {
      "aggregation": "image macro mean; semantic foreground mask, not COCO instance AP",
      "dilation_ratio": 0.02,
      "empty_union": 1.0,
      "operation": "inner boundary from 3x3 erosion repeated radius times; zero padding includes image borders",
      "radius": "max(1, round(.02 * sqrt(height**2 + width**2)))"
    },
    "input": "saved uint8 probability map: floor(255 * resized prediction); library prepare_data maps nonconstant uint8 maps to [0,1], constant maps divided by 255; GT is input mask>0",
    "per_image_maxima": "per-image E_max/Fbeta_max are diagnostic; their mean is NOT dataset E_max/Fbeta_max",
    "pysodmetrics_version": "1.6.2",
    "schema_version": 2,
    "sources": {
      "Boundary_IoU": "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
      "Boundary_IoU_author_code": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py",
      "Emeasure": "https://www.ijcai.org/proceedings/2018/97",
      "Fmeasure": "https://doi.org/10.1109/CVPR.2009.5206596",
      "PySODMetrics": "https://github.com/lartpang/PySODMetrics",
      "Smeasure": "https://openaccess.thecvf.com/content_ICCV_2017/html/Fan_Structure-Measure_A_New_ICCV_2017_paper.html",
      "weighted_Fmeasure": "https://openaccess.thecvf.com/content_cvpr_2014/html/Margolin_How_to_Evaluate_2014_CVPR_paper.html"
    }
  },
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "prediction_protocol": "sigmoid; per-image min-max at model resolution; bilinear resize to original mask; mask>0",
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## chameleon · S ↑

Cohort `8e3878420234` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | native_conv__sk | 0.90663642 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__asff | 0.90424632 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__mean | 0.90413162 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | native_conv__mean | 0.90159444 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | dysample__asff | 0.90077289 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | projection__mean | 0.90024525 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | dysample__sk | 0.89854446 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "evaluation_count": 76,
  "evaluation_packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "evaluation_source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  },
  "geometry_protocol": {
    "alignment": "full-image normalized coordinates; registration is not verified",
    "ground_truth": "original bytes and resolution; no crop, resize or sample exclusion",
    "policy": "resize-prediction-to-gt",
    "prediction_grid": "original GT mask height/width",
    "prediction_resize": "bilinear; align_corners=False; after model-resolution min-max",
    "version": 1
  },
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "metric_protocol": "official_py_sod_metrics_uint8_v2",
  "metric_protocol_details": {
    "Fbeta": {
      "adaptive_threshold": "min(2 * mean(prepared prediction), 1)",
      "beta_squared": 0.3,
      "empty_masks": "unaltered PySODMetrics FmeasureV2 conventions (zero denominator -> zero score)",
      "threshold_curve": "256 integer thresholds [0,255], >= comparison after library float-to-uint8 conversion"
    },
    "aggregation": "equal weight per image within each dataset; E/Fbeta mean and max taken AFTER averaging threshold curves across images",
    "binary_empty_masks": {
      "BER": "0.5 * (FNR + FPR), range [0,1]; an absent GT class contributes zero error",
      "BalancedAccuracy": "1 - BER",
      "overlap": "IoU, Dice/F1 and BoundaryIoU are 1 when both masks are empty; 0 when only one is empty",
      "precision": "no predicted foreground -> 1 if GT also empty, otherwise 0",
      "recall": "no GT foreground -> 1; FNR -> 0",
      "specificity": "no GT background -> 1; FPR -> 0"
    },
    "binary_threshold": {
      "comparison": ">=",
      "input": "PySODMetrics prepared float64 map",
      "value": 0.5
    },
    "boundary": {
      "aggregation": "image macro mean; semantic foreground mask, not COCO instance AP",
      "dilation_ratio": 0.02,
      "empty_union": 1.0,
      "operation": "inner boundary from 3x3 erosion repeated radius times; zero padding includes image borders",
      "radius": "max(1, round(.02 * sqrt(height**2 + width**2)))"
    },
    "input": "saved uint8 probability map: floor(255 * resized prediction); library prepare_data maps nonconstant uint8 maps to [0,1], constant maps divided by 255; GT is input mask>0",
    "per_image_maxima": "per-image E_max/Fbeta_max are diagnostic; their mean is NOT dataset E_max/Fbeta_max",
    "pysodmetrics_version": "1.6.2",
    "schema_version": 2,
    "sources": {
      "Boundary_IoU": "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
      "Boundary_IoU_author_code": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py",
      "Emeasure": "https://www.ijcai.org/proceedings/2018/97",
      "Fmeasure": "https://doi.org/10.1109/CVPR.2009.5206596",
      "PySODMetrics": "https://github.com/lartpang/PySODMetrics",
      "Smeasure": "https://openaccess.thecvf.com/content_ICCV_2017/html/Fan_Structure-Measure_A_New_ICCV_2017_paper.html",
      "weighted_Fmeasure": "https://openaccess.thecvf.com/content_cvpr_2014/html/Margolin_How_to_Evaluate_2014_CVPR_paper.html"
    }
  },
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "prediction_protocol": "sigmoid; per-image min-max at model resolution; bilinear resize to original mask; mask>0",
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## nc4k · S ↑

Cohort `2df3df475a3e` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | dysample__sk | 0.87466747 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__mean | 0.87330593 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__mean | 0.87276041 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | dysample__asff | 0.87274598 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | projection__mean | 0.87213169 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | native_conv__sk | 0.87148111 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__asff | 0.8707242 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "evaluation_count": 4121,
  "evaluation_packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "evaluation_source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  },
  "geometry_protocol": {
    "alignment": "full-image normalized coordinates; registration is not verified",
    "ground_truth": "original bytes and resolution; no crop, resize or sample exclusion",
    "policy": "resize-prediction-to-gt",
    "prediction_grid": "original GT mask height/width",
    "prediction_resize": "bilinear; align_corners=False; after model-resolution min-max",
    "version": 1
  },
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "metric_protocol": "official_py_sod_metrics_uint8_v2",
  "metric_protocol_details": {
    "Fbeta": {
      "adaptive_threshold": "min(2 * mean(prepared prediction), 1)",
      "beta_squared": 0.3,
      "empty_masks": "unaltered PySODMetrics FmeasureV2 conventions (zero denominator -> zero score)",
      "threshold_curve": "256 integer thresholds [0,255], >= comparison after library float-to-uint8 conversion"
    },
    "aggregation": "equal weight per image within each dataset; E/Fbeta mean and max taken AFTER averaging threshold curves across images",
    "binary_empty_masks": {
      "BER": "0.5 * (FNR + FPR), range [0,1]; an absent GT class contributes zero error",
      "BalancedAccuracy": "1 - BER",
      "overlap": "IoU, Dice/F1 and BoundaryIoU are 1 when both masks are empty; 0 when only one is empty",
      "precision": "no predicted foreground -> 1 if GT also empty, otherwise 0",
      "recall": "no GT foreground -> 1; FNR -> 0",
      "specificity": "no GT background -> 1; FPR -> 0"
    },
    "binary_threshold": {
      "comparison": ">=",
      "input": "PySODMetrics prepared float64 map",
      "value": 0.5
    },
    "boundary": {
      "aggregation": "image macro mean; semantic foreground mask, not COCO instance AP",
      "dilation_ratio": 0.02,
      "empty_union": 1.0,
      "operation": "inner boundary from 3x3 erosion repeated radius times; zero padding includes image borders",
      "radius": "max(1, round(.02 * sqrt(height**2 + width**2)))"
    },
    "input": "saved uint8 probability map: floor(255 * resized prediction); library prepare_data maps nonconstant uint8 maps to [0,1], constant maps divided by 255; GT is input mask>0",
    "per_image_maxima": "per-image E_max/Fbeta_max are diagnostic; their mean is NOT dataset E_max/Fbeta_max",
    "pysodmetrics_version": "1.6.2",
    "schema_version": 2,
    "sources": {
      "Boundary_IoU": "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
      "Boundary_IoU_author_code": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py",
      "Emeasure": "https://www.ijcai.org/proceedings/2018/97",
      "Fmeasure": "https://doi.org/10.1109/CVPR.2009.5206596",
      "PySODMetrics": "https://github.com/lartpang/PySODMetrics",
      "Smeasure": "https://openaccess.thecvf.com/content_ICCV_2017/html/Fan_Structure-Measure_A_New_ICCV_2017_paper.html",
      "weighted_Fmeasure": "https://openaccess.thecvf.com/content_cvpr_2014/html/Margolin_How_to_Evaluate_2014_CVPR_paper.html"
    }
  },
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "prediction_protocol": "sigmoid; per-image min-max at model resolution; bilinear resize to original mask; mask>0",
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## nc4k · S ↑

Cohort `70a0b88227f2` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 0.87454115 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "evaluation_count": 4121,
  "evaluation_packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "evaluation_source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  },
  "geometry_protocol": {
    "alignment": "full-image normalized coordinates; registration is not verified",
    "ground_truth": "original bytes and resolution; no crop, resize or sample exclusion",
    "policy": "resize-prediction-to-gt",
    "prediction_grid": "original GT mask height/width",
    "prediction_resize": "bilinear; align_corners=False; after model-resolution min-max",
    "version": 1
  },
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "metric_protocol": "official_py_sod_metrics_uint8_v2",
  "metric_protocol_details": {
    "Fbeta": {
      "adaptive_threshold": "min(2 * mean(prepared prediction), 1)",
      "beta_squared": 0.3,
      "empty_masks": "unaltered PySODMetrics FmeasureV2 conventions (zero denominator -> zero score)",
      "threshold_curve": "256 integer thresholds [0,255], >= comparison after library float-to-uint8 conversion"
    },
    "aggregation": "equal weight per image within each dataset; E/Fbeta mean and max taken AFTER averaging threshold curves across images",
    "binary_empty_masks": {
      "BER": "0.5 * (FNR + FPR), range [0,1]; an absent GT class contributes zero error",
      "BalancedAccuracy": "1 - BER",
      "overlap": "IoU, Dice/F1 and BoundaryIoU are 1 when both masks are empty; 0 when only one is empty",
      "precision": "no predicted foreground -> 1 if GT also empty, otherwise 0",
      "recall": "no GT foreground -> 1; FNR -> 0",
      "specificity": "no GT background -> 1; FPR -> 0"
    },
    "binary_threshold": {
      "comparison": ">=",
      "input": "PySODMetrics prepared float64 map",
      "value": 0.5
    },
    "boundary": {
      "aggregation": "image macro mean; semantic foreground mask, not COCO instance AP",
      "dilation_ratio": 0.02,
      "empty_union": 1.0,
      "operation": "inner boundary from 3x3 erosion repeated radius times; zero padding includes image borders",
      "radius": "max(1, round(.02 * sqrt(height**2 + width**2)))"
    },
    "input": "saved uint8 probability map: floor(255 * resized prediction); library prepare_data maps nonconstant uint8 maps to [0,1], constant maps divided by 255; GT is input mask>0",
    "per_image_maxima": "per-image E_max/Fbeta_max are diagnostic; their mean is NOT dataset E_max/Fbeta_max",
    "pysodmetrics_version": "1.6.2",
    "schema_version": 2,
    "sources": {
      "Boundary_IoU": "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
      "Boundary_IoU_author_code": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py",
      "Emeasure": "https://www.ijcai.org/proceedings/2018/97",
      "Fmeasure": "https://doi.org/10.1109/CVPR.2009.5206596",
      "PySODMetrics": "https://github.com/lartpang/PySODMetrics",
      "Smeasure": "https://openaccess.thecvf.com/content_ICCV_2017/html/Fan_Structure-Measure_A_New_ICCV_2017_paper.html",
      "weighted_Fmeasure": "https://openaccess.thecvf.com/content_cvpr_2014/html/Margolin_How_to_Evaluate_2014_CVPR_paper.html"
    }
  },
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "prediction_protocol": "sigmoid; per-image min-max at model resolution; bilinear resize to original mask; mask>0",
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## inference · Peak CUDA allocated bytes ↓

Cohort `560d76bb8f24` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 2.1546138e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## inference · Peak CUDA allocated bytes ↓

Cohort `5bac70efe8b0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | projection__mean | 2.1556787e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | dysample__sk | 2.1559859e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__asff | 2.1560883e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | native_conv__sk | 2.1567846e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | native_conv__asff | 2.156887e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | dysample__mean | 2.1607782e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__mean | 2.161577e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## inference · Estimated total FLOPs / image ↓

Cohort `560d76bb8f24` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 2.8561004e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## inference · Estimated total FLOPs / image ↓

Cohort `5bac70efe8b0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | dysample__mean | 2.8627591e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | dysample__sk | 2.863055e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__asff | 2.8694788e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | projection__mean | 2.8729622e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | native_conv__mean | 2.8789843e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | native_conv__sk | 2.8792802e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__asff | 2.885704e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## inference · Contraction FLOPs / image (2 x MACs) ↓

Cohort `560d76bb8f24` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 2.7968634e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## inference · Contraction FLOPs / image (2 x MACs) ↓

Cohort `5bac70efe8b0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | dysample__mean | 2.8006235e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | dysample__sk | 2.8006254e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__asff | 2.8066162e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | projection__mean | 2.8102588e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | native_conv__mean | 2.8136077e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | native_conv__sk | 2.8136095e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__asff | 2.8196004e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## inference · Images / second ↑

Cohort `560d76bb8f24` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 159.29032 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## inference · Images / second ↑

Cohort `5bac70efe8b0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | projection__mean | 145.23363 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__mean | 141.99574 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__mean | 138.49077 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | native_conv__sk | 135.42669 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | dysample__sk | 134.18824 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | dysample__asff | 128.63766 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__asff | 126.25363 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## inference · Median inference latency (ms) ↓

Cohort `560d76bb8f24` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 6.273456 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## inference · Median inference latency (ms) ↓

Cohort `5bac70efe8b0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | projection__mean | 6.8840975 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__mean | 7.037527 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__mean | 7.222933 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | native_conv__sk | 7.3863219 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | dysample__sk | 7.446824 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | dysample__asff | 7.7768683 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__asff | 7.9187095 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## inference · MACs / image ↓

Cohort `560d76bb8f24` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 1.3984317e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## inference · MACs / image ↓

Cohort `5bac70efe8b0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | dysample__mean | 1.4003118e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | dysample__sk | 1.4003127e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__asff | 1.4033081e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | projection__mean | 1.4051294e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 5 | native_conv__mean | 1.4068039e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | native_conv__sk | 1.4068048e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__asff | 1.4098002e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>

## inference · parameters ↓

Cohort `560d76bb8f24` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 25318163 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "601266ab27130b43a6be61d303e212ee7071356e0d8044af1de96bed821e3e00",
    "fsa_study/engine.py": "9cc2686c8b897d609cb255e129bed994d06c00d9220dea9be3dba82b77492a73",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "a37de749a89f90e752e8c1af54c12a06c0db8a1e44be68a970a8a4d7a5b01b54",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "run_plan.py": "33efd25bf4f0a4d051e9ad33525d8cf0a411cc52bc27fa294eef5b18999f4d49",
    "run_study.py": "85b3eb78e6f3c78d442a2dc5af7a6c5bbf07ca2a534c7ab67bf9f1b88feed2eb",
    "summarize_study.py": "ceb14a6013d3ccb655b267281fca9397f95c0044d6906d4f67d0034bfd863210",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "19bd954022af9d7e2daceb5481de7dced43e9593da48f2eae418b4fec4d76ad7",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "3eca79f1741888673ce7ed2dba9b333738777e58cb975164668d3d140e2b5751"
  }
}
```

</details>

## inference · parameters ↓

Cohort `5bac70efe8b0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | dysample__mean | 25324435 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | dysample__sk | 25334275 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | dysample__asff | 25334623 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | native_conv__mean | 25341587 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 4 | projection__mean | 25341587 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 6 | native_conv__sk | 25351427 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 7 | native_conv__asff | 25351775 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "minimum sample-weighted validation float MAE; strict improvement; no test access",
  "complexity": {
    "all_observed_ops_classified": true,
    "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
    "scope": "one inference forward including encoder, generator, utilizer and all output heads",
    "version": "aten_analytical_v1"
  },
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "loss_weights": {
    "deep": 0.25,
    "edge": 1,
    "region": 0.5,
    "seg": 1,
    "ual": 1
  },
  "lr": 0.0001,
  "manifest_sha256": "21a61141c06e7a2cd3cd36d90d71aec26977757efd56c4df985f21cb33dbf08a",
  "packages": {
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "numpy": "2.3.1",
    "pysodmetrics": "1.6.2",
    "timm": "1.0.30",
    "torch": "2.11.0",
    "torchvision": "0.26.0"
  },
  "profile": {
    "batch_size": 1,
    "complexity_protocol": {
      "convention": "MACs: dense convolution and tensor contractions; FLOPs: 2*MACs; other operations estimated separately",
      "convolution": "dense padded kernels; transposed kernels counted at input spatial positions, including subsequently cropped positions",
      "estimates": "scalar arithmetic/comparison/transcendental operations each cost 1; interpolation excludes internal coordinate/index computation",
      "exclusions": "IO, transfer, memory movement, integer indexing, backward and optimizer; no claim of hardware-exact FLOPs",
      "scope": "one inference forward including encoder, generator, utilizer and all output heads",
      "version": "aten_analytical_v1"
    },
    "device": "cuda:0",
    "dtype": "float32",
    "input": "synthetic_uniform_RGB",
    "repeats": 50,
    "seed": 3407,
    "size": 384,
    "threads": 2,
    "timed_scope": "device tensor to all heads, including deep/region/edge; excludes IO/transfer",
    "warmup": 10
  },
  "profile_environment": {
    "device": "NVIDIA RTX PRO 5000 Blackwell",
    "python": "3.12.13",
    "torch": "2.11.0+cu130"
  },
  "profile_input_sha256": null,
  "size": 384,
  "source_sha256": {
    "fsa_study/__init__.py": "58a6b4c16acd83a89ad26ea7bbeb4f67272f4860484c5f949e7f2e947bce68b9",
    "fsa_study/complexity.py": "1ed6d4350acbdb171814f34b0b5afdb72e85663138a4c402fcacc5287ebc68de",
    "fsa_study/data.py": "dd432d5fe34a32de58ab8b0afe96e0b48e661c8460dda3a15cb394b8c110498c",
    "fsa_study/engine.py": "7c160478f6d6fdcbcb95484c527275e658700dc81a65af0b41b7c63258b12e12",
    "fsa_study/generators.py": "7115e7dbde8987b06bdfc1b12b5119619c722aea662aa3ccb3050c2b02b94e9e",
    "fsa_study/leaderboard.py": "e65630d3716b4e4638b1d5491ac4d2ed0726a4e27acf84bee64a8222f4a303d1",
    "fsa_study/metrics.py": "f9093f745df73b5df8e0b9da3a91ff54441f47094cac9279f4c4be869ff68e22",
    "fsa_study/model.py": "a33b41c3193ec5dfb88db9faff97a0aada2c2efd6e39b72e99073acbb449599e",
    "fsa_study/pcd.py": "5f14b1ca736703ab1c70478befddee42a2a3aaed1cf9d14fb957ca621f00da8f",
    "fsa_study/utilizers.py": "f96033a6df7b9655f998a4007863f98972099c4235224a6ca21ef8bef40c6705",
    "plan_study.py": "d8554285bd8dbaf2cba90239c4032e46911115790b3f3c3cbd452076a491c5dd",
    "profile_study.py": "a94c5e8c387f48b83bc719e27b229a16e674d296fa0a7e507836d493bd740387",
    "recover_evaluation.py": "f1bb77f989703c98c415f5e67c0d64093d16a229a08a2d465d60ddd37a5ee4f8",
    "run_plan.py": "6a3df742b224f6e7d827bcaa7c18a0a3f91d40fed971c3af2c68ae4d144eb350",
    "run_study.py": "98eae9155e52388ffa948cd39bd81980e8409f4e82e2a5e842e24f68ff2229fd",
    "summarize_study.py": "944e02de20e4982dd3048a409429d4661794b9fe10e077da6f7880fcda4dca97",
    "tests/conftest.py": "b7a46249b7524cd2506620649047e80abd7521287e8e45c6fd5b1611ab849f54",
    "tests/test_complexity.py": "f1757892bc681ffcb656cf0f11f84a7cee7631069e19a79de66e7d7839c1fe99",
    "tests/test_data.py": "25e4728ebbd3d817bd6c9d90c9711c01160599baeb80a9d78bb18023dd200b6d",
    "tests/test_generators.py": "2457d58ddae7086e969f717ece2a0afbbdf332d6070b8987e8ae82b90cf3979c",
    "tests/test_geometry.py": "d76e17b43ec0f30ad738dbba4907b1364684dae946bb7ac3ba670e96231cdc21",
    "tests/test_leaderboard.py": "fad20dd9bc08a51ba57447c71aa0022a88008f4f9556568bd2236466c6c8c820",
    "tests/test_metrics.py": "c5f5c68555644f12a5d60ab9c9d218bbc3c1046d65f0ef76517e21e99135a37a",
    "tests/test_model.py": "3e0904b01a5a26a817ba07f4deb17f1bd680abe7a6f1beb90f70e7825e36200d",
    "tests/test_recover_evaluation.py": "b96844229b7a1269060ac80e161d7cf1eaf05a3f4613f9c975435950d9060f7a",
    "tests/test_reference_operators.py": "92882841e13533e70993317b2fca55689dbede2b8b2a51b398676f6d39f35e81",
    "tests/test_run_plan.py": "e708b5481067cacfae85eafc7149ddbbe170e58f5468abbaf332d3849f71f69d",
    "tests/test_utilizers.py": "b2a6539bef179232286451b9bbc8c348e16647144649088c825ac05e9b9ea944",
    "tests/test_workflow.py": "4320bf6043b0e524afa8a12fb3b32873a314d71f77fdaff16f9eccd7196748d8"
  }
}
```

</details>
