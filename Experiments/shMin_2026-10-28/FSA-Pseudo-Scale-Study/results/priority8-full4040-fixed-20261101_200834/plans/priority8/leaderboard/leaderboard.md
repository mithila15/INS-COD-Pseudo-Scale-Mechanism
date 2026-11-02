# Experiment leaderboard

Holdout runs: lowest mean validation MAE among variants with identical training protocol and seed set. Full-train runs: fixed final epoch checkpoint, without validation selection or fabricated validation scores. Test results never select a checkpoint or winner.

Separate ranks for every dataset, metric, protocol and observed seed set; exact ties share a competition rank. No composite accuracy score and no mixing hardware for inference costs.

Descriptive seed mean and sample standard deviation; not a significance test. A single seed is provisional. Without a plan, target seed completeness is unknown.

Contraction FLOPs = 2 x MACs. Estimated total FLOPs additionally approximate supported noncontraction operators; consult each profile's coverage. Different hardware/input/timing protocols have separate ranks.

The standalone `leaderboard.html` includes every metric and protocol. This Markdown view shows validation MAE, the selected test metric, and principal inference costs.

## Progress

total_runs: 8 | planned_runs: 8 | training_complete: 3 | evaluation_complete: 3 | profile_complete: 3 | not_started: 5 | training_incomplete: 0 | all_stages_complete: 3 | execution_complete: 3 | benchmark_complete: 3

## Test benchmark coverage

Execution completion only covers the configured datasets. Benchmark completion requires all four standard test image counts for every run; counts alone do not prove image identities, split disjointness or metric correctness.

**Standard four-dataset coverage: 3/8 runs complete.**

| Dataset | Expected images | Observed counts per run | Evaluated runs | Runs with standard count |
| --- | ---: | --- | ---: | ---: |
| camo | 250 | 250 | 3/8 | 3/8 |
| chameleon | 76 | 76 | 3/8 | 3/8 |
| cod10k | 2026 | 2026 | 3/8 | 3/8 |
| nc4k | 4121 | 4121 | 3/8 | 3/8 |

Missing in one or more runs: camo, chameleon, cod10k, nc4k.

## Split overlap audit

Image path/byte-hash split overlap is separate from benchmark image counts. Acknowledged overlap scores include the affected test images and are not disjoint-test evidence. Unknown means the historical metric artifact has no verified overlap audit; matching four dataset counts does not establish disjointness.

**Evaluated runs: 0 disjoint, 3 overlapping, 0 unknown.**

| Experiment | Seed | Policy | Disjoint | Train–test images | Val–test images | Affected test images |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| single | 3407 | acknowledged-cod10k-v1 | False | 7 | 0 | 7 |
| native_conv__asff | 3407 | acknowledged-cod10k-v1 | False | 7 | 0 | 7 |
| native_conv__sk | 3407 | acknowledged-cod10k-v1 | False | 7 | 0 | 7 |

## camo · S ↑

Cohort `fdb63d851b8a` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | native_conv__asff | 0.8391877 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__sk | 0.83769063 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | single | 0.83650563 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
  "overlap_audit": {
    "affected_test_count": 7,
    "disjoint": false,
    "groups": [
      {
        "image_sha256": "1be584edf381ee65c4e802dedefb671742122d6db3f00af3b204d6b4754309d7",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-31-Gecko-1892",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-26-Chameleon-1694",
        "train_split": "train"
      },
      {
        "image_sha256": "311809538532626c3c07e95076c32130b7010d16cf5ae6e6d5624819ab1d70d0",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-23-Cat-1506",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-23-Cat-1506",
        "train_split": "train"
      },
      {
        "image_sha256": "58a9a7b96f9f5643139d33d73331c6f1851b599df7fa60a916e451ef01ca3afe",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-28-Deer-1762",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-28-Deer-1796",
        "train_split": "train"
      },
      {
        "image_sha256": "6368c9991b9cc8f2d45eb87bb172e2611154efaaa99113fc1fab0d1943276390",
        "test_id": "cod10k_test/COD10K-CAM-3-Flying-65-Owl-4633",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-3-Flying-53-Bird-3205",
        "train_split": "train"
      },
      {
        "image_sha256": "6988e9b908fb9a6c8157405f49dcd37586ef14f08d6374e33996c780dcbb67e0",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-32-Giraffe-1932",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-32-Giraffe-1930",
        "train_split": "train"
      },
      {
        "image_sha256": "e5f0ea8dec16c4b4ed178c58a9827cd946fa9e0e860e1d56ae769b50d59e6c4e",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-31-Gecko-1895",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-31-Gecko-1928",
        "train_split": "train"
      },
      {
        "image_sha256": "e60fced9bfb2eeb8df4b7b9dcce931d6537b7f4f8ff4e49296e4bd763897d27a",
        "test_id": "cod10k_test/COD10K-CAM-1-Aquatic-3-Crab-32",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-1-Aquatic-3-Crab-32",
        "train_split": "train"
      }
    ],
    "policy": "acknowledged-cod10k-v1",
    "schema_version": 1,
    "train_test_count": 7,
    "val_test_count": 0
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## chameleon · S ↑

Cohort `b96feb5829ae` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | native_conv__sk | 0.90742662 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__asff | 0.90348041 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | single | 0.90062112 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
  "overlap_audit": {
    "affected_test_count": 7,
    "disjoint": false,
    "groups": [
      {
        "image_sha256": "1be584edf381ee65c4e802dedefb671742122d6db3f00af3b204d6b4754309d7",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-31-Gecko-1892",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-26-Chameleon-1694",
        "train_split": "train"
      },
      {
        "image_sha256": "311809538532626c3c07e95076c32130b7010d16cf5ae6e6d5624819ab1d70d0",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-23-Cat-1506",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-23-Cat-1506",
        "train_split": "train"
      },
      {
        "image_sha256": "58a9a7b96f9f5643139d33d73331c6f1851b599df7fa60a916e451ef01ca3afe",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-28-Deer-1762",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-28-Deer-1796",
        "train_split": "train"
      },
      {
        "image_sha256": "6368c9991b9cc8f2d45eb87bb172e2611154efaaa99113fc1fab0d1943276390",
        "test_id": "cod10k_test/COD10K-CAM-3-Flying-65-Owl-4633",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-3-Flying-53-Bird-3205",
        "train_split": "train"
      },
      {
        "image_sha256": "6988e9b908fb9a6c8157405f49dcd37586ef14f08d6374e33996c780dcbb67e0",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-32-Giraffe-1932",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-32-Giraffe-1930",
        "train_split": "train"
      },
      {
        "image_sha256": "e5f0ea8dec16c4b4ed178c58a9827cd946fa9e0e860e1d56ae769b50d59e6c4e",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-31-Gecko-1895",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-31-Gecko-1928",
        "train_split": "train"
      },
      {
        "image_sha256": "e60fced9bfb2eeb8df4b7b9dcce931d6537b7f4f8ff4e49296e4bd763897d27a",
        "test_id": "cod10k_test/COD10K-CAM-1-Aquatic-3-Crab-32",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-1-Aquatic-3-Crab-32",
        "train_split": "train"
      }
    ],
    "policy": "acknowledged-cod10k-v1",
    "schema_version": 1,
    "train_test_count": 7,
    "val_test_count": 0
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## cod10k · S ↑

Cohort `09f9a03b1050` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | native_conv__asff | 0.85550535 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | single | 0.85513856 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__sk | 0.85504774 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
  "determinism": "warn",
  "edge_protocol": "5x5 morphological gradient after nearest mask resize",
  "epochs": 150,
  "eval_interval": 1,
  "evaluation_count": 2026,
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
  "overlap_audit": {
    "affected_test_count": 7,
    "disjoint": false,
    "groups": [
      {
        "image_sha256": "1be584edf381ee65c4e802dedefb671742122d6db3f00af3b204d6b4754309d7",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-31-Gecko-1892",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-26-Chameleon-1694",
        "train_split": "train"
      },
      {
        "image_sha256": "311809538532626c3c07e95076c32130b7010d16cf5ae6e6d5624819ab1d70d0",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-23-Cat-1506",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-23-Cat-1506",
        "train_split": "train"
      },
      {
        "image_sha256": "58a9a7b96f9f5643139d33d73331c6f1851b599df7fa60a916e451ef01ca3afe",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-28-Deer-1762",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-28-Deer-1796",
        "train_split": "train"
      },
      {
        "image_sha256": "6368c9991b9cc8f2d45eb87bb172e2611154efaaa99113fc1fab0d1943276390",
        "test_id": "cod10k_test/COD10K-CAM-3-Flying-65-Owl-4633",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-3-Flying-53-Bird-3205",
        "train_split": "train"
      },
      {
        "image_sha256": "6988e9b908fb9a6c8157405f49dcd37586ef14f08d6374e33996c780dcbb67e0",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-32-Giraffe-1932",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-32-Giraffe-1930",
        "train_split": "train"
      },
      {
        "image_sha256": "e5f0ea8dec16c4b4ed178c58a9827cd946fa9e0e860e1d56ae769b50d59e6c4e",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-31-Gecko-1895",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-31-Gecko-1928",
        "train_split": "train"
      },
      {
        "image_sha256": "e60fced9bfb2eeb8df4b7b9dcce931d6537b7f4f8ff4e49296e4bd763897d27a",
        "test_id": "cod10k_test/COD10K-CAM-1-Aquatic-3-Crab-32",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-1-Aquatic-3-Crab-32",
        "train_split": "train"
      }
    ],
    "policy": "acknowledged-cod10k-v1",
    "schema_version": 1,
    "train_test_count": 7,
    "val_test_count": 0
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## nc4k · S ↑

Cohort `c3e419452dd0` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 0.87244166 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__asff | 0.8718836 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__sk | 0.87149105 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
  "overlap_audit": {
    "affected_test_count": 7,
    "disjoint": false,
    "groups": [
      {
        "image_sha256": "1be584edf381ee65c4e802dedefb671742122d6db3f00af3b204d6b4754309d7",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-31-Gecko-1892",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-26-Chameleon-1694",
        "train_split": "train"
      },
      {
        "image_sha256": "311809538532626c3c07e95076c32130b7010d16cf5ae6e6d5624819ab1d70d0",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-23-Cat-1506",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-23-Cat-1506",
        "train_split": "train"
      },
      {
        "image_sha256": "58a9a7b96f9f5643139d33d73331c6f1851b599df7fa60a916e451ef01ca3afe",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-28-Deer-1762",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-28-Deer-1796",
        "train_split": "train"
      },
      {
        "image_sha256": "6368c9991b9cc8f2d45eb87bb172e2611154efaaa99113fc1fab0d1943276390",
        "test_id": "cod10k_test/COD10K-CAM-3-Flying-65-Owl-4633",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-3-Flying-53-Bird-3205",
        "train_split": "train"
      },
      {
        "image_sha256": "6988e9b908fb9a6c8157405f49dcd37586ef14f08d6374e33996c780dcbb67e0",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-32-Giraffe-1932",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-32-Giraffe-1930",
        "train_split": "train"
      },
      {
        "image_sha256": "e5f0ea8dec16c4b4ed178c58a9827cd946fa9e0e860e1d56ae769b50d59e6c4e",
        "test_id": "cod10k_test/COD10K-CAM-2-Terrestrial-31-Gecko-1895",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-2-Terrestrial-31-Gecko-1928",
        "train_split": "train"
      },
      {
        "image_sha256": "e60fced9bfb2eeb8df4b7b9dcce931d6537b7f4f8ff4e49296e4bd763897d27a",
        "test_id": "cod10k_test/COD10K-CAM-1-Aquatic-3-Crab-32",
        "test_split": "test",
        "train_id": "cod10k_train/COD10K-CAM-1-Aquatic-3-Crab-32",
        "train_split": "train"
      }
    ],
    "policy": "acknowledged-cod10k-v1",
    "schema_version": 1,
    "train_test_count": 7,
    "val_test_count": 0
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## inference · Peak CUDA allocated bytes ↓

Cohort `252397ca8e9d` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 2.1546138e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__sk | 2.1567846e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__asff | 2.156887e+08 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## inference · Estimated total FLOPs / image ↓

Cohort `252397ca8e9d` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 2.8561004e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__sk | 2.8792802e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__asff | 2.885704e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## inference · Contraction FLOPs / image (2 x MACs) ↓

Cohort `252397ca8e9d` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 2.7968634e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__sk | 2.8136095e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__asff | 2.8196004e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## inference · Images / second ↑

Cohort `252397ca8e9d` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 158.53405 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__sk | 134.29515 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__asff | 128.84568 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## inference · Median inference latency (ms) ↓

Cohort `252397ca8e9d` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 6.3071015 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__sk | 7.4474325 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__asff | 7.7615826 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## inference · MACs / image ↓

Cohort `252397ca8e9d` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 1.3984317e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__sk | 1.4068048e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__asff | 1.4098002e+10 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## inference · parameters ↓

Cohort `252397ca8e9d` · same observed seeds: [3407]. Ranks only compare rows in this cohort.

| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | single | 25318163 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 2 | native_conv__sk | 25351427 | — | 1 | [3407] | [3407] | provisional_single_seed |
| 3 | native_conv__asff | 25351775 | — | 1 | [3407] | [3407] | provisional_single_seed |

<details><summary>Comparison protocol</summary>

```json
{
  "augmentation": "stateless seeded horizontal flip p=.5, vertical flip p=.2; resized RGB/mask",
  "backbone_initialization": "local_checkpoint",
  "backbone_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
  "batch_size": 4,
  "channels": 48,
  "checkpoint_selection": "fixed final epoch; no validation or test selection",
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
  "manifest_sha256": "af2cd10f1a6db6f8c474493c751e860c40009025b35a6cc53d9baaa7f5f9aa7f",
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
    "complete_benchmark.py": "ba44b97b0d0dcdb080a0390a58caa5984195b4d0bac1b78f15ce2826d59397d1",
    "deploy/install_study.py": "b20d2ca52f364a2762eb2e8780d632c9d86cc020cdc7746624aa38af82eec3d6",
    "fsa_study/__init__.py": "5ae4aa55efd6789781ffec11f0ed02943ea5f8f4fa5c0c61408ccc354865f6e8",
    "fsa_study/benchmark.py": "a6ed3fca79ba864d11fcf303688e0bcef398c15a9dbc46cd9cfa03738ea41435",
    "fsa_study/complexity.py": "b8f44b7c984f2814f7099bbbe764cb3af8bc3c32958051329ebe11ecfda69565",
    "fsa_study/data.py": "6083f0f02835f694f8aeb051958de2902cde489feffd95e4fd147ed07202b32b",
    "fsa_study/engine.py": "c52e9906d30753fbde2932007ebe093fb833629a04903789112c50d2fa739eed",
    "fsa_study/generators.py": "9c98ee17c00d693d240e1a312af7e86f8aef9733d7b3a835a950be4e5841dc61",
    "fsa_study/leaderboard.py": "da6eb46e0ddd569b7e13e07effcd6b05abd32d7362ab0a1b46be1e34cd06f21d",
    "fsa_study/metrics.py": "a90a9182f37388a0203a35f42aa3449efb96eef4d988e5516ea12b9f1163ac52",
    "fsa_study/model.py": "1054143e21893105f356521fa248803d420a6e1c36ffbac7715a603fa692e5d8",
    "fsa_study/overlap.py": "30fc1c4cdf4f9ddd640a4cf46ca15032d59e4530340fb9e616a1dca171bf88d9",
    "fsa_study/training_protocol.py": "78125f601835fb836bf2eb2276b2af18c2b3d0018f5d2f1b85ed27c7e9c7293f",
    "fsa_study/utilizers.py": "cd99aa30d95991f2e834b44aa864456303e6bfc4b5dd715633a22318161840a8",
    "plan_study.py": "60e1f4d72e3434137ffeed900be3754562abdac68923eedef80c2a8718f94804",
    "prepare_full_training.py": "7f2f8754f1db2fb7d2219bb004efb2ed985184f38977ecac361a7d77fa027f6f",
    "profile_study.py": "8bb5d68764d710481569656d83fc96a5b959dff48d6c93fc74d64f2f40174163",
    "recover_evaluation.py": "25c68385130f946e4459e91cb037fc34b91d1c5313bd0448b7b01be5b10bc77d",
    "run_plan.py": "74c83f6913c29d886cdb767235a78993b98f1fa28213b4059b0fcbffecc0fccb",
    "run_study.py": "c132a04d15a1acd4aa82426d7bb9766ad497a19656fdd034edd54d77c0c5265d",
    "summarize_study.py": "6c611e95c608f42099b7db3ff00d7a996cedf062e1bbfc663f62589c5cbfada4",
    "tests/conftest.py": "2942a561b121bf448028b6d06c3bd27195d5c63b6f8cb9e3e2589b280dc51ab0",
    "tests/test_benchmark_engine.py": "9929cdfcc54b5f5aae3629f6e303fdaf0a942f16cc2fcabc6093df4a12f0cd86",
    "tests/test_benchmark_manifest.py": "92f04d6f6444e92eea2b44383e9a43916584981eb456f87fe4e55b97c99c3285",
    "tests/test_cod10k_overlap.py": "a364ceb10ba28ecd271cdcd48a9d96efeb4df7c22a921b825bfebe15322ad3dc",
    "tests/test_complete_benchmark.py": "ddd93cf489928a82d399fa4b3a8323c821208c5a1e8a0284efd9181117aae7c6",
    "tests/test_complexity.py": "e5b29b230261ca0457ce4086a02c7bbf1b698070498358d5a85c0cbf9bf7e525",
    "tests/test_data.py": "27814b0f845d0d78617c221bc9a8fb09b4add2849210346b531bb9611cc6a320",
    "tests/test_generators.py": "150891535d3e66f306022e82dac6a6407fde38f99dcaf93433ce9873005aad4e",
    "tests/test_geometry.py": "9ffd7144bd6fc62977320a92db1f2699b470418f6b132a7ec3234fe9de2b99ab",
    "tests/test_install_study.py": "93e005e83b0b0ec31e77a61e1038293e1c69e9e86817decd27915d36ab0625a9",
    "tests/test_leaderboard.py": "8b4984de9aba644b56615bd1b4d7bfc0dcbf27cfb0042555a6f339586f10b90f",
    "tests/test_metrics.py": "f3535960a3458fab1f68ea5b3a6a494a288561b07a6e1055d749440ef7f07608",
    "tests/test_model.py": "b7fe0a9445a826976efb0cc1c7112ff2926ab13fd302031cdf6ffd38c755da04",
    "tests/test_overlap_reporting.py": "17d8ee1038c56c4dd51c8c4138dd7038315754ea694804e72a17369d43e1e7d4",
    "tests/test_prepare_full_training.py": "071b668fb4eea4062e347ef8bc9381a2b928465a11d4594ade64a589856621dd",
    "tests/test_recover_evaluation.py": "e2a0d9a0009c30efd38f1ab7408a61fc8066b345198e4747f519d3f43c09bfad",
    "tests/test_reference_operators.py": "c03042ac7a5841fbffe2a2397c696a13fb9f7a0e097e7f29e749f8fe995ba638",
    "tests/test_run_plan.py": "2d52af90553d90f641a440fc870902d7de860c2a5034bd6d09495878246411a6",
    "tests/test_training_protocol.py": "1f371efe917e7278e5a48a5390ca22820c35dc8175b53a9d4fce242691716a1b",
    "tests/test_utilizers.py": "219fc9b59a7e1f87405d0593ba45147fc095fc959161d9270fa9e6c694ef5e55",
    "tests/test_workflow.py": "3e6c93007318154d695f2c39ff308683e5e502ede4374db918eea4db865afcbe"
  },
  "training_protocol": "full_train",
  "training_sample_count": 4040,
  "training_split_counts": {
    "train": {
      "camo": 1000,
      "cod10k": 3040
    },
    "val": {}
  },
  "validation_sample_count": 0
}
```

</details>

## Pending runs

| Experiment | Seed | State | Run |
| --- | ---: | --- | --- |
| native_conv__mean | 3407 | not_started | `/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-full4040-fixed-20261101_200834/runs/priority8/native_conv__mean__seed3407` |
| projection__mean | 3407 | not_started | `/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-full4040-fixed-20261101_200834/runs/priority8/projection__mean__seed3407` |
| dysample__asff | 3407 | not_started | `/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-full4040-fixed-20261101_200834/runs/priority8/dysample__asff__seed3407` |
| dysample__sk | 3407 | not_started | `/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-full4040-fixed-20261101_200834/runs/priority8/dysample__sk__seed3407` |
| dysample__mean | 3407 | not_started | `/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-full4040-fixed-20261101_200834/runs/priority8/dysample__mean__seed3407` |

## Runs missing standard benchmark coverage

| Experiment | Seed | Missing datasets | Nonstandard counts |
| --- | ---: | --- | --- |
| native_conv__mean | 3407 | camo, chameleon, cod10k, nc4k | `{}` |
| projection__mean | 3407 | camo, chameleon, cod10k, nc4k | `{}` |
| dysample__asff | 3407 | camo, chameleon, cod10k, nc4k | `{}` |
| dysample__sk | 3407 | camo, chameleon, cod10k, nc4k | `{}` |
| dysample__mean | 3407 | camo, chameleon, cod10k, nc4k | `{}` |
