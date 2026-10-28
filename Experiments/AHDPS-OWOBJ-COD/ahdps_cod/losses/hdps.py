from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ahdps_cod.utils.mask_targets import build_hdps_targets, build_region_objectness_targets

from .segmentation import bce_dice_loss


def _safe_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)


def _safe_mean(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return _safe_zero(reference)
    return values.mean()


def _prob_bce_autocast_safe(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.float().clamp(eps, 1.0 - eps)
    target = target.float()
    logits = torch.logit(prob, eps=eps)
    return F.binary_cross_entropy_with_logits(logits, target)


def diagonal_gaussian_kl(
    q_mean: torch.Tensor,
    q_var: torch.Tensor,
    p_mean: torch.Tensor,
    p_var: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    q_var = q_var.clamp_min(eps)
    p_var = p_var.clamp_min(eps)
    term = torch.log(p_var / q_var) + (q_var + (q_mean - p_mean) ** 2) / p_var - 1.0
    return 0.5 * term.mean()


class AHDPSCriterion(nn.Module):
    def __init__(self, cfg_model: Dict) -> None:
        super().__init__()
        hdps_cfg = cfg_model["hdps"]
        self.core_kernel = int(hdps_cfg.get("core_kernel", 7))
        self.halo_kernel = int(hdps_cfg.get("halo_kernel", 11))
        self.halo_target = float(hdps_cfg.get("halo_target", 0.2))
        self.boundary_weight = float(hdps_cfg.get("boundary_weight", 2.0))
        self.aux_weights = {k: float(v) for k, v in hdps_cfg.get("aux_weights", {"p3": 0.7, "p4": 0.5, "p5": 0.25}).items()}
        self.coarse_weight = float(hdps_cfg.get("coarse_weight", 0.35))
        self.prior_weight = float(hdps_cfg.get("prior_weight", 0.15))
        self.final_weight = float(hdps_cfg.get("final_weight", 1.0))

        obj_cfg = dict(cfg_model.get("objectness", {}))
        self.objectness_enabled = bool(obj_cfg.get("enabled", False))
        self.obj_pos_body_thresh = float(obj_cfg.get("pos_body_thresh", 0.55))
        self.obj_pos_core_thresh = float(obj_cfg.get("pos_core_thresh", 0.10))
        self.obj_neg_body_thresh = float(obj_cfg.get("neg_body_thresh", 0.02))
        self.obj_neg_halo_thresh = float(obj_cfg.get("neg_halo_thresh", 0.05))
        self.obj_amb_halo_thresh = float(obj_cfg.get("amb_halo_thresh", 0.10))
        self.obj_amb_boundary_thresh = float(obj_cfg.get("amb_boundary_thresh", 0.10))
        self.obj_amb_guidance_mix = float(obj_cfg.get("amb_guidance_mix", 0.5))
        self.obj_eps = float(obj_cfg.get("eps", 1e-4))
        loss_weights = obj_cfg.get("loss_weights", {}) if isinstance(obj_cfg.get("loss_weights", {}), dict) else {}
        self.obj_bce_weight = float(loss_weights.get("bce", 0.20))
        self.obj_proto_weight = float(loss_weights.get("proto", 0.10))
        self.obj_kl_weight = float(loss_weights.get("kl", 0.05))
        self.obj_energy_weight = float(loss_weights.get("energy", 0.05))
        self.obj_consistency_weight = float(loss_weights.get("consistency", 0.10))
        self.energy_margin_pos_amb = float(obj_cfg.get("energy_margin_pos_amb", 0.05))
        self.energy_margin_amb_neg = float(obj_cfg.get("energy_margin_amb_neg", 0.05))

    def build_targets(self, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        return build_hdps_targets(
            masks=masks,
            core_kernel=self.core_kernel,
            halo_kernel=self.halo_kernel,
            halo_target=self.halo_target,
        )

    def _build_region_targets(self, outputs: Dict[str, torch.Tensor], masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        grid_size = tuple(int(v) for v in outputs.get("region_grid", (8, 8)))
        guidance = 0.5 * (torch.sigmoid(outputs["coarse_logits"]).detach() + outputs["dense_prior"].detach())
        return build_region_objectness_targets(
            masks=masks,
            grid_size=grid_size,
            core_kernel=self.core_kernel,
            halo_kernel=self.halo_kernel,
            halo_target=self.halo_target,
            pos_body_thresh=self.obj_pos_body_thresh,
            pos_core_thresh=self.obj_pos_core_thresh,
            neg_body_thresh=self.obj_neg_body_thresh,
            neg_halo_thresh=self.obj_neg_halo_thresh,
            amb_halo_thresh=self.obj_amb_halo_thresh,
            amb_boundary_thresh=self.obj_amb_boundary_thresh,
            amb_guidance_mix=self.obj_amb_guidance_mix,
            guidance_map=guidance,
        )

    def _objectness_losses(self, outputs: Dict[str, torch.Tensor], body: torch.Tensor) -> Dict[str, torch.Tensor]:
        reference = outputs["final_logits"]
        zeros = {
            "loss_obj_bce": _safe_zero(reference),
            "loss_obj_proto": _safe_zero(reference),
            "loss_obj_kl": _safe_zero(reference),
            "loss_obj_energy": _safe_zero(reference),
            "loss_obj_consistency": _safe_zero(reference),
            "loss_obj_total": _safe_zero(reference),
        }
        if not self.objectness_enabled:
            return zeros
        if "region_logits" not in outputs or "region_embeddings" not in outputs:
            return zeros
        if not bool(outputs.get("objectness_active", False)):
            return zeros
        if not bool(outputs.get("prototype_ready", False)):
            return zeros

        target_info = self._build_region_targets(outputs, body)
        valid_mask = target_info["valid"]
        pos_mask = target_info["positive"]
        amb_mask = target_info["ambiguous"]
        neg_mask = target_info["negative"]
        soft_targets = target_info["targets"]

        region_logits = outputs["region_logits"]
        region_d_fg = outputs["region_d_fg"]
        region_d_bg = outputs["region_d_bg"]
        region_energy = outputs["region_energy"]
        region_embeds = outputs["region_embeddings"]

        if bool(valid_mask.any()):
            loss_obj_bce = F.binary_cross_entropy_with_logits(region_logits[valid_mask], soft_targets[valid_mask])
        else:
            loss_obj_bce = _safe_zero(reference)

        loss_obj_proto = _safe_mean(region_d_fg[pos_mask], reference) + _safe_mean(region_d_bg[neg_mask], reference)

        stats = outputs.get("prototype_stats", {})
        loss_obj_kl = _safe_zero(reference)
        if bool(pos_mask.any()) and int(pos_mask.sum().item()) > 1 and isinstance(stats, dict) and "fg_mean" in stats:
            fg_embeds = region_embeds[pos_mask]
            fg_mean = fg_embeds.mean(dim=0)
            fg_var = fg_embeds.var(dim=0, unbiased=False).clamp_min(self.obj_eps)
            loss_obj_kl = loss_obj_kl + diagonal_gaussian_kl(fg_mean, fg_var, stats["fg_mean"], stats["fg_var"], eps=self.obj_eps)
        if bool(neg_mask.any()) and int(neg_mask.sum().item()) > 1 and isinstance(stats, dict) and "bg_mean" in stats:
            bg_embeds = region_embeds[neg_mask]
            bg_mean = bg_embeds.mean(dim=0)
            bg_var = bg_embeds.var(dim=0, unbiased=False).clamp_min(self.obj_eps)
            loss_obj_kl = loss_obj_kl + diagonal_gaussian_kl(bg_mean, bg_var, stats["bg_mean"], stats["bg_var"], eps=self.obj_eps)

        loss_obj_energy = _safe_zero(reference)
        if bool(pos_mask.any()) and bool(amb_mask.any()):
            loss_obj_energy = loss_obj_energy + F.relu(
                region_energy[pos_mask].mean() - region_energy[amb_mask].mean() + self.energy_margin_pos_amb
            )
        if bool(amb_mask.any()) and bool(neg_mask.any()):
            loss_obj_energy = loss_obj_energy + F.relu(
                region_energy[amb_mask].mean() - region_energy[neg_mask].mean() + self.energy_margin_amb_neg
            )

        loss_obj_consistency = _prob_bce_autocast_safe(outputs["objectness_prior"], body, eps=self.obj_eps)

        loss_obj_total = (
            self.obj_bce_weight * loss_obj_bce +
            self.obj_proto_weight * loss_obj_proto +
            self.obj_kl_weight * loss_obj_kl +
            self.obj_energy_weight * loss_obj_energy +
            self.obj_consistency_weight * loss_obj_consistency
        )
        return {
            "loss_obj_bce": loss_obj_bce,
            "loss_obj_proto": loss_obj_proto,
            "loss_obj_kl": loss_obj_kl,
            "loss_obj_energy": loss_obj_energy,
            "loss_obj_consistency": loss_obj_consistency,
            "loss_obj_total": loss_obj_total,
        }

    def forward(self, outputs: Dict[str, torch.Tensor], masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        targets = self.build_targets(masks)
        body = targets["body"]
        core = targets["core"]
        p3_soft = targets["p3_soft"]
        boundary = targets["boundary"]

        loss_final = bce_dice_loss(outputs["final_logits"], body)
        loss_coarse = bce_dice_loss(outputs["coarse_logits"], body)

        p3_weight_map = 1.0 + self.boundary_weight * boundary
        loss_p3 = bce_dice_loss(outputs["aux_logits"]["p3"], p3_soft, weight_map=p3_weight_map, dice_target=body)
        loss_p4 = bce_dice_loss(outputs["aux_logits"]["p4"], body)
        loss_p5 = bce_dice_loss(outputs["aux_logits"]["p5"], core)

        loss_prior_dense = _prob_bce_autocast_safe(outputs["dense_prior"], body, eps=self.obj_eps)
        loss_prior_hybrid = _prob_bce_autocast_safe(outputs["hybrid_prior"], body, eps=self.obj_eps)
        loss_prior = 0.5 * (loss_prior_dense + loss_prior_hybrid)

        obj_losses = self._objectness_losses(outputs, body)

        total = (
            self.final_weight * loss_final +
            self.coarse_weight * loss_coarse +
            self.aux_weights.get("p3", 0.0) * loss_p3 +
            self.aux_weights.get("p4", 0.0) * loss_p4 +
            self.aux_weights.get("p5", 0.0) * loss_p5 +
            self.prior_weight * loss_prior +
            obj_losses["loss_obj_total"]
        )
        return {
            "loss_total": total,
            "loss_final": loss_final.detach(),
            "loss_coarse": loss_coarse.detach(),
            "loss_p3": loss_p3.detach(),
            "loss_p4": loss_p4.detach(),
            "loss_p5": loss_p5.detach(),
            "loss_prior": loss_prior.detach(),
            "loss_prior_dense": loss_prior_dense.detach(),
            "loss_prior_hybrid": loss_prior_hybrid.detach(),
            **{k: v.detach() for k, v in obj_losses.items()},
        }
