"""Optional feature-distillation warmup, preserving the constant-weight default."""
import math


def hidden_weight_at_epoch(distillation, epoch):
    weight = float(distillation.get('hidden_weight', 0.0))
    schedule = distillation.get('hidden_schedule', {})
    start = int(schedule.get('start_epoch', 1))
    warmup = int(schedule.get('warmup_epochs', 0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('distillation.hidden_weight must be finite and non-negative')
    if start < 1 or warmup < 0 or epoch < 1:
        raise ValueError('Feature schedule requires start_epoch>=1, warmup_epochs>=0 and epoch>=1')
    if epoch < start:
        return 0.0
    factor = min(1.0, (epoch-start+1)/warmup) if warmup else 1.0
    return weight * factor
