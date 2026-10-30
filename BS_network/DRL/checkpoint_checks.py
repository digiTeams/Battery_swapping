"""Checkpoint identity and effective-configuration compatibility checks.
Free of PyTorch imports; BaseDRL.load_checkpoint uses these helpers before
committing a restore."""

_IDENTITY_FIELDS = ("policy_type", "ablation", "gat_attention", "envy_weight")
_MISSING = object()
_MAX_PROBLEMS = 25

# Architecture-relevant value-input fields compared on every load; env-dependent
# graph arrays/normalization are absent so checkpoints transfer across networks.
VALUE_ARCHITECTURE_FIELDS = (
    "total_steps",
    "hour_dims",
    "use_gat",
    "attention_kind",
    "step_embed_dims",
    "gat_dims",
    "hidden_dims",
    "node_dims",
    "sender_dims",
)


def identity_mismatches(checkpoint, policy_type, ablation, gat_attention, envy_weight):
    """Return problem strings when the checkpoint identity fields differ from the learner."""
    expected = {
        "policy_type": policy_type,
        "ablation": ablation,
        "gat_attention": gat_attention,
        "envy_weight": envy_weight,
    }
    problems = []
    for field in _IDENTITY_FIELDS:
        saved = checkpoint.get(field, _MISSING)
        if saved is _MISSING:
            problems.append(f"identity field '{field}' is missing from the checkpoint")
        elif saved != expected[field]:
            problems.append(
                f"identity field '{field}': checkpoint={saved!r} "
                f"!= learner={expected[field]!r}"
            )
    return problems


def configuration_mismatches(current, saved, keys=None):
    """Compare selected top-level keys item by item and return problem
    strings (empty = match). ``keys`` selects keys (None = all)."""
    problems = []
    selected = sorted(current) if keys is None else list(keys)

    for key in selected:
        if key not in current:
            problems.append(f"{key}: missing from the current configuration")
        elif key not in saved:
            problems.append(f"{key}: missing from the checkpoint")
        else:
            _walk(current[key], saved[key], str(key), problems)
        if len(problems) >= _MAX_PROBLEMS:
            problems.append(f"... truncated after {_MAX_PROBLEMS} differences")
            break
    return problems


def _walk(current, saved, path, problems):
    """Recurse containers and record every differing leaf with its path."""
    if len(problems) >= _MAX_PROBLEMS:
        return
    if isinstance(current, dict) and isinstance(saved, dict):
        keys = sorted(set(current) | set(saved), key=str)
        for key in keys:
            sub = f"{path}.{key}"
            if key not in current:
                problems.append(f"{sub}: present only in the checkpoint")
            elif key not in saved:
                problems.append(f"{sub}: present only in the current configuration")
            else:
                _walk(current[key], saved[key], sub, problems)
            if len(problems) >= _MAX_PROBLEMS:
                return
    elif isinstance(current, (list, tuple)) and isinstance(saved, (list, tuple)):
        if len(current) != len(saved):
            problems.append(
                f"{path}: length {len(current)} (current) != {len(saved)} (checkpoint)"
            )
            return
        for index, (a, b) in enumerate(zip(current, saved)):
            _walk(a, b, f"{path}[{index}]", problems)
            if len(problems) >= _MAX_PROBLEMS:
                return
    else:
        if current != saved:
            problems.append(f"{path}: current={current!r} != checkpoint={saved!r}")


def value_input_mismatches(current, saved):
    """Compare architecture-relevant value-input fields; fields missing from
    older checkpoints are skipped so they stay loadable."""
    fields = [
        key for key in VALUE_ARCHITECTURE_FIELDS
        if key in current and key in saved
    ]
    return configuration_mismatches(
        {key: current[key] for key in fields},
        {key: saved[key] for key in fields},
    )
