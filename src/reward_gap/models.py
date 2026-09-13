"""Explicit, single-device Hugging Face model loading for workshop experiments.

This module loads weights and tokenizers only. Chat formatting, scoring,
embeddings, LoRA adapters, and the policy value head belong in other modules.
Importing it does not import PyTorch, download files, or allocate a model.
Requires torch and transformers when loading (CPU checks used torch 2.14.0
and transformers 5.17.0). Install the appropriate PyTorch build for the device.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from reward_gap.config import ModelDtype

if TYPE_CHECKING:
    from reward_gap.config import ExperimentConfig
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelLoadError(RuntimeError):
    """A model or tokenizer could not be loaded with the requested settings."""


@dataclass(frozen=True)
class ModelSpec:
    """A Hub model ID plus revision, or an existing local checkpoint path.

    For local checkpoints use a Path object; relative paths resolve against
    project_root. Tokenizer files must be in the same checkpoint repository.
    """

    source: str | Path
    revision: str = "main"


@dataclass(frozen=True)
class LoadOptions:
    device: str = "cpu"
    dtype: ModelDtype = "float32"
    cache_dir: str | Path = Path("model_cache")
    allow_downloads: bool = False
    # Opt-in for checkpoints that have EOS but no padding token.
    pad_to_eos: bool = False


@dataclass(frozen=True)
class LoadedModel:
    model: "PreTrainedModel"
    tokenizer: "PreTrainedTokenizerBase"
    source: str
    revision: str | None
    role: Literal["policy", "reward"]


def _project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        root = Path(project_root).resolve()
        if not (root / "pyproject.toml").is_file():
            raise ModelLoadError(f"Project root has no pyproject.toml: {root}")
        return root
    # Anchor to the installed source location, not a notebook's working dir.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ModelLoadError("Cannot locate project root; pass project_root explicitly")


def _load(spec: ModelSpec, options: LoadOptions, role: Literal["policy", "reward"],
          project_root: str | Path | None) -> LoadedModel:
    if not isinstance(spec, ModelSpec) or not isinstance(options, LoadOptions):
        raise ModelLoadError("Expected ModelSpec and LoadOptions objects")
    if not isinstance(spec.source, (str, Path)) or not str(spec.source).strip():
        raise ModelLoadError("Model source must be a nonempty Hub ID or local Path")
    if not isinstance(spec.revision, str) or not spec.revision.strip():
        raise ModelLoadError("Model revision must be a nonempty string")
    if options.dtype not in ("float32", "float16", "bfloat16"):
        raise ModelLoadError("dtype must be float32, float16, or bfloat16")
    if type(options.allow_downloads) is not bool or type(options.pad_to_eos) is not bool:
        raise ModelLoadError("allow_downloads and pad_to_eos must be booleans")
    if not isinstance(options.cache_dir, (str, Path)) or not str(options.cache_dir).strip():
        raise ModelLoadError("cache_dir must be a nonempty path")
    root = _project_root(project_root)
    if isinstance(spec.source, Path):
        local = (root / spec.source).resolve()
        if not local.is_dir():
            raise ModelLoadError(f"Local checkpoint directory does not exist: {local}")
        source = str(local)
    else:
        source = spec.source

    try:
        import torch
        from transformers import (AutoConfig, AutoModelForCausalLM,
                                  AutoModelForSequenceClassification, AutoTokenizer)
    except ImportError as exc:
        raise ModelLoadError(
            "Model loading requires PyTorch and Transformers in this Python environment. "
            "Install a PyTorch build for your CPU/GPU and install transformers."
        ) from exc

    try:
        device = torch.device(options.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ModelLoadError(f"Invalid device: {options.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ModelLoadError("Only CPU and single CUDA-device loading are supported")
    if device.type == "cpu" and (device.index is not None or options.dtype != "float32"):
        raise ModelLoadError("CPU loading requires device='cpu' and dtype='float32'")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA was requested but is unavailable in this environment")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ModelLoadError(f"CUDA device {index} does not exist")
        device = torch.device(f"cuda:{index}")
        with torch.cuda.device(device):
            if options.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ModelLoadError(f"{device} does not support bfloat16")

    kwargs = {
        "cache_dir": str((root / options.cache_dir).resolve()),
        "local_files_only": not options.allow_downloads,
        "trust_remote_code": False,
        "revision": spec.revision,
    }
    try:
        config = AutoConfig.from_pretrained(source, **kwargs)
        if getattr(config, "quantization_config", None) is not None:
            raise ModelLoadError("Prequantized checkpoints need a dedicated loader")
        # Once Hub config is resolved, use that same version for all assets.
        revision = None if isinstance(spec.source, Path) else getattr(config, "_commit_hash", None)
        if revision is not None:
            kwargs["revision"] = revision
        if role == "policy" and getattr(config, "is_encoder_decoder", False):
            raise ModelLoadError("Policy loading supports decoder-only causal language models")
        if role == "reward":
            if getattr(config, "num_labels", None) != 1:
                raise ModelLoadError("Reward checkpoint must already have a single scalar output (num_labels=1)")
            architectures = getattr(config, "architectures", None) or []
            if not any(name.endswith("ForSequenceClassification") for name in architectures):
                raise ModelLoadError("Reward checkpoint must declare a sequence-classification architecture")
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        if tokenizer.pad_token_id is None:
            if not options.pad_to_eos or tokenizer.eos_token_id is None:
                raise ModelLoadError("Tokenizer has no padding token. If appropriate for the checkpoint, "
                                     "explicitly set pad_to_eos=True; no vocabulary is added automatically.")
            tokenizer.pad_token = tokenizer.eos_token
        if config.pad_token_id is not None and config.pad_token_id != tokenizer.pad_token_id:
            raise ModelLoadError("Checkpoint and tokenizer disagree on pad_token_id")
        config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left" if role == "policy" else "right"
        loader = AutoModelForCausalLM if role == "policy" else AutoModelForSequenceClassification
        loaded = loader.from_pretrained(
            source, config=config, dtype=getattr(torch, options.dtype),
            output_loading_info=True, ignore_mismatched_sizes=False, **kwargs,
        )
        # Transformers returns (model, diagnostics) for output_loading_info=True,
        # but its AutoModel return annotation describes only the model.
        model, info = cast("tuple[PreTrainedModel, dict[str, Any]]", loaded)
        issues = {key: info.get(key) for key in
                  ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
        if issues:
            raise ModelLoadError(f"Checkpoint did not load exactly; refusing uninitialized or unused weights: {issues}")
        embeddings = model.get_input_embeddings()
        vocabulary_size = getattr(embeddings, "num_embeddings", None)
        if not isinstance(vocabulary_size, int):
            raise ModelLoadError("Checkpoint does not expose an input embedding vocabulary size")
        if len(tokenizer) > vocabulary_size:
            raise ModelLoadError("Tokenizer vocabulary exceeds checkpoint embeddings; explicit adaptation is required")
        # Frozen base for later LoRA attachment; proxy/judge also stay frozen.
        model.requires_grad_(False)
        model.eval()
        # Use nn.Module's typed interface; Transformers decorates this method
        # in a way that loses its bound-method signature for static checkers.
        cast(torch.nn.Module, model).to(device)
        if role == "policy":
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        return LoadedModel(model, tokenizer, source, revision, role)
    except ModelLoadError:
        raise
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        raise ModelLoadError(f"Cannot load {role} checkpoint {source!r}: {exc}") from exc


def load_policy_model(spec: ModelSpec, options: LoadOptions = LoadOptions(), *,
                      project_root: str | Path | None = None) -> LoadedModel:
    """Load a frozen causal-LM base; policy.py will attach trainable LoRA/value layers."""
    return _load(spec, options, "policy", project_root)


def load_reward_model(spec: ModelSpec, options: LoadOptions = LoadOptions(), *,
                      project_root: str | Path | None = None) -> LoadedModel:
    """Load a frozen scalar sequence classifier for either proxy or judge.

    This validates loading structure, not reward semantics. Score direction,
    chat formatting, pooling, and embeddings need checkpoint-specific checks
    in scorers.py. Custom reward architectures are not supported here.
    """
    return _load(spec, options, "reward", project_root)


def load_experiment_model(config: "ExperimentConfig", role: Literal["policy", "proxy", "judge"], *,
                          project_root: str | Path | None = None) -> LoadedModel:
    """Load one selected role using the experiment's model and runtime settings.

    Roles are loaded individually so the caller controls model lifetimes and
    memory use. No models are loaded by load_config() itself.
    """
    if role not in ("policy", "proxy", "judge"):
        raise ModelLoadError("role must be policy, proxy, or judge")
    if config.models is None:
        raise ModelLoadError("The experiment configuration has no models section")
    reference = getattr(config.models, role)
    options = LoadOptions(device=config.runtime.device, dtype=config.runtime.dtype,
                          cache_dir=config.runtime.model_cache,
                          allow_downloads=config.runtime.allow_downloads)
    loader = load_policy_model if role == "policy" else load_reward_model
    return loader(ModelSpec(reference.id, reference.revision), options, project_root=project_root)
