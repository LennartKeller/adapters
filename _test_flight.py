import torch
from adapters import Wav2Vec2AdapterModel, init, AutoAdapterModel
from transformers import Wav2Vec2ForCTC

# Test some imports
try:
    from adapters.heads import CTCHead
except Exception as e:
    print("Failed to import CTCHead due to:")
    print(e)


def show_param_states(model: torch.nn.Module) -> dict[str, bool]:
    states = {n: p.requires_grad for n, p in model.named_parameters()}
    return states


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from rich import print

    case = "easy"
    if len(sys.argv) > 1:
        case = sys.argv[1]
    assert case in ("easy", "complex")
    print(f"Using case {case}")

    base_model_name_or_path = "facebook/wav2vec2-xls-r-300m"
    print(f"Loading base model ('{base_model_name_or_path}')...")
    model = Wav2Vec2ForCTC.from_pretrained(base_model_name_or_path)
    init(model)
    print("Loaded base model!")

    def easy_case(model: Wav2Vec2AdapterModel) -> Wav2Vec2AdapterModel:
        print("Applying easy adapter setting: Add two consecutive bottle neck adapters in each transformers layers.")
        adapters = ("language_en", "task")
        for adapter in adapters:
            model.add_adapter(adapter, config="seq_bn", set_active=False)

        # Activate adapters (check if this really works)
        # And see if freezing the other parameters works
        model.set_active_adapters("language_en")
        model.set_active_adapters("task")
        model.train_adapter("language_en")
        model.train_adapter("task")

        print(model.state_dict().keys())
        print(show_param_states(model))
        return model, adapters

    def complex_case(wav2vec2_adapter_model: Wav2Vec2AdapterModel) -> Wav2Vec2AdapterModel: ...

    case_fn = easy_case if case == "easy" else complex_case
    model, adapters = case_fn(model)

    x = torch.randn(8, 16000, device=model.device)
    inputs = {"input_values": x}
    outputs = model(**inputs)
    print(outputs.keys())

    # Test saving and loading of adapters weights
    # Any examples/checkpoint-* will be ignored by default .gitignore!
    checkpoint_dir = Path("examples/checkpoint-test")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_dir = checkpoint_dir / "full-model"
    all_adapters_dir = checkpoint_dir / "adapters-only-all"
    manual_adapters_dir = checkpoint_dir / "adapters-only-manually"

    print("Saving and loading from full model...")
    model.save_pretrained(model_dir)
    print("Saved full model!")
    print("\tLoading full model from checkpoint...")
    _model = AutoAdapterModel.from_pretrained(model_dir)
    print("\tLoaded full model!")
    print(_model.state_dict().keys())

    print("Saving and loading all adapters at once")
    model.save_all_adapters(all_adapters_dir, with_head=True)
    print("Saved all adapters at once!")
    print("\tLoading all adapters from checkpoint...")
    _model = Wav2Vec2ForCTC.from_pretrained(base_model_name_or_path)
    init(_model)
    for adapter in adapters:
        _model.load_adapter(str(all_adapters_dir / adapter), with_head=True)
    print("\tLoaded all adapters!")
    print(_model.state_dict().keys())

    print("Saving and loading all adapters manually")
    for adapter in adapters:
        adapter_dir = manual_adapters_dir / adapter
        adapter_dir.mkdir(parents=True, exist_ok=True)
        print(f"\tSaving adapter '{adapter}' to '{adapter_dir}'...")
        model.save_adapter(str(adapter_dir), adapter, with_head=True)

    _model = Wav2Vec2ForCTC.from_pretrained(base_model_name_or_path)
    init(_model)
    for adapter in adapters:
        _model.load_adapter(str(all_adapters_dir / adapter))
    

