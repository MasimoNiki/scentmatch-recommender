import importlib.util
from pathlib import Path


def test_finetune_script_parser_defaults() -> None:
    path = Path(__file__).parents[1] / "scripts" / "finetune_e5.py"
    spec = importlib.util.spec_from_file_location("finetune_e5", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args([])
    assert args.batch_size == 8
    assert args.epochs == 1.0
