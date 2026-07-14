import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_robocoin_modelscope.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_robocoin_modelscope", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolves_modelscope_dataset_names_and_preserves_apostrophes():
    module = load_module()
    datasets = {
        "AI2_Alphabot_2_carry_the_clothes_basket": {"dataset_size": "1.00 GB"},
        "Airbot_MMK2_storage_cup_rubik's_cube": {"dataset_size": "512.00 MB"},
    }
    hub_names = {
        "modelscope": {
            "AI2_Alphabot_2_carry_the_clothes_basket": "alpha_bot_2_carry_the_clothes_basket",
        }
    }

    names = module.resolve_dataset_names(datasets, hub_names, hub="modelscope")

    assert names == [
        "alpha_bot_2_carry_the_clothes_basket",
        "Airbot_MMK2_storage_cup_rubik's_cube",
    ]


def test_batches_keep_dataset_arguments_as_plain_list_items():
    module = load_module()

    batches = list(module.batched(["a", "b", "c", "d", "e"], 2))

    assert batches == [["a", "b"], ["c", "d"], ["e"]]


def test_build_download_command_uses_argument_vector_not_shell_string(tmp_path):
    module = load_module()

    command = module.build_download_command(
        ["a", "Airbot_MMK2_storage_cup_rubik's_cube"],
        target_dir=tmp_path,
    )

    assert command == [
        "robocoin-download",
        "--hub",
        "modelscope",
        "--ds_lists",
        "a",
        "Airbot_MMK2_storage_cup_rubik's_cube",
        "--target-dir",
        str(tmp_path),
    ]


def test_select_smoke_dataset_names_uses_smallest_current_modelscope_names():
    module = load_module()
    datasets = {
        "Large_Dataset": {"dataset_size": "2.00 GB"},
        "Small_A": {"dataset_size": "20.00 MB"},
        "Small_B": {"dataset_size": "30.00 MB"},
    }
    hub_names = {"modelscope": {"Small_A": "small_a_ms"}}

    names = module.select_smoke_dataset_names(datasets, hub_names, count=2)

    assert names == ["small_a_ms", "Small_B"]
