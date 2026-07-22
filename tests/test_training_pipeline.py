from app.core.artifacts import load_bundle
from scripts.generate_demo_data import generate_events
from src.training.data import write_jsonl
from src.training.train import run_training


def test_training_pipeline_produces_loadable_evaluated_bundle(tmp_path) -> None:
    dataset = tmp_path / "events.jsonl"
    bundle = tmp_path / "bundle"
    write_jsonl(dataset, generate_events(sessions=12, events_per_session=6))

    report = run_training(dataset, bundle, epochs=1, top_k=5, seed=7)
    processor, model, manifest = load_bundle(bundle)

    assert report["validation"]["recall_at_k"] >= 0
    assert "popularity_baseline" in report
    assert manifest.dataset_id == report["dataset_id"]
    assert model.recommend(processor.to_tensor(["item-1"]), top_k=3)
