from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

import faceless
from faceless.userdata.logging import setup_logging as setup_logging_real


LABEL_CONTENT = "216 0.50 0.50 0.25 0.25\n264 0.50 0.50 0.25 0.25\n"


def write_image(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


class FakeYOLO:
    names = {216: "woman", 264: "human face"}
    predict_calls: list[dict[str, object]] = []

    def __init__(self, _model_name: str):
        pass

    def predict(self, **kwargs: object) -> list[object]:
        FakeYOLO.predict_calls.append(kwargs)
        source_dir = Path(str(kwargs["project"]))
        def run_prediction() -> list[object]:
            labels_dir = source_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            for file_path in source_dir.iterdir():
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                (labels_dir / f"{file_path.stem}.txt").write_text(LABEL_CONTENT, encoding="utf-8")
            logging.getLogger("ultralytics").info("YOLO predict called for %s", source_dir)
            return []

        if bool(kwargs.get("stream")):
            def generator():
                run_prediction()
                yield {"ok": True}

            return generator()  # type: ignore[return-value]

        return run_prediction()


class FailingYOLO(FakeYOLO):
    def predict(self, **kwargs: object) -> list[object]:
        raise ValueError("forced predict failure")


def run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: Path,
    *,
    force_labels: bool = False,
    yolo_class: type[FakeYOLO] = FakeYOLO,
) -> None:
    model_path = tmp_path / "fake-model.pt"
    model_path.write_text("fake model", encoding="utf-8")
    log_dir = tmp_path / "logs"

    FakeYOLO.predict_calls = []

    monkeypatch.setattr(faceless, "YOLO", yolo_class)
    monkeypatch.setattr(faceless, "resolve_model_path", lambda _model_name: model_path)
    monkeypatch.setattr(faceless, "prompt_open_folders", lambda **_kwargs: None)
    monkeypatch.setattr(
        faceless,
        "setup_logging",
        lambda verbose=False: setup_logging_real(verbose=verbose, log_dir=log_dir),
    )

    argv = ["faceless", str(source), "-Auto"]
    if force_labels:
        argv.append("-Label")
    monkeypatch.setattr(sys, "argv", argv)
    faceless.main()


def test_generates_labels_when_labels_directory_exists_but_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "sample01.png")
    (source / "labels").mkdir()

    run_main(monkeypatch, tmp_path, source)

    assert len(FakeYOLO.predict_calls) == 1
    assert FakeYOLO.predict_calls[0]["stream"] is True
    assert (source / "labels" / "sample01.txt").is_file()


def test_skips_generation_when_all_labels_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "sample01.png")
    labels_dir = source / "labels"
    labels_dir.mkdir()
    (labels_dir / "sample01.txt").write_text(LABEL_CONTENT, encoding="utf-8")

    run_main(monkeypatch, tmp_path, source)

    assert len(FakeYOLO.predict_calls) == 0


def test_force_labels_triggers_generation_even_when_labels_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "sample01.png")
    labels_dir = source / "labels"
    labels_dir.mkdir()
    (labels_dir / "sample01.txt").write_text(LABEL_CONTENT, encoding="utf-8")

    run_main(monkeypatch, tmp_path, source, force_labels=True)

    assert len(FakeYOLO.predict_calls) == 1


def test_raises_runtime_error_when_yolo_predict_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "sample01.png")

    with pytest.raises(RuntimeError, match="YOLO label generation failed"):
        run_main(monkeypatch, tmp_path, source, yolo_class=FailingYOLO)
