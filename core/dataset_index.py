from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    sign: str
    subject_id: str
    session_id: str
    frame_count: int
    fps: float
    duration_s: float
    collection_mode: str
    dataset_version: str
    schema_version: str
    sample_path: Path
    metadata_path: Path
    landmarks_path: Path
    video_path: Path
    has_video: bool

    def to_row(self, project_root: Path, split: str) -> dict[str, object]:
        project_root = Path(project_root).resolve()

        def rel(path: Path) -> str:
            path = Path(path).resolve()
            try:
                return path.relative_to(project_root).as_posix()
            except ValueError:
                return path.as_posix()

        return {
            "sample_id": self.sample_id,
            "sign": self.sign,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "split": split,
            "frame_count": self.frame_count,
            "fps": f"{self.fps:.6f}",
            "duration_s": f"{self.duration_s:.6f}",
            "collection_mode": self.collection_mode,
            "dataset_version": self.dataset_version,
            "schema_version": self.schema_version,
            "sample_path": rel(self.sample_path),
            "metadata_path": rel(self.metadata_path),
            "landmarks_path": rel(self.landmarks_path),
            "video_path": rel(self.video_path) if self.has_video else "",
            "has_video": int(self.has_video),
        }


@dataclass
class DatasetScanReport:
    records: list[SampleRecord]
    errors: list[str]
    warnings: list[str]

    @property
    def subject_ids(self) -> set[str]:
        return {r.subject_id for r in self.records}

    @property
    def classes(self) -> set[str]:
        return {r.sign for r in self.records}


def _required_str(meta: dict, key: str, metadata_path: Path) -> str:
    value = meta.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{metadata_path}: missing/invalid '{key}'")
    return value.strip()


def _required_int(meta: dict, key: str, metadata_path: Path) -> int:
    value = meta.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{metadata_path}: invalid '{key}'")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{metadata_path}: missing/invalid '{key}'") from exc
    if result < 1:
        raise ValueError(f"{metadata_path}: '{key}' must be >= 1")
    return result


def _float(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def scan_raw_dataset(
    raw_root: Path,
    *,
    video_filename: str = "source_video.mp4",
    strict_structure: bool = True,
) -> DatasetScanReport:
    raw_root = Path(raw_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    records: list[SampleRecord] = []

    if not raw_root.is_dir():
        return DatasetScanReport([], [f"raw_root does not exist: {raw_root}"], [])

    metadata_files = sorted(raw_root.rglob("metadata.json"))
    if not metadata_files:
        return DatasetScanReport([], [f"No metadata.json found under: {raw_root}"], [])

    for metadata_path in metadata_files:
        sample_dir = metadata_path.parent
        landmarks_path = sample_dir / "landmarks.npz"
        video_path = sample_dir / video_filename

        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))

            sample_id = _required_str(meta, "sample_id", metadata_path)
            sign = _required_str(meta, "sign", metadata_path)
            subject_id = _required_str(meta, "subject_id", metadata_path)
            session_id = _required_str(meta, "session_id", metadata_path)
            frame_count = _required_int(meta, "frame_count", metadata_path)

            fps = _float(meta, "fps", 0.0)
            duration_s = _float(meta, "duration_s", 0.0)
            collection_mode = str(meta.get("collection_mode", "unknown"))
            dataset_version = str(meta.get("dataset_version", "unknown"))
            schema_version = str(meta.get("schema_version", "unknown"))

            if fps <= 0:
                errors.append(f"{metadata_path}: fps must be > 0")
            if duration_s < 0:
                errors.append(f"{metadata_path}: duration_s must be >= 0")
            if not landmarks_path.is_file():
                errors.append(f"{sample_dir}: missing landmarks.npz")

            try:
                rel = sample_dir.relative_to(raw_root)
                parts = rel.parts
            except ValueError:
                parts = ()

            if strict_structure:
                if len(parts) != 4:
                    errors.append(
                        f"{sample_dir}: expected raw/<sign>/<subject>/<session>/<sample>"
                    )
                else:
                    folder_sign, folder_subject, folder_session, folder_sample = parts
                    if folder_sign != sign:
                        errors.append(
                            f"{sample_dir}: folder sign '{folder_sign}' != metadata sign '{sign}'"
                        )
                    if folder_subject != subject_id:
                        errors.append(
                            f"{sample_dir}: folder subject '{folder_subject}' "
                            f"!= metadata subject '{subject_id}'"
                        )
                    if folder_session != session_id:
                        errors.append(
                            f"{sample_dir}: folder session '{folder_session}' "
                            f"!= metadata session '{session_id}'"
                        )
                    if folder_sample != sample_id:
                        errors.append(
                            f"{sample_dir}: folder sample '{folder_sample}' "
                            f"!= metadata sample_id '{sample_id}'"
                        )

            if not video_path.is_file():
                warnings.append(f"{sample_dir}: source video not found")

            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    sign=sign,
                    subject_id=subject_id,
                    session_id=session_id,
                    frame_count=frame_count,
                    fps=fps,
                    duration_s=duration_s,
                    collection_mode=collection_mode,
                    dataset_version=dataset_version,
                    schema_version=schema_version,
                    sample_path=sample_dir,
                    metadata_path=metadata_path,
                    landmarks_path=landmarks_path,
                    video_path=video_path,
                    has_video=video_path.is_file(),
                )
            )
        except Exception as exc:
            errors.append(f"{metadata_path}: {exc}")

    sample_ids: dict[str, list[Path]] = {}
    resolved_paths: dict[str, int] = {}

    for r in records:
        sample_ids.setdefault(r.sample_id, []).append(r.sample_path)
        key = str(r.sample_path.resolve())
        resolved_paths[key] = resolved_paths.get(key, 0) + 1

    for sample_id, paths in sample_ids.items():
        if len(paths) > 1:
            errors.append(
                f"duplicate sample_id '{sample_id}' in: "
                + ", ".join(str(p) for p in paths)
            )

    for path, count in resolved_paths.items():
        if count > 1:
            errors.append(f"duplicate sample path '{path}' x{count}")

    return DatasetScanReport(records=records, errors=errors, warnings=warnings)
