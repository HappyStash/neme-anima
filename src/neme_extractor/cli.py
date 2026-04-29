"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="neme-extractor",
    help="Extract anime character crops from video for LoRA training.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def extract(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
                                 help="Source video file."),
    refs: Path = typer.Argument(..., exists=True, file_okay=False, readable=True,
                                help="Folder containing reference images of the target character."),
    out: Path = typer.Option(Path("output"), "--out", "-o",
                             help="Output root directory."),
    config: Path | None = typer.Option(None, "--config", "-c",
                                       help="Optional thresholds.json to override defaults."),
) -> None:
    """Run the full extraction pipeline on a video."""
    raise NotImplementedError(
        "The legacy extract command has been replaced by the project-centric API. "
        "Use 'neme-extractor project extract' (Task 8)."
    )


@app.command()
def rerun(
    out_dir: Path = typer.Argument(..., exists=True, file_okay=False,
                                   help="Existing output/<video_stem>/ folder to re-tune."),
) -> None:
    """Re-run selection / framing / tagging from cached detections with edited thresholds.json."""
    raise NotImplementedError(
        "The legacy rerun command has been replaced by the project-centric API. "
        "Use 'neme-extractor project rerun' (Task 8)."
    )


if __name__ == "__main__":
    app()
