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
    from neme_extractor.pipeline import run_extract

    run_extract(video=video, refs_dir=refs, out_root=out, config_path=config)


@app.command()
def rerun(
    out_dir: Path = typer.Argument(..., exists=True, file_okay=False,
                                   help="Existing output/<video_stem>/ folder to re-tune."),
) -> None:
    """Re-run selection / framing / tagging from cached detections with edited thresholds.json."""
    from neme_extractor.pipeline import run_rerun

    run_rerun(out_dir=out_dir)


if __name__ == "__main__":
    app()
