"""Typer CLI: project subcommand group + extract/rerun."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from neme_extractor.storage.project import Project

app = typer.Typer(
    name="neme-extractor",
    help="Extract anime character crops from video for LoRA training.",
    no_args_is_help=True,
    add_completion=False,
)
project_app = typer.Typer(name="project", help="Manage projects (sources, refs, runs).")
app.add_typer(project_app, name="project")
console = Console()


@project_app.command("create")
def project_create(
    folder: Path = typer.Argument(..., help="Path of the project folder to create."),
    name: str = typer.Option(..., "--name", "-n", help="Display name."),
) -> None:
    """Create a new project folder."""
    try:
        p = Project.create(folder, name=name)
    except FileExistsError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)
    console.print(f"[green]created[/green] {p.root}  name={p.name}  slug={p.slug}")


@project_app.command("add-video")
def project_add_video(
    project_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Append a video to the project's sources."""
    p = Project.load(project_dir)
    s = p.add_source(video)
    console.print(f"[green]+ source[/green] {Path(s.path).name}")


@project_app.command("add-ref")
def project_add_ref(
    project_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Append a reference image to the project."""
    p = Project.load(project_dir)
    r = p.add_ref(image)
    console.print(f"[green]+ ref[/green] {Path(r.path).name}")


@project_app.command("extract")
def project_extract(
    project_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    video: str | None = typer.Option(None, "--video", "-v",
                                     help="Video stem to extract; default = all sources sequentially."),
) -> None:
    """Run extraction on one or all sources in this project."""
    from neme_extractor.pipeline import run_extract

    p = Project.load(project_dir)
    indices = (
        [i for i, s in enumerate(p.sources) if Path(s.path).stem == video]
        if video else list(range(len(p.sources)))
    )
    if not indices:
        console.print(f"[red]error:[/red] no matching source")
        raise typer.Exit(code=1)
    for i in indices:
        run_extract(project=p, source_idx=i)


@project_app.command("rerun")
def project_rerun(
    project_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    video: str = typer.Option(..., "--video", "-v", help="Video stem to rerun."),
) -> None:
    """Re-run with cached detections + current thresholds."""
    from neme_extractor.pipeline import run_rerun

    p = Project.load(project_dir)
    run_rerun(project=p, video_stem=video)


if __name__ == "__main__":
    app()
