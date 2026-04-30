"""Tests for /api/projects/{slug}/frames routes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from httpx import ASGITransport, AsyncClient

from neme_extractor.server.app import create_app
from neme_extractor.storage.metadata import FrameRecord, MetadataLog
from neme_extractor.storage.project import Project


@pytest.fixture
def project_with_frames(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "p", name="p")
    # Two synthetic kept frames so listing has something to return.
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for stem, fi in [("ep01", 10), ("ep02", 20)]:
        name = f"{stem}__s000_t001_f{fi:06}"
        Image.fromarray(img).save(p.kept_dir / f"{name}.png")
        (p.kept_dir / f"{name}.txt").write_text("1girl, smile\n")
        MetadataLog(p.metadata_path).append(FrameRecord(
            filename=name, kept=True,
            scene_idx=0, tracklet_id=1, frame_idx=fi,
            timestamp_seconds=fi / 24.0,
            bbox=(0, 0, 16, 16),
            ccip_distance=0.1, sharpness=10.0, visibility=1.0, aspect=0.95,
            score=0.9, video_stem=stem,
        ))
    return p


@pytest.fixture
def app(tmp_path: Path, project_with_frames: Project):
    a = create_app(state_dir=tmp_path / "state")
    a.state.registry.register(project_with_frames)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_list_all_frames(client, project_with_frames: Project):
    resp = await client.get(f"/api/projects/{project_with_frames.slug}/frames")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    filenames = sorted(f["filename"] for f in body["items"])
    assert filenames[0].startswith("ep01__")
    assert filenames[1].startswith("ep02__")


async def test_list_hides_crop_derivatives(
    client, project_with_frames: Project,
) -> None:
    """`_crop` derivatives are an internal training-target replacement
    for their original; the frames grid only shows the user-facing rows."""
    name = "ep01__s000_t001_f000010"
    big = np.zeros((100, 200, 3), dtype=np.uint8)
    Image.fromarray(big).save(project_with_frames.kept_dir / f"{name}.png")
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop",
        json={"x": 10, "y": 20, "width": 80, "height": 60},
    )
    assert resp.status_code == 200
    # Crop appended a `_crop` record + wrote `_crop.png` to disk; the
    # listing must still only return the originals.
    resp = await client.get(f"/api/projects/{project_with_frames.slug}/frames")
    assert resp.status_code == 200
    filenames = [f["filename"] for f in resp.json()["items"]]
    assert all(not f.endswith("_crop") for f in filenames), filenames


async def test_list_filtered_by_source(client, project_with_frames: Project):
    resp = await client.get(
        f"/api/projects/{project_with_frames.slug}/frames",
        params={"source": "ep02"},
    )
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["filename"].startswith("ep02__")


async def test_get_tags(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    resp = await client.get(f"/api/projects/{project_with_frames.slug}/frames/{name}/tags")
    assert resp.status_code == 200
    assert resp.json()["text"] == "1girl, smile"


async def test_put_tags_overwrites(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    resp = await client.put(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/tags",
        json={"text": "1girl, blue_hair"},
    )
    assert resp.status_code == 200
    txt = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    assert txt == "1girl, blue_hair\n"


async def test_delete_frame_removes_png_and_txt(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    resp = await client.delete(f"/api/projects/{project_with_frames.slug}/frames/{name}")
    assert resp.status_code == 204
    assert not (project_with_frames.kept_dir / f"{name}.png").exists()
    assert not (project_with_frames.kept_dir / f"{name}.txt").exists()


async def test_bulk_delete(client, project_with_frames: Project):
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-delete",
        json={"filenames": [
            "ep01__s000_t001_f000010", "ep02__s000_t001_f000020",
        ]},
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert sorted(p.name for p in project_with_frames.kept_dir.iterdir()) == []


async def test_bulk_tags_replace_uses_regex(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    # Write a known tag set first.
    (project_with_frames.kept_dir / f"{name}.txt").write_text("red_eyes, blue_hair\n")
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-tags-replace",
        json={
            "filenames": [name],
            "pattern": r"red_eyes",
            "replacement": "ruby_eyes",
            "case_insensitive": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["changed"] >= 1
    text = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    assert "ruby_eyes" in text
    assert "red_eyes" not in text


async def test_bulk_tags_replace_invalid_regex_returns_422(
    client, project_with_frames: Project
):
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-tags-replace",
        json={"filenames": [], "pattern": "[unclosed", "replacement": ""},
    )
    assert resp.status_code == 422


async def test_bulk_tags_replace_only_touches_first_line(
    client, project_with_frames: Project,
):
    """The regex must match the danbooru tag line only — the LLM description
    on row two stays byte-identical so users can rewrite tags without losing
    captions written by a separate model."""
    name = "ep01__s000_t001_f000010"
    (project_with_frames.kept_dir / f"{name}.txt").write_text(
        "red_eyes, blue_hair\nA young woman with red eyes stands in a park.\n",
        encoding="utf-8",
    )
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-tags-replace",
        json={
            "filenames": [name],
            "pattern": r"red_eyes",
            "replacement": "ruby_eyes",
        },
    )
    assert resp.status_code == 200
    text = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    # Tag line rewritten…
    assert text.startswith("ruby_eyes, blue_hair\n")
    # …description line untouched (still says "red eyes").
    assert "A young woman with red eyes" in text


async def test_bulk_tags_replace_can_prepend_a_tag(
    client, project_with_frames: Project,
):
    """Demonstrates the "how to add tags" answer for the user: anchored
    regexes act as insertion points."""
    name = "ep01__s000_t001_f000010"
    (project_with_frames.kept_dir / f"{name}.txt").write_text(
        "1girl, smile\n", encoding="utf-8",
    )
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-tags-replace",
        json={
            "filenames": [name],
            "pattern": r"^",
            "replacement": "masterpiece, ",
        },
    )
    assert resp.status_code == 200
    text = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    assert text == "masterpiece, 1girl, smile\n"


async def test_bulk_retag_llm_422_when_no_model(
    client, project_with_frames: Project,
):
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-retag-llm",
        json={"filenames": ["ep01__s000_t001_f000010"]},
    )
    assert resp.status_code == 422


async def test_bulk_retag_danbooru_prefers_crop_derivative(
    client, app, project_with_frames: Project,
):
    """When a `_crop` derivative is on disk for an original, the WD14
    tagger must run on the cropped pixels but write the result to the
    *original's* sidecar — there is only ever one .txt per kept frame and
    training pairs the crop's image with that single sidecar at staging."""
    name = "ep01__s000_t001_f000010"
    # Distinct pixel values so the fake tagger can prove which image it saw.
    original = np.full((128, 128, 3), 200, dtype=np.uint8)
    crop = np.full((64, 64, 3), 50, dtype=np.uint8)
    Image.fromarray(original).save(project_with_frames.kept_dir / f"{name}.png")
    Image.fromarray(crop).save(project_with_frames.kept_dir / f"{name}_crop.png")
    (project_with_frames.kept_dir / f"{name}.txt").write_text(
        "old_orig_tags\nkeep_this_caption\n", encoding="utf-8",
    )

    seen_pixel_means: list[float] = []

    class FakeTagger:
        def tag(self, arr):
            seen_pixel_means.append(float(arr.mean()))

            class Result:
                text = "from_crop_image"

            return Result()

    app.state._tagger = FakeTagger()

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-retag-danbooru",
        json={"filenames": [name]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retagged"] == 1
    # The original's filename is the row the user sees — UI badge updates
    # target it directly (the crop row is filtered out of the grid).
    assert body["effective_filenames"] == [name]

    # The tagger saw the crop's pixels (~50), not the original's (~200).
    assert seen_pixel_means and seen_pixel_means[0] < 100

    # Original sidecar updated; description on row 2 preserved.
    orig_txt = (project_with_frames.kept_dir / f"{name}.txt").read_text(
        encoding="utf-8",
    )
    assert orig_txt.startswith("from_crop_image\n")
    assert "keep_this_caption" in orig_txt

    # No `_crop.txt` is created — the derivative is image-only.
    assert not (project_with_frames.kept_dir / f"{name}_crop.txt").exists()


async def test_bulk_retag_danbooru_uses_original_when_no_crop(
    client, app, project_with_frames: Project,
):
    """Sanity: original-only frames keep the existing single-sidecar
    behavior — the retarget rule only kicks in when a crop sibling exists."""
    name = "ep01__s000_t001_f000010"

    class FakeTagger:
        def tag(self, arr):  # noqa: D401, ARG002
            class Result:
                text = "wd14_only"

            return Result()

    app.state._tagger = FakeTagger()

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-retag-danbooru",
        json={"filenames": [name]},
    )
    assert resp.status_code == 200
    assert resp.json()["effective_filenames"] == [name]
    txt = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    assert txt.startswith("wd14_only\n")


async def test_bulk_retag_llm_prefers_crop_derivative(
    client, project_with_frames: Project, monkeypatch,
):
    """Same retarget rule for the LLM path: the description is generated
    from the crop's pixels but written to the original's sidecar — the
    crop is image-only and the original `.txt` is the single source of
    truth for tags + caption."""
    name = "ep01__s000_t001_f000010"

    # Configure a model so the route gets past its 422 guard.
    project_with_frames.llm.enabled = True
    project_with_frames.llm.model = "fake-model"
    project_with_frames.llm.endpoint = "http://localhost:1234"
    project_with_frames.save()

    original = np.full((128, 128, 3), 200, dtype=np.uint8)
    crop = np.full((64, 64, 3), 50, dtype=np.uint8)
    Image.fromarray(original).save(project_with_frames.kept_dir / f"{name}.png")
    Image.fromarray(crop).save(project_with_frames.kept_dir / f"{name}_crop.png")
    (project_with_frames.kept_dir / f"{name}.txt").write_text(
        "orig_tags\nold_orig_caption\n", encoding="utf-8",
    )

    seen_image_paths: list[Path] = []
    seen_danbooru: list[str | None] = []

    def fake_describe_image(*, endpoint, model, image_path, prompt, danbooru_tags):
        seen_image_paths.append(image_path)
        seen_danbooru.append(danbooru_tags)
        return "Description of the cropped subject."

    monkeypatch.setattr(
        "neme_extractor.llm.describe_image", fake_describe_image,
    )

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-retag-llm",
        json={"filenames": [name]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["described"] == 1
    assert body["effective_filenames"] == [name]

    # describe_image was passed the crop's PNG path, not the original.
    assert seen_image_paths == [
        project_with_frames.kept_dir / f"{name}_crop.png",
    ]
    # Danbooru hint comes from the original's sidecar (the only one).
    assert seen_danbooru == ["orig_tags"]

    # Original sidecar got the new description; tag line preserved.
    orig_txt = (project_with_frames.kept_dir / f"{name}.txt").read_text(
        encoding="utf-8",
    )
    assert orig_txt.startswith("orig_tags\n")
    assert "Description of the cropped subject." in orig_txt

    # No `_crop.txt` is created — the derivative is image-only.
    assert not (project_with_frames.kept_dir / f"{name}_crop.txt").exists()


async def test_bulk_retag_llm_resolves_crop_filename_to_original(
    client, project_with_frames: Project, monkeypatch,
):
    """If a `_crop` filename is passed directly (legacy callers, manual
    API hits), we treat it as the original it derives from: tag the crop
    image, write to the original's sidecar, and report the original as
    the effective filename. There is no `_crop_crop` ghost."""
    name = "ep01__s000_t001_f000010"
    crop_name = f"{name}_crop"

    project_with_frames.llm.enabled = True
    project_with_frames.llm.model = "fake-model"
    project_with_frames.save()

    original = np.full((64, 64, 3), 200, dtype=np.uint8)
    crop = np.full((64, 64, 3), 80, dtype=np.uint8)
    Image.fromarray(original).save(project_with_frames.kept_dir / f"{name}.png")
    Image.fromarray(crop).save(project_with_frames.kept_dir / f"{crop_name}.png")
    (project_with_frames.kept_dir / f"{name}.txt").write_text(
        "orig_tags\n", encoding="utf-8",
    )

    captured: list[Path] = []

    def fake_describe_image(*, endpoint, model, image_path, prompt, danbooru_tags):
        captured.append(image_path)
        return "ok"

    monkeypatch.setattr(
        "neme_extractor.llm.describe_image", fake_describe_image,
    )

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-retag-llm",
        json={"filenames": [crop_name]},
    )
    assert resp.status_code == 200
    assert resp.json()["effective_filenames"] == [name]
    # Crop pixels are what we asked the LLM to describe.
    assert captured == [project_with_frames.kept_dir / f"{crop_name}.png"]
    # Description landed on the original's sidecar, not a `_crop.txt`.
    orig_txt = (project_with_frames.kept_dir / f"{name}.txt").read_text(
        encoding="utf-8",
    )
    assert "ok" in orig_txt
    assert not (project_with_frames.kept_dir / f"{crop_name}.txt").exists()


async def test_list_frames_has_description_flag(
    client, project_with_frames: Project,
):
    """The grid uses this flag to render an at-a-glance "described" badge —
    must reflect line 2 presence, not just file existence."""
    described = "ep01__s000_t001_f000010"
    plain = "ep02__s000_t001_f000020"
    (project_with_frames.kept_dir / f"{described}.txt").write_text(
        "1girl, smile\nA young woman smiling against a wood panel wall.\n",
        encoding="utf-8",
    )
    (project_with_frames.kept_dir / f"{plain}.txt").write_text(
        "1girl, smile\n", encoding="utf-8",
    )
    resp = await client.get(f"/api/projects/{project_with_frames.slug}/frames")
    assert resp.status_code == 200
    by_name = {f["filename"]: f for f in resp.json()["items"]}
    assert by_name[described]["has_description"] is True
    assert by_name[plain]["has_description"] is False


async def test_get_description_returns_only_second_line(
    client, project_with_frames: Project,
):
    name = "ep01__s000_t001_f000010"
    (project_with_frames.kept_dir / f"{name}.txt").write_text(
        "1girl, smile\nA young woman smiling against a wood panel wall.\n",
        encoding="utf-8",
    )
    resp = await client.get(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/description"
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "A young woman smiling against a wood panel wall."


async def test_put_description_preserves_danbooru_line(
    client, project_with_frames: Project,
):
    name = "ep01__s000_t001_f000010"
    (project_with_frames.kept_dir / f"{name}.txt").write_text(
        "1girl, smile\nold description\n", encoding="utf-8",
    )
    resp = await client.put(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/description",
        json={"text": "brand new caption"},
    )
    assert resp.status_code == 200
    text = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    assert text == "1girl, smile\nbrand new caption\n"


async def test_put_description_empty_collapses_to_one_line(
    client, project_with_frames: Project,
):
    """Clearing the description must round-trip back to the single-line
    sidecar form so files written before LLM tagging stay byte-clean."""
    name = "ep01__s000_t001_f000010"
    (project_with_frames.kept_dir / f"{name}.txt").write_text(
        "1girl, smile\nold description\n", encoding="utf-8",
    )
    resp = await client.put(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/description",
        json={"text": ""},
    )
    assert resp.status_code == 200
    text = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    assert text == "1girl, smile\n"


async def test_get_frame_image(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    resp = await client.get(f"/api/projects/{project_with_frames.slug}/frames/{name}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 0


async def test_crop_creates_image_only_derivative(
    client, project_with_frames: Project, tmp_path: Path,
) -> None:
    """Cropping produces a NEW image-only derivative and leaves the source
    untouched. Tags stay on the original's `<name>.txt` — there is only
    ever one .txt per kept frame, and training pairs the crop with that
    sidecar at staging time."""
    # Replace the 16×16 fixture image with a larger one so we can crop a
    # meaningful sub-rectangle and verify dimensions.
    name = "ep01__s000_t001_f000010"
    big = np.zeros((100, 200, 3), dtype=np.uint8)
    Image.fromarray(big).save(project_with_frames.kept_dir / f"{name}.png")
    orig_txt = project_with_frames.kept_dir / f"{name}.txt"
    orig_text_before = orig_txt.read_text(encoding="utf-8")

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop",
        json={"x": 10, "y": 20, "width": 80, "height": 60},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == f"{name}_crop"
    assert body["video_stem"] == "ep01"

    # Original still on disk.
    assert (project_with_frames.kept_dir / f"{name}.png").exists()
    # New cropped derivative on disk with the right size.
    new_png = project_with_frames.kept_dir / f"{name}_crop.png"
    assert new_png.exists()
    with Image.open(new_png) as im:
        assert im.size == (80, 60)
    # No `_crop.txt` is created — crop is image-only.
    assert not (project_with_frames.kept_dir / f"{name}_crop.txt").exists()
    # Original sidecar untouched.
    assert orig_txt.read_text(encoding="utf-8") == orig_text_before


async def test_crop_removes_legacy_crop_txt(
    client, project_with_frames: Project,
) -> None:
    """Projects from before the image-only-derivative change may still
    have a `<name>_crop.txt` on disk. Re-cropping must wipe it so the
    trainer-staging step doesn't see a phantom paired sidecar."""
    name = "ep01__s000_t001_f000010"
    big = np.zeros((100, 200, 3), dtype=np.uint8)
    Image.fromarray(big).save(project_with_frames.kept_dir / f"{name}.png")
    legacy = project_with_frames.kept_dir / f"{name}_crop.txt"
    legacy.write_text("stale tags\n", encoding="utf-8")

    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop",
        json={"x": 0, "y": 0, "width": 50, "height": 50},
    )
    assert resp.status_code == 200
    assert not legacy.exists()


async def test_crop_clamps_oob_rectangle(
    client, project_with_frames: Project,
) -> None:
    name = "ep01__s000_t001_f000010"
    big = np.zeros((100, 100, 3), dtype=np.uint8)
    Image.fromarray(big).save(project_with_frames.kept_dir / f"{name}.png")
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop",
        json={"x": 90, "y": 90, "width": 999, "height": 999},
    )
    assert resp.status_code == 200
    new_png = project_with_frames.kept_dir / f"{name}_crop.png"
    with Image.open(new_png) as im:
        # Clamped to the bottom-right 10×10 corner.
        assert im.size == (10, 10)


async def test_crop_404_for_unknown_frame(
    client, project_with_frames: Project,
) -> None:
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/nope/crop",
        json={"x": 0, "y": 0, "width": 10, "height": 10},
    )
    assert resp.status_code == 404


async def test_crop_overwrites_previous_derivative(
    client, project_with_frames: Project,
) -> None:
    """Re-cropping the same original must overwrite the derivative (single
    crop per original) and update the .crop.json sidecar — otherwise the
    user would accumulate _crop1/_crop2/... and lose the round-trip."""
    name = "ep01__s000_t001_f000010"
    big = np.zeros((100, 200, 3), dtype=np.uint8)
    Image.fromarray(big).save(project_with_frames.kept_dir / f"{name}.png")
    url = f"/api/projects/{project_with_frames.slug}/frames/{name}/crop"

    r1 = await client.post(url, json={"x": 0, "y": 0, "width": 50, "height": 50})
    assert r1.status_code == 200
    r2 = await client.post(url, json={"x": 10, "y": 20, "width": 80, "height": 60})
    assert r2.status_code == 200

    # Same filename for both responses — we never produce _crop2.
    assert r1.json()["filename"] == r2.json()["filename"] == f"{name}_crop"
    # Derivative reflects the LATEST crop dimensions.
    new_png = project_with_frames.kept_dir / f"{name}_crop.png"
    with Image.open(new_png) as im:
        assert im.size == (80, 60)
    # Sidecar reflects the latest rect.
    spec = project_with_frames.kept_dir / f"{name}.crop.json"
    import json as _json
    assert _json.loads(spec.read_text()) == {
        "x": 10, "y": 20, "width": 80, "height": 60,
    }


async def test_get_crop_rect_returns_saved_rectangle(
    client, project_with_frames: Project,
) -> None:
    name = "ep01__s000_t001_f000010"
    big = np.zeros((100, 200, 3), dtype=np.uint8)
    Image.fromarray(big).save(project_with_frames.kept_dir / f"{name}.png")
    await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop",
        json={"x": 10, "y": 20, "width": 80, "height": 60},
    )
    resp = await client.get(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop"
    )
    assert resp.status_code == 200
    assert resp.json() == {"x": 10, "y": 20, "width": 80, "height": 60}


async def test_get_crop_rect_404_when_no_crop(
    client, project_with_frames: Project,
) -> None:
    """The modal uses the 404 as the "no overlay, start full-image" signal."""
    name = "ep01__s000_t001_f000010"
    resp = await client.get(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop"
    )
    assert resp.status_code == 404


async def test_deleting_original_also_clears_crop_artifacts(
    client, project_with_frames: Project,
) -> None:
    """Otherwise an orphaned sidecar would attach itself to the next frame
    that happens to be added with the same filename."""
    name = "ep01__s000_t001_f000010"
    big = np.zeros((100, 200, 3), dtype=np.uint8)
    Image.fromarray(big).save(project_with_frames.kept_dir / f"{name}.png")
    await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop",
        json={"x": 10, "y": 20, "width": 80, "height": 60},
    )
    deriv = project_with_frames.kept_dir / f"{name}_crop.png"
    spec = project_with_frames.kept_dir / f"{name}.crop.json"
    assert deriv.exists() and spec.exists()

    resp = await client.delete(
        f"/api/projects/{project_with_frames.slug}/frames/{name}"
    )
    assert resp.status_code == 204
    assert not deriv.exists()
    assert not spec.exists()


async def test_deleting_derivative_clears_only_sidecar(
    client, project_with_frames: Project,
) -> None:
    """Deleting just the derivative must remove the saved rect (so reopening
    the original starts clean) but leave the original alone."""
    name = "ep01__s000_t001_f000010"
    big = np.zeros((100, 200, 3), dtype=np.uint8)
    Image.fromarray(big).save(project_with_frames.kept_dir / f"{name}.png")
    await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/crop",
        json={"x": 10, "y": 20, "width": 80, "height": 60},
    )
    spec = project_with_frames.kept_dir / f"{name}.crop.json"
    assert spec.exists()

    resp = await client.delete(
        f"/api/projects/{project_with_frames.slug}/frames/{name}_crop"
    )
    assert resp.status_code == 204
    assert not spec.exists()
    # Original untouched.
    assert (project_with_frames.kept_dir / f"{name}.png").exists()
