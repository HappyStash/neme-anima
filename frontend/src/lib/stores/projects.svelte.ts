import * as api from "$lib/api";
import type { ProjectListEntry, ProjectView } from "$lib/types";
import { viewStore } from "$lib/stores/view.svelte";

class ProjectsStore {
  list = $state<ProjectListEntry[]>([]);
  active = $state<ProjectView | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.list = await api.listProjects();
    } catch (e) {
      this.error = String(e);
    } finally {
      this.loading = false;
    }
  }

  async load(slug: string) {
    this.loading = true;
    this.error = null;
    try {
      this.active = await api.getProject(slug);
      this._syncActiveCharacter();
    } catch (e) {
      this.error = String(e);
    } finally {
      this.loading = false;
    }
  }

  async create(name: string, folder: string) {
    const created = await api.createProject({ name, folder });
    this.active = created;
    this._syncActiveCharacter();
    await this.refresh();
    return created;
  }

  clearActive() {
    this.active = null;
  }

  /** Pick a sensible active character after a project (re)loads.
   *
   *  Keeps the existing selection when it's still valid, so a refresh
   *  triggered by an unrelated mutation (adding a video, editing tags,
   *  etc.) doesn't yank the user back to the first character — that
   *  would silently swap the per-video ref strips out from under them.
   *  Falls back to the first character only when the previous selection
   *  was deleted or the project list was empty before this load. */
  private _syncActiveCharacter() {
    const project = this.active;
    if (!project || project.characters.length === 0) return;
    const slugs = new Set(project.characters.map((c) => c.slug));
    if (!slugs.has(viewStore.activeCharacterSlug)) {
      viewStore.activeCharacterSlug = project.characters[0].slug;
    }
  }

  /** Convenience accessor for the currently-active character record.
   *  Returns null when no project is loaded — components that surface
   *  per-character UI bail early on null. */
  get activeCharacter() {
    const project = this.active;
    if (!project) return null;
    return (
      project.characters.find((c) => c.slug === viewStore.activeCharacterSlug)
      ?? project.characters[0]
      ?? null
    );
  }
}

export const projectsStore = new ProjectsStore();
