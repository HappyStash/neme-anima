import * as api from "$lib/api";
import type { ProjectListEntry, ProjectView } from "$lib/types";

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
    } catch (e) {
      this.error = String(e);
    } finally {
      this.loading = false;
    }
  }

  async create(name: string, folder: string) {
    const created = await api.createProject({ name, folder });
    this.active = created;
    await this.refresh();
    return created;
  }

  clearActive() {
    this.active = null;
  }
}

export const projectsStore = new ProjectsStore();
