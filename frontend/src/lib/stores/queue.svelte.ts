import * as api from "$lib/api";
import type { QueueItem, ServerEvent } from "$lib/types";

class QueueStore {
  items = $state<QueueItem[]>([]);
  status = $state<"open" | "closed" | "reconnecting">("closed");

  async refresh() {
    this.items = await api.listQueue();
  }

  ingest(event: ServerEvent) {
    if (event.type === "queue.update") {
      const queue = event.payload.queue as QueueItem[] | undefined;
      if (queue) this.items = queue;
    }
  }

  setStatus(s: "open" | "closed" | "reconnecting") {
    this.status = s;
  }

  running(): QueueItem | null {
    return this.items.find(i => i.status === "running") ?? null;
  }

  pendingCount(): number {
    return this.items.filter(i => i.status === "pending").length;
  }

  totalActive(): number {
    return this.items.filter(i => i.status === "pending" || i.status === "running").length;
  }
}

export const queueStore = new QueueStore();
