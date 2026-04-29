import type { ServerEvent } from "./types";

export interface ConnectOptions {
  url: string;
  onEvent?: (event: ServerEvent) => void;
  onStatus?: (status: "open" | "closed" | "reconnecting") => void;
}

export interface Connection {
  close(): void;
}

const INITIAL_BACKOFF_MS = 250;
const MAX_BACKOFF_MS = 5000;

export function connectEvents(opts: ConnectOptions): Connection {
  let socket: WebSocket | null = null;
  let stopped = false;
  let backoff = INITIAL_BACKOFF_MS;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function open() {
    if (stopped) return;
    socket = new WebSocket(opts.url);
    socket.onopen = () => {
      backoff = INITIAL_BACKOFF_MS;
      opts.onStatus?.("open");
    };
    socket.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as ServerEvent;
        opts.onEvent?.(data);
      } catch {
        // Ignore malformed payloads — server contract owns the format.
      }
    };
    socket.onclose = () => {
      socket = null;
      if (stopped) return;
      opts.onStatus?.("reconnecting");
      timer = setTimeout(open, backoff);
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
    };
    socket.onerror = () => {
      socket?.close();
    };
  }

  open();

  return {
    close() {
      stopped = true;
      if (timer) clearTimeout(timer);
      socket?.close();
      opts.onStatus?.("closed");
    },
  };
}
