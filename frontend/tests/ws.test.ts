import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { connectEvents } from "../src/lib/ws";

class FakeWebSocket {
  static last: FakeWebSocket | null = null;
  onopen: ((ev: any) => void) | null = null;
  onmessage: ((ev: any) => void) | null = null;
  onclose: ((ev: any) => void) | null = null;
  onerror: ((ev: any) => void) | null = null;
  readyState = 0;
  closed = false;
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.last = this;
  }
  open() { this.readyState = 1; this.onopen?.({}); }
  emit(data: string) { this.onmessage?.({ data }); }
  close() { this.closed = true; this.readyState = 3; this.onclose?.({ code: 1000 }); }
}

describe("ws client", () => {
  beforeEach(() => {
    (globalThis as any).WebSocket = FakeWebSocket;
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("delivers parsed events to handlers", () => {
    const events: any[] = [];
    const conn = connectEvents({ url: "/api/ws", onEvent: (e) => events.push(e) });
    FakeWebSocket.last!.open();
    FakeWebSocket.last!.emit(JSON.stringify({ type: "job.log", payload: { line: "hi" } }));
    expect(events).toEqual([{ type: "job.log", payload: { line: "hi" } }]);
    conn.close();
  });

  it("reconnects on close with backoff", () => {
    const conn = connectEvents({ url: "/api/ws" });
    const firstSocket = FakeWebSocket.last!;
    firstSocket.open();
    firstSocket.close();
    vi.advanceTimersByTime(300);
    expect(FakeWebSocket.last).not.toBe(firstSocket);
    conn.close();
  });

  it("close() stops auto-reconnect", () => {
    const conn = connectEvents({ url: "/api/ws" });
    const firstSocket = FakeWebSocket.last!;
    firstSocket.open();
    conn.close();
    firstSocket.close();
    vi.advanceTimersByTime(5000);
    expect(FakeWebSocket.last).toBe(firstSocket);
  });
});
