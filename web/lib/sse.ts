// Incremental parser for text/event-stream bodies read through fetch().
// EventSource can't be used because it only supports GET requests and /ask is a POST.

export interface SseEvent {
  event: string;
  data: string;
}

export class SseParser {
  private buffer = "";

  /** Feed a decoded chunk; returns every event completed by it. */
  push(chunk: string): SseEvent[] {
    this.buffer = (this.buffer + chunk).replace(/\r\n/g, "\n");
    const events: SseEvent[] = [];
    let end: number;
    while ((end = this.buffer.indexOf("\n\n")) !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line === "" || line.startsWith(":")) continue; // comment / keep-alive
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
        if (field === "event") event = value;
        else if (field === "data") data.push(value);
      }
      if (data.length > 0) events.push({ event, data: data.join("\n") });
    }
    return events;
  }
}
