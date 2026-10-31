// Run with: npm test  (Node's built-in test runner; Node strips the TypeScript types)
import assert from "node:assert/strict";
import { test } from "node:test";

import { SseParser } from "./sse.ts";

test("parses complete events", () => {
  const p = new SseParser();
  const events = p.push('event: meta\ndata: {"a":1}\n\nevent: token\ndata: {"text":"hi"}\n\n');
  assert.deepEqual(events, [
    { event: "meta", data: '{"a":1}' },
    { event: "token", data: '{"text":"hi"}' },
  ]);
});

test("buffers events split across network chunks", () => {
  const p = new SseParser();
  assert.deepEqual(p.push("event: tok"), []);
  assert.deepEqual(p.push('en\ndata: {"text":'), []);
  assert.deepEqual(p.push('"x"}\n'), []);
  assert.deepEqual(p.push("\n"), [{ event: "token", data: '{"text":"x"}' }]);
});

test("defaults event name, joins multi-line data, ignores comments and CRLF", () => {
  const p = new SseParser();
  const events = p.push(": keep-alive\r\n\r\ndata: line1\r\ndata: line2\r\n\r\n");
  assert.deepEqual(events, [{ event: "message", data: "line1\nline2" }]);
});
