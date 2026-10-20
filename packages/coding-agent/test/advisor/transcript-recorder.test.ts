/**
 * Contracts: AdvisorTranscriptRecorder persists the advisor agent's turns to a
 * subagent-style JSONL (`<session>/__advisor.jsonl`) so the advisor model's usage
 * is attributed in stats and its transcript shows in the Agent Hub.
 *
 * - Assistant turns land as `{type:"message", message:{role:"assistant", usage}}`
 *   entries — exactly the shape the stats parser reads for usage.
 * - User deltas are persisted but flagged `synthetic`/agent-attributed so stats'
 *   user-message metrics skip them.
 * - Non-conversational message kinds are not persisted.
 * - The target follows the session file: a switch routes later turns to the new
 *   session's `__advisor.jsonl`, leaving the prior file intact.
 */
import { describe, expect, it, vi } from "bun:test";
import * as fsSync from "node:fs";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, Message } from "@oh-my-pi/pi-ai";
import {
	ADVISOR_TRANSCRIPT_FILENAME,
	AdvisorTranscriptRecorder,
	advisorTranscriptFilename,
	loadAdvisorTranscriptCosts,
} from "@oh-my-pi/pi-coding-agent/advisor/transcript-recorder";
import { parseSessionContent } from "@oh-my-pi/pi-coding-agent/session/session-loader";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { FileSessionStorage } from "@oh-my-pi/pi-coding-agent/session/session-storage";
import * as sessionStorage from "@oh-my-pi/pi-coding-agent/session/session-storage";
import * as dirs from "@oh-my-pi/pi-utils";
import { removeWithRetries } from "@oh-my-pi/pi-utils";

interface AdvisorEntry {
	type?: string;
	id?: unknown;
	message?: {
		role?: string;
		model?: string;
		usage?: { input?: number };
		synthetic?: boolean;
		attribution?: string;
	};
}

async function withTempDir<T>(fn: (dir: string) => Promise<T>): Promise<T> {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "advisor-recorder-"));
	try {
		return await fn(dir);
	} finally {
		await removeWithRetries(dir);
	}
}

/** Parse the message entries (skipping the session header) from an advisor JSONL. */
async function readMessageEntries(file: string): Promise<AdvisorEntry[]> {
	const text = await Bun.file(file).text();
	// JSON.parse returns `any`; assigning to the typed array narrows reads below.
	const entries: AdvisorEntry[] = text
		.trim()
		.split("\n")
		.map(line => JSON.parse(line));
	return entries.filter(entry => entry.type === "message");
}

function assistantMessage(text: string, inputTokens: number, cost = 0, provider = "anthropic"): AgentMessage {
	const message = {
		role: "assistant" as const,
		content: [{ type: "text" as const, text }],
		api: "anthropic-messages",
		provider,
		model: "test-advisor-model",
		usage: {
			input: inputTokens,
			output: 3,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: inputTokens + 3,
			cost: { input: 0, output: cost, cacheRead: 0, cacheWrite: 0, total: cost },
		},
		stopReason: "stop" as const,
		timestamp: 1,
	};
	return message as unknown as AgentMessage;
}

function userMessage(text: string): AgentMessage {
	const message = { role: "user" as const, content: [{ type: "text" as const, text }], timestamp: 1 };
	return message as unknown as AgentMessage;
}

function developerMessage(text: string): AgentMessage {
	const message = { role: "developer" as const, content: [{ type: "text" as const, text }], timestamp: 1 };
	return message as unknown as AgentMessage;
}

function largeToolResult(): Message {
	return {
		role: "toolResult",
		toolCallId: "advisor-tool",
		toolName: "read",
		content: [
			{ type: "text", text: "tool output ".repeat(60_000) },
			{ type: "image", data: Buffer.alloc(1500, 23).toString("base64"), mimeType: "image/png" },
		],
		details: { output: "detail ".repeat(100_000) },
		isError: false,
		timestamp: 1,
	};
}

function copilotMessage(text: string, inputTokens = 1, cost = 0): AssistantMessage {
	return {
		...(assistantMessage(text, inputTokens, cost) as AssistantMessage),
		api: "openai-responses",
		provider: "github-copilot",
		content: [
			{ type: "thinking", thinking: text, thinkingSignature: `signature-${text}` },
			{ type: "text", text },
		],
		providerPayload: {
			type: "openaiResponsesHistory",
			provider: "github-copilot",
			items: [{ type: "message", role: "assistant", content: [{ type: "output_text", text }] }],
		},
	};
}

async function normalizedTranscript(file: string): Promise<unknown> {
	const text = await Bun.file(file).text();
	const { entries, titleSlot, malformedRecords } = parseSessionContent(text);
	expect(malformedRecords).toBe(0);
	const ids = new Map(entries.map((entry, index) => [entry.id, index]));
	expect(ids.size).toBe(entries.length);
	return {
		titleSlot: titleSlot && { ...titleSlot, updatedAt: "<timestamp>" },
		entries: entries.map((entry, index) => {
			expect(typeof entry.id).toBe("string");
			expect(Number.isNaN(Date.parse(entry.timestamp))).toBe(false);
			if (entry.type === "session") return { ...entry, id: index, timestamp: "<timestamp>" };
			if (entry.parentId !== null) expect(ids.has(entry.parentId)).toBe(true);
			return {
				...entry,
				id: index,
				parentId: entry.parentId === null ? null : ids.get(entry.parentId),
				timestamp: "<timestamp>",
			};
		}),
	};
}

/** Do not let a strong local in the test body mask the recorder's retention. */
function recordCollectibleMessages(recorder: AdvisorTranscriptRecorder): WeakRef<object>[] {
	const assistant = assistantMessage("collectible assistant", 1);
	const tool = largeToolResult();
	const user = { role: "user" as const, content: [{ type: "text" as const, text: "collectible user" }], timestamp: 1 };
	recorder.record(assistant);
	recorder.record(tool);
	recorder.record(user);
	return [new WeakRef(assistant), new WeakRef(tool), new WeakRef(user.content)];
}

function recordCollectibleRejectedMessages(recorder: AdvisorTranscriptRecorder): WeakRef<object>[] {
	const refs: WeakRef<object>[] = [];
	for (let index = 0; index < 20; index++) {
		const message = largeToolResult();
		refs.push(new WeakRef(message));
		recorder.record(message);
	}
	return refs;
}

// GC and JSC's optimizing worklist need real macrotask turns; fake timers cannot
// drive object collection. Poll the condition, as in parked-subagent-session-release.
async function collected(ref: WeakRef<object>, deadlineMs: number): Promise<boolean> {
	const deadline = Date.now() + deadlineMs;
	for (;;) {
		Bun.gc(true);
		if (ref.deref() === undefined) return true;
		if (Date.now() > deadline) return false;
		await Bun.sleep(100);
	}
}

describe("AdvisorTranscriptRecorder", () => {
	it("matches SessionManager disk output across large results, image blobs, resume and queued file switches", async () => {
		await withTempDir(async dir => {
			const blobs = vi.spyOn(dirs, "getBlobsDir").mockReturnValue(path.join(dir, "blobs"));
			let sessionFile = path.join(dir, "actual", "first.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			try {
				const baselineFirst = path.join(dir, "baseline", "first", ADVISOR_TRANSCRIPT_FILENAME);
				const seed = await SessionManager.open(baselineFirst, undefined, undefined, {
					initialCwd: dir,
					suppressBreadcrumb: true,
				});
				seed.appendMessage(assistantMessage("existing turn", 3) as Message);
				await seed.close();
				const actualFirst = path.join(dir, "actual", "first", ADVISOR_TRANSCRIPT_FILENAME);
				await Bun.write(actualFirst, Bun.file(baselineFirst));
				const sequence: { target: string; message: Message }[] = [
					{
						target: "first",
						message: {
							...(userMessage("session update") as Message),
							synthetic: true,
							attribution: "agent",
						} as Message,
					},
					{ target: "first", message: largeToolResult() },
					{ target: "first", message: assistantMessage("before switch", 4) as Message },
					{ target: "second", message: assistantMessage("new session", 5) as Message },
					{ target: "first", message: assistantMessage("resumed session", 6) as Message },
				];
				// Queue all switches before any asynchronous open completes.
				for (const { target, message } of sequence) {
					sessionFile = path.join(dir, "actual", `${target}.jsonl`);
					recorder.record(message);
				}
				await recorder.flush();
				for (const { target, message } of sequence) {
					const manager = await SessionManager.open(
						path.join(dir, "baseline", target, ADVISOR_TRANSCRIPT_FILENAME),
						undefined,
						undefined,
						{ initialCwd: dir, suppressBreadcrumb: true },
					);
					manager.appendMessage(message);
					await manager.close();
				}
				for (const target of ["first", "second"]) {
					expect(
						await normalizedTranscript(path.join(dir, "actual", target, ADVISOR_TRANSCRIPT_FILENAME)),
					).toEqual(await normalizedTranscript(path.join(dir, "baseline", target, ADVISOR_TRANSCRIPT_FILENAME)));
				}
				const output = await Bun.file(actualFirst).text();
				expect(output).toContain("blob:sha256:");
				expect(output).toContain("[Session persistence truncated large content]");
				expect(output).not.toContain("tool output ".repeat(60_000));
			} finally {
				await recorder.close();
				blobs.mockRestore();
			}
		});
	});

	it("releases durable message objects while the recorder and its writer stay live", async () => {
		await withTempDir(async dir => {
			const blobs = vi.spyOn(dirs, "getBlobsDir").mockReturnValue(path.join(dir, "blobs"));
			const recorder = new AdvisorTranscriptRecorder(
				() => path.join(dir, "sess.jsonl"),
				() => dir,
			);
			try {
				const refs = recordCollectibleMessages(recorder);
				await recorder.flush();
				for (const ref of refs) expect(await collected(ref, 15_000)).toBe(true);
				recorder.record(assistantMessage("still live after collection", 2));
				await recorder.flush();
				const entries = await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME));
				expect(entries.map(entry => entry.message?.role)).toEqual(["assistant", "toolResult", "user", "assistant"]);
			} finally {
				await recorder.close();
				blobs.mockRestore();
			}
		});
	}, 60_000);

	it.each([1, 2])("preserves parent links when resuming version %i transcripts", async version => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const entries = [
				{ type: "session", id: "advisor", version, timestamp: new Date(0).toISOString(), cwd: dir },
				{
					type: "message",
					id: "existing",
					parentId: null,
					timestamp: new Date(0).toISOString(),
					message: assistantMessage("old", 1),
				},
			];
			await Bun.write(transcript, `${entries.map(entry => JSON.stringify(entry)).join("\n")}\n`);
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(assistantMessage("new", 2));
			await recorder.close();
			const loaded = parseSessionContent(await Bun.file(transcript).text()).entries;
			expect(loaded).toHaveLength(3);
			expect(loaded[0]).toMatchObject({ type: "session", version: 3 });
			expect(loaded[2]).toMatchObject({ parentId: loaded[1].id });
		});
	});

	it("keeps the prior history when an ownership claim moves the writer to a sibling", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const seed = await SessionManager.open(transcript, undefined, undefined, {
				initialCwd: dir,
				suppressBreadcrumb: true,
			});
			seed.appendMessage(assistantMessage("existing turn", 1) as Message);
			await seed.close();
			const previous = await Bun.file(transcript).text();
			const claim = vi
				.spyOn(FileSessionStorage.prototype, "claimSession")
				.mockImplementationOnce(() => null)
				.mockImplementationOnce(() => null);
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			try {
				recorder.record(assistantMessage("moved turn", 2));
				await recorder.flush();
				const siblings = (await fs.readdir(path.dirname(transcript))).filter(
					name => name.endsWith(".jsonl") && name !== ADVISOR_TRANSCRIPT_FILENAME,
				);
				expect(siblings).toHaveLength(1);
				expect(await Bun.file(transcript).text()).toBe(previous);
				const loaded = parseSessionContent(
					await Bun.file(path.join(path.dirname(transcript), siblings[0])).text(),
				).entries;
				expect(loaded).toHaveLength(3);
				expect(loaded[0]).toMatchObject({ parentSession: seed.getSessionId() });
				expect(loaded[2]).toMatchObject({ parentId: loaded[1].id });
			} finally {
				await recorder.close();
				claim.mockRestore();
			}
		});
	});

	it("recovers a failed append without dropping previously evicted history", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const seed = await SessionManager.open(transcript, undefined, undefined, {
				initialCwd: dir,
				suppressBreadcrumb: true,
			});
			seed.appendMessage(assistantMessage("existing turn", 1) as Message);
			await seed.close();
			const originalOpenWriter = FileSessionStorage.prototype.openWriter;
			const openWriter = vi
				.spyOn(FileSessionStorage.prototype, "openWriter")
				.mockImplementationOnce(function (this: FileSessionStorage, file, options) {
					const writer = originalOpenWriter.call(this, file, options);
					vi.spyOn(writer, "appendSync").mockImplementationOnce(() => {
						const originalSize = fsSync.statSync(file).size;
						fsSync.appendFileSync(file, "{partial append");
						fsSync.truncateSync(file, originalSize);
						throw new Error("append failure");
					});
					return writer;
				});
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			try {
				recorder.record(assistantMessage("failed turn", 2));
				await expect(recorder.flush()).rejects.toThrow("append failure");
				recorder.record(assistantMessage("recovery turn", 3));
				await recorder.flush();
				const loaded = parseSessionContent(await Bun.file(transcript).text()).entries;
				expect(loaded).toHaveLength(4);
				for (let index = 2; index < loaded.length; index++) {
					expect(loaded[index]).toMatchObject({ parentId: loaded[index - 1].id });
				}
				expect((await readMessageEntries(transcript)).map(entry => entry.message?.usage?.input)).toEqual([1, 2, 3]);
			} finally {
				await recorder.close();
				openWriter.mockRestore();
			}
		});
	});

	it("repairs a damaged live tail before appending the next turn", async () => {
		await withTempDir(async dir => {
			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const recorder = new AdvisorTranscriptRecorder(
				() => path.join(dir, "sess.jsonl"),
				() => dir,
			);
			try {
				recorder.record(assistantMessage("before damage", 1));
				await recorder.flush();
				await fs.appendFile(transcript, "{damaged tail without newline");
				recorder.record(assistantMessage("after damage", 2));
				await recorder.flush();
				recorder.record(assistantMessage("keeps recording", 3));
				await recorder.flush();
				expect((await readMessageEntries(transcript)).map(entry => entry.message?.usage?.input)).toEqual([1, 2, 3]);
				await normalizedTranscript(transcript);
			} finally {
				await recorder.close();
			}
		});
	});

	it("keeps migrated v1 ids and parents stable after the open-time rewrite fails", async () => {
		await withTempDir(async dir => {
			const transcript = path.join(dir, ADVISOR_TRANSCRIPT_FILENAME);
			await Bun.write(
				transcript,
				[
					{ type: "session", id: "legacy-advisor", version: 1, cwd: dir, timestamp: new Date(0).toISOString() },
					{ type: "message", message: assistantMessage("old first", 1), timestamp: new Date(0).toISOString() },
					{ type: "message", message: assistantMessage("old second", 2), timestamp: new Date(0).toISOString() },
				]
					.map(entry => JSON.stringify(entry))
					.join("\n") + "\n",
			);
			let migratedIds: string[] = [];
			const originalRewrite = FileSessionStorage.prototype.writeLinesSync;
			const rewrite = vi
				.spyOn(FileSessionStorage.prototype, "writeLinesSync")
				.mockImplementationOnce(function (this: FileSessionStorage, file, lines, options) {
					const prepare = options!.prepareLines!;
					originalRewrite.call(this, file, lines, {
						...options,
						prepareLines: () => {
							const prepared = prepare();
							migratedIds = parseSessionContent(prepared.lines.join(""))
								.entries.slice(1)
								.map(entry => entry.id);
							throw new Error("initial migration write failed");
						},
					});
				});
			const appender = await SessionManager.openAppendOnly(transcript, { initialCwd: dir });
			try {
				await expect(appender.flush()).rejects.toThrow("initial migration write failed");
				appender.appendMessage(assistantMessage("first live", 3) as Message);
				await appender.flush();
				appender.appendMessage(assistantMessage("second live", 4) as Message);
				await appender.flush();
				const entries = parseSessionContent(await Bun.file(transcript).text()).entries;
				expect(entries.slice(1, 3).map(entry => entry.id)).toEqual(migratedIds);
				expect(entries.map(entry => entry.id).filter(id => migratedIds.includes(id))).toHaveLength(2);
				for (let index = 2; index < entries.length; index++) {
					expect(entries[index]).toMatchObject({ parentId: entries[index - 1].id });
				}
				await normalizedTranscript(transcript);
			} finally {
				await appender.close();
				rewrite.mockRestore();
			}
		});
	});

	it("matches normal disk output without sanitizing live Copilot payloads during recovery", async () => {
		await withTempDir(async dir => {
			const baseline = path.join(dir, "baseline.jsonl");
			const actual = path.join(dir, ADVISOR_TRANSCRIPT_FILENAME);
			const seed = await SessionManager.open(baseline, undefined, undefined, {
				initialCwd: dir,
				suppressBreadcrumb: true,
			});
			seed.appendMessage(copilotMessage("old connection"));
			await seed.close();
			await Bun.write(actual, Bun.file(baseline));
			const failNext = new Set<string>();
			const originalOpenWriter = FileSessionStorage.prototype.openWriter;
			const writerSpy = vi
				.spyOn(FileSessionStorage.prototype, "openWriter")
				.mockImplementation(function (this: FileSessionStorage, file, options) {
					const writer = originalOpenWriter.call(this, file, options);
					const append = writer.appendSync!.bind(writer);
					vi.spyOn(writer, "appendSync").mockImplementation(line => {
						if (failNext.delete(file)) throw new Error("transient append failure");
						append(line);
					});
					return writer;
				});
			const normal = await SessionManager.open(baseline, undefined, undefined, {
				initialCwd: dir,
				suppressBreadcrumb: true,
			});
			const appender = await SessionManager.openAppendOnly(actual, { initialCwd: dir });
			try {
				// Resume-only sanitization is committed before any live message.
				expect(await Bun.file(actual).text()).not.toContain("signature-old connection");
				for (const manager of [normal, appender]) manager.appendMessage(copilotMessage("live before failure"));
				failNext.add(baseline);
				failNext.add(actual);
				for (const manager of [normal, appender]) {
					manager.appendMessage(copilotMessage("live failed append"));
					await expect(manager.flush()).rejects.toThrow("transient append failure");
					manager.appendMessage(copilotMessage("live recovery"));
					await manager.flush();
				}
				expect(await normalizedTranscript(actual)).toEqual(await normalizedTranscript(baseline));
				const entries = parseSessionContent(await Bun.file(actual).text()).entries.slice(2);
				for (const entry of entries) {
					expect(entry).toMatchObject({
						type: "message",
						message: {
							providerPayload: { type: "openaiResponsesHistory" },
							content: [expect.objectContaining({ thinkingSignature: expect.any(String) }), expect.any(Object)],
						},
					});
				}
			} finally {
				await normal.close();
				await appender.close();
				writerSpy.mockRestore();
			}
		});
	});

	it.each(["deleted", "zeroed", "headerless"] as const)(
		"recreates a %s live journal and continues recording",
		async mutation => {
			await withTempDir(async dir => {
				const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
				const recorder = new AdvisorTranscriptRecorder(
					() => path.join(dir, "sess.jsonl"),
					() => dir,
				);
				const warn = vi.spyOn(dirs.logger, "warn");
				try {
					recorder.record(assistantMessage("old history", 1));
					await recorder.flush();
					const oldHeader = parseSessionContent(await Bun.file(transcript).text()).entries[0];
					if (mutation === "deleted") await fs.unlink(transcript);
					else if (mutation === "zeroed") await fs.truncate(transcript, 0);
					else await Bun.write(transcript, "{headerless journal\n");
					recorder.record(assistantMessage("first new journal turn", 2));
					await recorder.flush();
					recorder.record(assistantMessage("later new journal turn", 3));
					await recorder.flush();
					const entries = parseSessionContent(await Bun.file(transcript).text()).entries;
					expect(entries[0].id).not.toBe(oldHeader.id);
					expect(entries[1]).toMatchObject({ parentId: null });
					expect(entries[2]).toMatchObject({ parentId: entries[1].id });
					expect((await readMessageEntries(transcript)).map(entry => entry.message?.usage?.input)).toEqual([2, 3]);
					expect(warn.mock.calls.filter(call => call[0].includes("Advisor transcript was deleted"))).toHaveLength(
						1,
					);
				} finally {
					await recorder.close();
					warn.mockRestore();
				}
			});
		},
	);

	it.each(["different-header", "same-size-inode", "same-size-in-place"] as const)(
		"leaves externally replaced journal bytes unchanged: %s",
		async mutation => {
			await withTempDir(async dir => {
				const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
				const recorder = new AdvisorTranscriptRecorder(
					() => path.join(dir, "sess.jsonl"),
					() => dir,
				);
				try {
					recorder.record(assistantMessage("original turn", 1));
					await recorder.flush();
					const previous = await Bun.file(transcript).text();
					const header = parseSessionContent(previous).entries[0];
					if (mutation === "same-size-inode") {
						const replacement = path.join(dir, "replacement.jsonl");
						await Bun.write(replacement, previous.replace(header.id, "f".repeat(header.id.length)));
						await fs.rename(replacement, transcript);
					} else if (mutation === "same-size-in-place") {
						await Bun.write(transcript, previous.replace(header.id, "f".repeat(header.id.length)));
					} else {
						await Bun.write(
							transcript,
							JSON.stringify({
								...header,
								...(mutation === "different-header" ? { id: "foreign-advisor" } : {}),
							}) + "\n",
						);
					}
					const replacementBytes = await Bun.file(transcript).text();
					recorder.record(assistantMessage("must not land", 2));
					await expect(recorder.flush()).rejects.toThrow("Cannot append advisor transcript");
					recorder.record(assistantMessage("still must not land", 3));
					await expect(recorder.flush()).rejects.toThrow("Cannot append advisor transcript");
					expect(await Bun.file(transcript).text()).toBe(replacementBytes);
					expect((await fs.readdir(path.dirname(transcript))).filter(name => name.endsWith(".jsonl"))).toEqual([
						ADVISOR_TRANSCRIPT_FILENAME,
					]);
				} finally {
					await recorder.close();
				}
			});
		},
	);

	it("accepts another manager's same-process appends without per-message header reads", async () => {
		await withTempDir(async dir => {
			const transcript = path.join(dir, ADVISOR_TRANSCRIPT_FILENAME);
			const seed = await SessionManager.open(transcript, undefined, undefined, {
				initialCwd: dir,
				suppressBreadcrumb: true,
			});
			seed.appendMessage(assistantMessage("seed", 1) as Message);
			await seed.close();
			const appender = await SessionManager.openAppendOnly(transcript, { initialCwd: dir });
			const peer = await SessionManager.open(transcript, undefined, undefined, {
				initialCwd: dir,
				suppressBreadcrumb: true,
			});
			const readHeader = vi.spyOn(sessionStorage, "readSessionHeaderIdSync");
			try {
				appender.appendMessage(assistantMessage("own first", 2) as Message);
				const afterFirst = readHeader.mock.calls.length;
				appender.appendMessage(assistantMessage("own hot append", 3) as Message);
				await appender.flush();
				expect(readHeader.mock.calls.length).toBe(afterFirst);
				peer.appendMessage(assistantMessage("same-process peer", 4) as Message);
				appender.appendMessage(assistantMessage("after peer", 5) as Message);
				await appender.flush();
				expect(readHeader.mock.calls.length).toBeGreaterThan(afterFirst);
				const afterPeer = readHeader.mock.calls.length;
				appender.appendMessage(assistantMessage("hot again", 6) as Message);
				await appender.flush();
				expect(readHeader.mock.calls.length).toBe(afterPeer);
				const entries = parseSessionContent(await Bun.file(transcript).text()).entries;
				expect((await readMessageEntries(transcript)).map(entry => entry.message?.usage?.input)).toEqual([
					1, 2, 3, 4, 5, 6,
				]);
				expect(entries[5]).toMatchObject({ parentId: entries[3].id });
				await normalizedTranscript(transcript);
			} finally {
				await peer.close();
				await appender.close();
				readHeader.mockRestore();
			}
		});
	});

	it.each(["identical-bytes-rewrite", "sanitize-shrink", "partial-rollback"] as const)(
		"continues after a same-process peer's %s",
		async mutation => {
			await withTempDir(async dir => {
				const sessionFile = path.join(dir, "sess.jsonl");
				const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
				const seed = await SessionManager.open(transcript, undefined, undefined, {
					initialCwd: dir,
					suppressBreadcrumb: true,
				});
				seed.appendMessage(assistantMessage("seed", 1, 0.1) as Message);
				await seed.close();
				const appender = await SessionManager.openAppendOnly(transcript, { initialCwd: dir });
				appender.appendMessage(
					mutation === "sanitize-shrink"
						? copilotMessage("large live payload ".repeat(1_000), 2, 0.2)
						: (assistantMessage("live", 2, 0.2) as Message),
				);
				await appender.flush();
				const before = await Bun.file(transcript).text();
				const peer = await SessionManager.open(transcript, undefined, undefined, {
					initialCwd: dir,
					suppressBreadcrumb: true,
				});
				const originalOpenWriter = FileSessionStorage.prototype.openWriter;
				const openWriter = vi.spyOn(FileSessionStorage.prototype, "openWriter");
				const errors = vi.spyOn(dirs.logger, "error");
				try {
					if (mutation === "identical-bytes-rewrite") {
						await peer.rewriteEntries();
						expect(await Bun.file(transcript).text()).toBe(before);
					} else if (mutation === "sanitize-shrink") {
						peer.appendMessage(assistantMessage("peer after sanitize", 3, 0.3) as Message);
						await peer.flush();
						expect((await fs.stat(transcript)).size).toBeLessThan(Buffer.byteLength(before));
					} else {
						openWriter.mockImplementationOnce(function (this: FileSessionStorage, file, options) {
							const writer = originalOpenWriter.call(this, file, options);
							vi.spyOn(writer, "appendSync").mockImplementationOnce(() => {
								const originalSize = fsSync.statSync(file).size;
								fsSync.appendFileSync(file, "{partial peer append");
								fsSync.truncateSync(file, originalSize);
								throw new Error("peer rollback");
							});
							return writer;
						});
						peer.appendMessage(assistantMessage("failed peer", 3, 0.3) as Message);
						await expect(peer.flush()).rejects.toThrow("peer rollback");
						expect(await Bun.file(transcript).text()).toBe(before);
					}
					appender.appendMessage(
						assistantMessage(
							"appender alone after peer",
							mutation === "sanitize-shrink" ? 4 : 3,
							mutation === "sanitize-shrink" ? 0.4 : 0.3,
						) as Message,
					);
					await appender.flush();
					appender.appendMessage(assistantMessage("continues", 5, 0.5) as Message);
					await appender.flush();
					expect((await readMessageEntries(transcript)).map(entry => entry.message?.usage?.input)).toEqual(
						mutation === "sanitize-shrink" ? [1, 2, 3, 4, 5] : [1, 2, 3, 5],
					);
					await normalizedTranscript(transcript);
					expect((await loadAdvisorTranscriptCosts(sessionFile)).get("")).toBeCloseTo(
						mutation === "sanitize-shrink" ? 1.5 : 1.1,
						8,
					);
					if (mutation !== "partial-rollback") {
						expect(errors.mock.calls.filter(call => call[0].includes("Session persistence error"))).toHaveLength(
							0,
						);
					}
				} finally {
					openWriter.mockRestore();
					errors.mockRestore();
					await peer.close().catch(() => undefined);
					await appender.close();
				}
			});
		},
	);

	it("continues from surviving history after a same-header truncation", async () => {
		await withTempDir(async dir => {
			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const recorder = new AdvisorTranscriptRecorder(
				() => path.join(dir, "sess.jsonl"),
				() => dir,
			);
			try {
				recorder.record(assistantMessage("old history", 1));
				await recorder.flush();
				const header = parseSessionContent(await Bun.file(transcript).text()).entries[0];
				await Bun.write(transcript, `${JSON.stringify(header)}\n`);
				recorder.record(assistantMessage("after truncation", 2));
				await recorder.flush();
				recorder.record(assistantMessage("continues", 3));
				await recorder.flush();
				expect((await readMessageEntries(transcript)).map(entry => entry.message?.usage?.input)).toEqual([2, 3]);
				const entries = parseSessionContent(await Bun.file(transcript).text()).entries;
				expect(entries[0].id).toBe(header.id);
				expect(entries[1]).toMatchObject({ parentId: null });
				await normalizedTranscript(transcript);
			} finally {
				await recorder.close();
			}
		});
	});

	it.each(["in-place-before-gate", "rename-before-gate", "rename-during-retry"] as const)(
		"does not cross into a foreign header at the writer boundary: %s",
		async mutation => {
			await withTempDir(async dir => {
				const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
				const replacement = path.join(dir, "foreign.jsonl");
				const recorder = new AdvisorTranscriptRecorder(
					() => path.join(dir, "sess.jsonl"),
					() => dir,
				);
				const seed = await SessionManager.open(transcript, undefined, undefined, {
					initialCwd: dir,
					suppressBreadcrumb: true,
				});
				seed.appendMessage(assistantMessage("claim original", 1) as Message);
				await seed.close();
				const before = await Bun.file(transcript).text();
				const header = parseSessionContent(before).entries[0];
				const foreign = before.replace(header.id, "f".repeat(header.id.length));
				const originalOpenWriter = FileSessionStorage.prototype.openWriter;
				const originalWriteSync = fsSync.writeSync;
				const writeSync = vi.spyOn(fsSync, "writeSync");
				const openWriter = vi.spyOn(FileSessionStorage.prototype, "openWriter");
				let replaced = false;
				const replace = () => {
					replaced = true;
					if (mutation === "in-place-before-gate") fsSync.writeFileSync(transcript, foreign);
					else {
						fsSync.writeFileSync(replacement, foreign);
						fsSync.renameSync(replacement, transcript);
					}
				};
				if (mutation === "rename-during-retry") {
					writeSync.mockImplementation((...args: unknown[]) => {
						const written = Reflect.apply(originalWriteSync, fsSync, args) as number;
						if (!replaced && Buffer.isBuffer(args[1]) && args[1].includes("boundary turn")) replace();
						return written;
					});
				} else {
					openWriter.mockImplementationOnce(function (this: FileSessionStorage, file, options) {
						const writer = originalOpenWriter.call(this, file, options);
						const append = writer.appendSync!.bind(writer);
						vi.spyOn(writer, "appendSync").mockImplementationOnce(line => {
							replace();
							append(line);
						});
						return writer;
					});
				}
				try {
					recorder.record(assistantMessage("boundary turn", 2));
					await expect(recorder.flush()).rejects.toThrow("Cannot append advisor transcript");
					recorder.record(assistantMessage("must never enter foreign", 3));
					await expect(recorder.flush()).rejects.toThrow("Cannot append advisor transcript");
					expect(replaced).toBe(true);
					expect(await Bun.file(transcript).text()).toBe(foreign);
				} finally {
					writeSync.mockRestore();
					openWriter.mockRestore();
					await recorder.close();
				}
			});
		},
	);

	it("revalidates a cold rewrite under the publish gate instead of replacing a foreign journal", async () => {
		await withTempDir(async dir => {
			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const originalOpenWriter = FileSessionStorage.prototype.openWriter;
			let failNext = false;
			const openWriter = vi
				.spyOn(FileSessionStorage.prototype, "openWriter")
				.mockImplementation(function (this: FileSessionStorage, file, options) {
					const writer = originalOpenWriter.call(this, file, options);
					const append = writer.appendSync!.bind(writer);
					vi.spyOn(writer, "appendSync").mockImplementation(line => {
						if (failNext) {
							failNext = false;
							throw new Error("force cold rewrite");
						}
						append(line);
					});
					return writer;
				});
			const recorder = new AdvisorTranscriptRecorder(
				() => path.join(dir, "sess.jsonl"),
				() => dir,
			);
			const originalRewrite = FileSessionStorage.prototype.writeLinesSync;
			const rewrite = vi.spyOn(FileSessionStorage.prototype, "writeLinesSync");
			try {
				recorder.record(assistantMessage("original", 1));
				await recorder.flush();
				const before = await Bun.file(transcript).text();
				const header = parseSessionContent(before).entries[0];
				const foreign = before.replace(header.id, "f".repeat(header.id.length));
				failNext = true;
				recorder.record(assistantMessage("pending failed append", 2));
				await expect(recorder.flush()).rejects.toThrow("force cold rewrite");
				rewrite.mockImplementationOnce(function (this: FileSessionStorage, file, lines, options) {
					fsSync.writeFileSync(file, foreign);
					originalRewrite.call(this, file, lines, options);
				});
				recorder.record(assistantMessage("cold rewrite", 3));
				await expect(recorder.flush()).rejects.toThrow("Cannot append advisor transcript");
				recorder.record(assistantMessage("still stopped", 4));
				await expect(recorder.flush()).rejects.toThrow("Cannot append advisor transcript");
				expect(await Bun.file(transcript).text()).toBe(foreign);
			} finally {
				openWriter.mockRestore();
				rewrite.mockRestore();
				await recorder.close();
			}
		});
	});

	it("does not retain large messages after recording is permanently stopped", async () => {
		await withTempDir(async dir => {
			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const recorder = new AdvisorTranscriptRecorder(
				() => path.join(dir, "sess.jsonl"),
				() => dir,
			);
			const warn = vi.spyOn(dirs.logger, "warn");
			try {
				recorder.record(assistantMessage("original", 1));
				await recorder.flush();
				const before = await Bun.file(transcript).text();
				const header = parseSessionContent(before).entries[0];
				const foreign = before.replace(header.id, "f".repeat(header.id.length));
				await Bun.write(transcript, foreign);
				const refs = recordCollectibleRejectedMessages(recorder);
				await expect(recorder.flush()).rejects.toThrow("Cannot append advisor transcript");
				for (const ref of refs) expect(await collected(ref, 15_000)).toBe(true);
				expect(await Bun.file(transcript).text()).toBe(foreign);
				expect(warn.mock.calls.filter(call => call[0].includes("permanently stopped"))).toHaveLength(1);
			} finally {
				await recorder.close();
				warn.mockRestore();
			}
		});
	}, 60_000);

	it("records when TMPDIR is not writable", async () => {
		await withTempDir(async dir => {
			const blocked = path.join(dir, "blocked-tmp");
			await fs.mkdir(blocked, { mode: 0o555 });
			const sessionFile = path.join(dir, "sess.jsonl");
			const script = path.join(dir, "tmpdir-probe.ts");
			await Bun.write(
				script,
				[
					`import { AdvisorTranscriptRecorder } from ${JSON.stringify(path.resolve(import.meta.dir, "../../src/advisor/transcript-recorder.ts"))};`,
					`const recorder = new AdvisorTranscriptRecorder(() => ${JSON.stringify(sessionFile)}, () => ${JSON.stringify(dir)});`,
					`recorder.record(${JSON.stringify(assistantMessage("no tmp dependency", 7))});`,
					"await recorder.flush();",
					"await recorder.close();",
				].join("\n"),
			);
			try {
				const child = Bun.spawn([process.execPath, script], {
					env: { ...process.env, TMPDIR: blocked },
					stdout: "pipe",
					stderr: "pipe",
				});
				const exitCode = await child.exited;
				const stderr = await new Response(child.stderr).text();
				expect({ exitCode, stderr }).toMatchObject({ exitCode: 0 });
				expect(
					(await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME)))[0].message?.usage?.input,
				).toBe(7);
			} finally {
				await fs.chmod(blocked, 0o755);
			}
		});
	});

	it("persists assistant turns with usage to <session>/__advisor.jsonl", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(assistantMessage("reviewing", 42));
			await recorder.close();

			const messages = await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME));
			expect(messages).toHaveLength(1);
			expect(messages[0].message?.role).toBe("assistant");
			expect(messages[0].message?.model).toBe("test-advisor-model");
			expect(messages[0].message?.usage?.input).toBe(42);
			// Stats keys on a non-empty entry id; SessionManager must assign one.
			expect(typeof messages[0].id).toBe("string");
			expect(String(messages[0].id).length).toBeGreaterThan(0);
		});
	});

	it("marks advisor user deltas synthetic and agent-attributed", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(userMessage("### Session update"));
			await recorder.close();

			const messages = await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME));
			expect(messages).toHaveLength(1);
			expect(messages[0].message?.role).toBe("user");
			expect(messages[0].message?.synthetic).toBe(true);
			expect(messages[0].message?.attribution).toBe("agent");
		});
	});

	it("skips non-conversational message kinds", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(developerMessage("noise"));
			recorder.record(assistantMessage("kept", 1));
			await recorder.close();

			const messages = await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME));
			expect(messages.map(m => m.message?.role)).toEqual(["assistant"]);
		});
	});

	it("routes later turns to the new session file after a switch", async () => {
		await withTempDir(async dir => {
			let sessionFile = path.join(dir, "first.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(assistantMessage("before switch", 1));
			sessionFile = path.join(dir, "second.jsonl");
			recorder.record(assistantMessage("after switch", 2));
			await recorder.close();

			const first = await readMessageEntries(path.join(dir, "first", ADVISOR_TRANSCRIPT_FILENAME));
			const second = await readMessageEntries(path.join(dir, "second", ADVISOR_TRANSCRIPT_FILENAME));
			expect(first).toHaveLength(1);
			expect(first[0].message?.usage?.input).toBe(1);
			expect(second).toHaveLength(1);
			expect(second[0].message?.usage?.input).toBe(2);
		});
	});

	it("skips a retried batch but keeps every billed assistant turn", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			// A failing advisor re-sends the identical batch each attempt; the turn
			// only commits once it finally succeeds (issue #9553).
			for (let attempt = 0; attempt < 5; attempt++) {
				recorder.beginTurn();
				recorder.record({ ...userMessage("### Session update"), timestamp: attempt + 1 } as AgentMessage);
				recorder.record(assistantMessage(`attempt ${attempt}`, 1, 0.1));
			}
			recorder.commitTurn();
			await recorder.close();

			const messages = await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME));
			expect(messages.filter(m => m.message?.role === "user")).toHaveLength(1);
			expect(messages.filter(m => m.message?.role === "assistant")).toHaveLength(5);
			expect((await loadAdvisorTranscriptCosts(sessionFile)).get("")).toBeCloseTo(0.5, 8);
		});
	});

	it("keeps identical deltas that belong to distinct committed turns", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			// The user re-submits the same prompt across three separate turns: each
			// renders an identical "Session update" yet is genuinely new content.
			for (let turn = 0; turn < 3; turn++) {
				recorder.beginTurn();
				recorder.record(userMessage("### Session update"));
				recorder.record(assistantMessage(`review ${turn}`, 1, 0.1));
				recorder.commitTurn();
			}
			await recorder.close();

			const messages = await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME));
			expect(messages.filter(m => m.message?.role === "user")).toHaveLength(3);
		});
	});

	it("keeps a repeated delta after the prior batch is abandoned", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.beginTurn();
			recorder.record(userMessage("### Session update"));
			recorder.abandonTurn();
			recorder.beginTurn();
			recorder.record(userMessage("### Session update"));
			recorder.commitTurn();
			await recorder.close();

			const messages = await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME));
			expect(messages.filter(m => m.message?.role === "user")).toHaveLength(2);
		});
	});

	it("holds post-snapshot records behind a byte boundary", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(assistantMessage("before", 1, 0.25));
			const gate = Promise.withResolvers<void>();
			const ready = recorder.blockWritesUntil(gate.promise);
			recorder.record(assistantMessage("after", 1, 0.5));
			await ready;

			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const beforeRelease = await readMessageEntries(transcript);
			expect(beforeRelease.filter(m => m.message?.role === "assistant")).toHaveLength(1);

			gate.resolve();
			await recorder.close();
			const afterRelease = await readMessageEntries(transcript);
			expect(afterRelease.filter(m => m.message?.role === "assistant")).toHaveLength(2);
		});
	});

	it("keeps identical deltas delivered within one turn", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			// Two tool runs with byte-identical output render two identical chunks in
			// one delivery; both must persist (they are distinct positions, not a replay).
			recorder.beginTurn();
			recorder.record(userMessage("### Session update"));
			recorder.record(userMessage("### Session update"));
			recorder.record(assistantMessage("review", 1, 0.1));
			recorder.commitTurn();
			await recorder.close();

			const messages = await readMessageEntries(path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME));
			expect(messages.filter(m => m.message?.role === "user")).toHaveLength(2);
		});
	});

	it("loads cumulative costs by advisor slug", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const primary = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			const security = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
				advisorTranscriptFilename("security"),
			);
			primary.record(assistantMessage("primary", 1, 0.25));
			security.record(assistantMessage("first", 1, 0.25));
			security.record(assistantMessage("second", 1, 0.5));
			await Promise.all([primary.close(), security.close()]);

			expect(Object.fromEntries(await loadAdvisorTranscriptCosts(sessionFile))).toEqual({
				"": 0.25,
				security: 0.75,
			});
		});
	});

	it("excludes providers that only produced zero-cost turns from subscription attribution", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(assistantMessage("paid", 1, 0.25, "openai"));
			recorder.record(assistantMessage("failed subscription fallback", 1, 0, "anthropic"));
			await recorder.close();

			const providersBySlug = new Map<string, Set<string>>();
			await loadAdvisorTranscriptCosts(sessionFile, { providersBySlug });
			expect([...(providersBySlug.get("") ?? [])]).toEqual(["openai"]);
		});
	});

	it("yields before snapshotting transcript metadata", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(assistantMessage("persisted", 1, 0.25));
			await recorder.close();

			let snapshotTaken = false;
			const costs = loadAdvisorTranscriptCosts(sessionFile, {
				onSnapshot: () => {
					snapshotTaken = true;
				},
			});
			expect(snapshotTaken).toBe(false);
			expect((await costs).get("")).toBeCloseTo(0.25, 8);
			expect(snapshotTaken).toBe(true);
		});
	});

	it("excludes transcript entries appended after the cost snapshot", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(assistantMessage("persisted before snapshot", 1, 0.25));
			await recorder.close();

			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const appended = Promise.withResolvers<void>();
			const costs = loadAdvisorTranscriptCosts(sessionFile, {
				onSnapshot: () => {
					const entry = JSON.stringify({
						type: "message",
						message: assistantMessage("billed after snapshot", 1, 0.5),
					});
					void fs.appendFile(transcript, `${entry}\n`).then(appended.resolve, appended.reject);
				},
			});
			await appended.promise;

			expect((await costs).get("")).toBeCloseTo(0.25, 8);
			expect((await loadAdvisorTranscriptCosts(sessionFile)).get("")).toBeCloseTo(0.75, 8);
		});
	});

	it("keeps valid costs when persisted entries are malformed", async () => {
		await withTempDir(async dir => {
			const sessionFile = path.join(dir, "sess.jsonl");
			const recorder = new AdvisorTranscriptRecorder(
				() => sessionFile,
				() => dir,
			);
			recorder.record(assistantMessage("valid", 1, 0.25));
			await recorder.close();
			const transcript = path.join(dir, "sess", ADVISOR_TRANSCRIPT_FILENAME);
			const lines = (await fs.readFile(transcript, "utf8")).trimEnd().split("\n");
			lines.splice(
				-1,
				0,
				JSON.stringify({ type: "message", message: { role: "assistant" } }),
				"{ this is not valid json",
				JSON.stringify({ type: "message" }),
				"null",
			);
			await fs.writeFile(transcript, `${lines.join("\n")}\n`);

			expect((await loadAdvisorTranscriptCosts(sessionFile)).get("")).toBe(0.25);
		});
	});
});
