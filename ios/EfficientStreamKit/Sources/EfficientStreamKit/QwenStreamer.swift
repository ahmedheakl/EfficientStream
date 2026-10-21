// The streaming Qwen3.5-0.8B's stream (proactive/stream_qwen.py `stream`): one update per second of video, its frames
// (8 at the app's 8 fps) as 2-frame temporal blocks with time labels, answered </Silence>, </Standby> or </Response> +
// text (the proactive mode's watch prompt) or </Silence>, </Observe> + a detail or </Summary> + a summary (the PHCM
// caption stream, `prompt` nil).
//
// The streaming KV cache: the runner re-renders the whole window every second; here the language model's caches are
// kept and each second appends its turn (exactly the tokens the re-rendered prompt adds), so an update prefills only
// the new second (~420 tokens at 8 fps). Turns interleave frames and the model's own answers in one cache, positions
// keep counting up (M-RoPE), and the 18 gated-delta layers carry their state over everything seen. What leaves the
// context is dropped from the 6 attention layers' caches only, without a recompute (`Context`):
//   .evict(8)     the app's choice: every 8 seconds the turns older than the last 32 seconds go (the first turn, with
//                 the instruction, stays)
//   .compress(n)  frames of the newest seconds only: the second before the newest keeps its n most attended frames
//                 (by the newest decision's attention), the one before that loses the rest; all text stays
// A question forks the latest update's caches (`snapshot`, copy-on-write) and prefills only itself; the stream goes on
// unchanged.

import Foundation
import MLX

public struct QwenStreamUpdate: Codable, Sendable {
    public var second: Int
    public var answer: String
    public var frames: Int
    /// Resize + patches of the second's frames, the vision tower on its two blocks (evaluated).
    public var prep = 0.0
    public var vision = 0.0
    /// Tokens prefilled for this update (the new turn, or the whole window when recomputed) and the time to the
    /// answer's first token.
    public var prefillTokens = 0
    public var prefill = 0.0
    public var decodeTokens = 0
    public var decode = 0.0
    public var total = 0.0
    /// Tokens in the context after this update, and whether the window was recomputed.
    public var contextTokens = 0
    public var recomputed = false
    /// `.compress`: choosing and dropping frame blocks after the answer (in `total`).
    public var compress = 0.0
    /// A response replaced by </Silence> as a repeat (`noRepeat`).
    public var repeated: String?
}

public final class QwenStreamer: @unchecked Sendable {
    public enum Context: Sendable, RawRepresentable {
        /// The runner's prompt: the last `window` seconds, recomputed every second once full.
        case slidingWindow
        /// Appended to forever.
        case growing
        /// Appended to until `extra` seconds past the window, then recomputed to the last `window` seconds (one
        /// recompute every `extra` seconds; the live app's choice).
        case windowEvery(Int)
        /// Appended to; once `extra` seconds past the window, the oldest seconds' tokens are dropped from the
        /// attention layers' caches (no recompute), keeping the first turn (the instruction) and the last `window`
        /// seconds. The linear-attention layers' states carry on over everything seen.
        case evict(Int)
        /// Appended to; frames are kept for the newest seconds only: after each update the second before the newest
        /// keeps its `n` frames (n / 2 two-frame blocks) the newest decision attended to most (its share of the
        /// attention, over the attention layers) and the second before that loses the rest. All text (time labels of
        /// the turns, answers, captions) stays; each dropped block takes its time label and vision tokens with it.
        /// "compress" keeps 2 frames, "compress<n>" n.
        case compress(Int)

        public init?(rawValue: String) {
            switch rawValue {
            case "slidingWindow": self = .slidingWindow
            case "growing": self = .growing
            case "compress": self = .compress(2)
            default:
                if rawValue.hasPrefix("compress"), let n = Int(rawValue.dropFirst(8)), n >= 2, n % 2 == 0 {
                    self = .compress(n)
                    return
                }
                if rawValue.hasPrefix("evict"), let n = Int(rawValue.dropFirst(5)) {
                    self = .evict(n)
                    return
                }
                guard rawValue.hasPrefix("windowEvery"), let n = Int(rawValue.dropFirst(11)) else { return nil }
                self = .windowEvery(n)
            }
        }

        public var rawValue: String {
            switch self {
            case .slidingWindow: return "slidingWindow"
            case .growing: return "growing"
            case .windowEvery(let n): return "windowEvery\(n)"
            case .evict(let n): return "evict\(n)"
            case .compress(let n): return n == 2 ? "compress" : "compress\(n)"
            }
        }
    }

    public let qwen: StreamQwen
    /// The watch prompt (proactive mode); nil = the caption memory's stream (PHCM: CAPTION_SYSTEM, the instruction
    /// around the first second's frames, answers </Silence>, </Observe> + a detail or </Summary> + a summary).
    public let prompt: String?
    public let window: Int
    public let context: Context
    public let size: QwenFrameSize
    public let maxNewTokens: Int
    /// The runner's `no_repeat`: a </Response> that repeats one of the last `noRepeat` responses word for word is
    /// replaced by </Silence>, also in the context the model sees next (0 = off).
    public let noRepeat: Int
    private var said: [String] = []
    private var saidTokens: [[Int]] = []

    private struct Turn {
        let second: Int
        let visual: String
        let blocks: [QwenFrame]
        var answer: String?
        /// Where the turn's tokens start in the caches (from its <|im_start|>; `.evict` only).
        var start = 0
        /// `.compress`: where each of its blocks still in the caches is (time label to <|vision_end|>), oldest first.
        var spans: [Range<Int>] = []
    }

    /// What a question forks: the caches after the latest update (its answer's tokens in, the <|im_end|> where
    /// decoding stopped not), the next position, and the seconds of video the context holds.
    public struct Snapshot: @unchecked Sendable {
        let caches: [LayerCache]
        let position: Int
        let pending: String
        public let contextTokens: Int
        /// The first and last second in the context, and the frames they hold.
        public let from: Int
        public let through: Int
        public let frames: Int
    }

    /// The caches no longer match the runner's re-tokenized answer: rebuilt at the next update (still a valid
    /// context for a question until then).
    private var stale = false
    /// `.evict` has dropped seconds after the first turn.
    private var evicted = false

    /// Held by an update; a snapshot waits for the one running (not for those queued behind it).
    private let busy = NSLock()

    /// The context after the latest update (nil before the first), waiting for an update that is running; the fork
    /// stays valid while the stream goes on.
    public func snapshot() -> Snapshot? {
        busy.withLock { snapshotLocked() }
    }

    private func snapshotLocked() -> Snapshot? {
        guard !caches.isEmpty, let first = turns.first, let last = turns.last else { return nil }
        let tp = qwen.model.config.vision.temporalPatchSize
        if case .compress = context {
            // the seconds that still have frames (the captions of the others are text in the context)
            return Snapshot(caches: Self.fork(caches), position: position, pending: pending,
                            contextTokens: contextTokens, from: turns.first { !$0.spans.isEmpty }?.second ?? last.second,
                            through: last.second, frames: turns.reduce(0) { $0 + $1.spans.count } * tp)
        }
        return Snapshot(caches: Self.fork(caches), position: position, pending: pending, contextTokens: contextTokens,
                        from: evicted && turns.count > 1 ? turns[1].second : first.second, through: last.second,
                        frames: turns.reduce(0) { $0 + $1.blocks.count } * tp)
    }

    private var turns: [Turn] = []
    private var caches: [LayerCache] = []
    private var position = 0
    /// A one-token answer that ends its turn (</Silence>, </Standby>): the step that would only predict its
    /// <|im_end|> is skipped, and the token is prefilled with the next turn instead.
    private var pending = ""
    private lazy var terminal: Set<Int> = Set(["</Silence>", "</Standby>"].compactMap {
        let t = qwen.tokenize($0)
        return t.count == 1 ? t[0] : nil
    })
    /// Called before each answer token (the caption stream waits here while the watch or a question has the GPU).
    public var beforeToken: (() -> Void)?
    private var contextTokens = 0

    public init(qwen: StreamQwen, prompt: String?, window: Int = 32, context: Context = .slidingWindow,
                size: QwenFrameSize? = nil, maxNewTokens: Int = 128, noRepeat: Int = 0)
    {
        self.noRepeat = noRepeat
        self.qwen = qwen
        self.prompt = prompt
        self.window = window
        self.context = context
        self.size = size ?? .resized(minPixels: qwen.spec.minPixels, maxPixels: qwen.spec.maxPixels)
        self.maxNewTokens = maxNewTokens
    }

    /// One second of video [a, a + 1): its frames (4 at 4 fps) and their times.
    public func update(second a: Int, frames: [RGBImage], times: [Double]) -> QwenStreamUpdate {
        update(QwenSecond(qwen: qwen, size: size, second: a, frames: frames, times: times))
    }

    /// One second already through the vision tower (shared by the streams that watch the same video).
    public func update(_ sec: QwenSecond) -> QwenStreamUpdate {
        busy.withLock { updateLocked(sec) }
    }

    private func updateLocked(_ sec: QwenSecond) -> QwenStreamUpdate {
        let a = sec.second
        var u = QwenStreamUpdate(second: a, answer: "", frames: sec.frames)
        let t0 = Date()
        u.prep = sec.prep
        u.vision = sec.vision
        let t2 = t0
        turns.append(Turn(second: a, visual: sec.visual, blocks: sec.blocks, answer: nil))

        // the prompt tokens to add: the new turn, or the whole window
        let text: String
        let newBlocks: [QwenFrame]
        let limit: Int? = {
            switch context {
            case .slidingWindow: return window
            case .growing: return nil
            case .windowEvery(let n): return window + n
            case .evict, .compress: return nil
            }
        }()
        if let limit, turns.count > limit {
            turns.removeFirst(turns.count - window)
            caches = qwen.model.makeCaches()
            position = 0
            contextTokens = 0
            text = render(turns)
            newBlocks = turns.flatMap(\.blocks)
            u.recomputed = true
        } else if turns.count == 1 || caches.isEmpty || stale {
            caches = qwen.model.makeCaches()
            position = 0
            contextTokens = 0
            text = render(turns)
            newBlocks = turns.flatMap(\.blocks)
        } else {
            // the previous answer's tokens are in the cache; its <|im_end|> (where decoding stopped) is not
            let close = pending + "<|im_end|>\n"
            turns[turns.count - 1].start = contextTokens + qwen.tokenize(close).count
            text = close + "<|im_start|>user\n\(sec.visual)<|im_end|>\n<|im_start|>assistant\n"
            newBlocks = sec.blocks
        }
        pending = ""
        stale = false
        let (ids, positions) = expandVideo(qwen.tokenize(text), blocks: newBlocks, start: position)
        let feats2 = concatenated(newBlocks.map(\.features), axis: 0)
        var compress = false
        if case .compress = context { compress = true }
        if compress {
            turns[turns.count - 1].spans = blockSpans(ids, base: contextTokens)
            for case .attention(let c) in caches { c.captureQuery = true }
        }
        let hidden = qwen.model.prefill(qwen.model.inputEmbeddings(tokenIDs: ids, features: feats2),
                                        positions: positions, caches: caches)
        var current = qwen.model.logits(hidden).reshaped(-1).argMax()
        eval(current)
        // the queries of the token that decides this second, per attention layer (`.compress`)
        var queries: [MLXArray] = []
        if compress {
            for case .attention(let c) in caches {
                c.captureQuery = false
                if let q = c.lastQuery { queries.append(q) }
                c.lastQuery = nil
            }
        }
        let t3 = Date()
        u.prefillTokens = ids.count
        u.prefill = t3.timeIntervalSince(t2)
        position = positions.next
        contextTokens += ids.count

        // the answer (the caches before it kept, to take a repeated response back)
        let before = noRepeat > 0 ? (caches: Self.fork(caches), position: position, tokens: contextTokens) : nil
        let eos = Set(qwen.spec.eosTokenIds)
        var generated: [Int] = []
        while true {
            let next = current.item(Int.self)
            if eos.contains(next) || generated.count >= maxNewTokens { break }
            generated.append(next)
            if generated.count == 1 && terminal.contains(next) {
                pending = qwen.decode([next])
                break
            }
            // a response that so far copies a recent one token for token (12 tokens on) is a repeat: stop decoding
            if let r = earlyRepeat(generated) {
                generated = r
                break
            }
            beforeToken?()
            current = qwen.model.logits(qwen.model.step(current, position: position, caches: caches)).reshaped(-1)
                .argMax()
            position += 1
            contextTokens += 1
            asyncEval(current)
        }
        var answer = StreamQwen.visible(qwen.decode(generated))
        if let before, answer.hasPrefix("</Response>") {
            if said.suffix(noRepeat).contains(answer) {
                u.repeated = answer
                answer = "</Silence>"
                caches = before.caches
                position = before.position
                contextTokens = before.tokens
                pending = answer
                generated = qwen.tokenize(answer)
            } else {
                said.append(answer)
                saidTokens.append(generated)
            }
        }
        turns[turns.count - 1].answer = answer
        u.decodeTokens = generated.count
        u.decode = Date().timeIntervalSince(t3)
        u.total = sec.prep + sec.vision + Date().timeIntervalSince(t0)
        u.answer = answer
        u.contextTokens = contextTokens
        // the runner re-tokenizes the answer text: if that differs from the generated ids, rebuild next second
        // (not when evicting: the caches hold what was generated)
        if case .evict(let extra) = context {
            if turns.count > window + extra { evictOldest() }
        } else if case .compress(let frames) = context {
            let t5 = Date()
            compressFrames(queries, frames: frames)
            u.compress = Date().timeIntervalSince(t5)
            u.total += u.compress
            u.contextTokens = contextTokens
        } else if qwen.tokenize(answer) != generated {
            stale = true
        }
        return u
    }

    /// The recent response `generated` is copying (its first 12+ tokens identical), if any: the full response.
    private func earlyRepeat(_ generated: [Int]) -> [Int]? {
        guard noRepeat > 0, generated.count >= 12 else { return nil }
        return saidTokens.suffix(noRepeat).first { $0.count > generated.count && Array($0.prefix(generated.count)) == generated }
    }

    /// `.evict`: drops the turns after the first up to the last `window` from the attention caches.
    private func evictOldest() {
        let keep = turns.count - window
        guard keep > 1 else { return }
        let (a, b) = (turns[1].start, turns[keep].start)
        for c in caches {
            if case .attention(let ac) = c { ac.evict(a, b) }
        }
        turns.removeSubrange(1 ..< keep)
        for i in 1 ..< turns.count { turns[i].start -= b - a }
        contextTokens -= b - a
        evicted = true
    }

    /// `.compress`, after an update: the second before the newest keeps the `frames` / 2 blocks the newest decision
    /// attended to most, the one before that drops the rest.
    private func compressFrames(_ queries: [MLXArray], frames: Int) {
        let keep = max(1, frames / qwen.model.config.vision.temporalPatchSize)
        var drop: [Range<Int>] = []
        let n = turns.count
        if n >= 3 { drop += turns[n - 3].spans }
        if n >= 2, turns[n - 2].spans.count > keep {
            let spans = turns[n - 2].spans
            var share = [Float](repeating: 0, count: spans.count)
            var layer = 0
            for case .attention(let c) in caches {
                guard layer < queries.count else { break }
                for (i, v) in c.attentionShare(queries[layer], spans: spans).enumerated() { share[i] += v }
                layer += 1
            }
            // most attended first
            let best = Array(share.indices.sorted { share[$0] > share[$1] }.prefix(keep))
            drop += spans.indices.filter { !best.contains($0) }.map { spans[$0] }
            lastKept = (turns[n - 2].second, best, share)
        }
        guard !drop.isEmpty else { return }
        // from the end, so the earlier ranges stay where they are
        for r in drop.sorted(by: { $0.lowerBound > $1.lowerBound }) {
            for case .attention(let c) in caches { c.evict(r.lowerBound, r.upperBound) }
            for i in turns.indices {
                turns[i].spans = turns[i].spans.compactMap { s in
                    if s == r { return nil }
                    return s.lowerBound >= r.upperBound ? (s.lowerBound - r.count) ..< (s.upperBound - r.count) : s
                }
                if turns[i].start >= r.upperBound { turns[i].start -= r.count }
            }
            contextTokens -= r.count
        }
    }

    /// `.compress`: the second, the blocks kept (most attended first) and the attention shares of the latest choice
    /// (for logs).
    public private(set) var lastKept: (second: Int, blocks: [Int], shares: [Float])?

    /// Where each video block of `ids` (a turn's tokens from cache index `base`) lies: its time label (the tokens
    /// after the previous <|vision_end|> or line break) through its <|vision_end|>.
    private func blockSpans(_ ids: [Int32], base: Int) -> [Range<Int>] {
        let (vs, ve) = (Int32(qwen.model.config.visionStartTokenID), Int32(qwen.model.config.visionEndTokenID))
        var spans: [Range<Int>] = []
        var i = 0
        while i < ids.count {
            if ids[i] == vs, let e = ids[i...].firstIndex(of: ve) {
                var a = i
                while a > 0, ids[a - 1] != ve, ids[a - 1] != vs, !qwen.decode([Int(ids[a - 1])]).contains("\n") {
                    a -= 1
                }
                spans.append((base + a) ..< (base + e + 1))
                i = e + 1
            } else {
                i += 1
            }
        }
        return spans
    }

    /// New cache objects holding the same (immutable) arrays: appending to one leaves the other as it was.
    static func fork(_ caches: [LayerCache]) -> [LayerCache] {
        caches.map {
            switch $0 {
            case .attention(let a):
                let c = AttentionCache()
                c.keys = a.keys
                c.values = a.values
                c.offset = a.offset
                return .attention(c)
            case .linear(let l):
                let c = LinearCache()
                c.conv = l.conv
                c.recurrent = l.recurrent
                return .linear(c)
            }
        }
    }

    /// A question on the stream's own context (the streaming KV cache): the latest snapshot's caches, forked, extended
    /// by the seconds the stream has not taken in yet (`recent` newer than the snapshot, at most `maxBehind`, as
    /// silent turns), then one user turn (`memory`, the captions of seconds no longer in the context, and the
    /// question), and the answer decoded greedily (never one of the stream's own action tokens). Only these tokens are
    /// prefilled; the stream itself is left as it was.
    public func ask(_ question: String, on snap: Snapshot, memory: String? = nil, recent: [QwenSecond] = [],
                    maxBehind: Int = 16, maxNewTokens: Int = 96, onText: ((String) -> Void)? = nil)
        -> (QwenAnswer, Snapshot, Int)
    {
        let t0 = Date()
        let behind = Array(recent.filter { $0.second > snap.through }.suffix(maxBehind))
        var text = snap.pending
        for sec in behind {
            text += "<|im_end|>\n<|im_start|>user\n\(sec.visual)<|im_end|>\n<|im_start|>assistant\n</Silence>"
        }
        let user = (memory.map { $0 + "\n" } ?? "") + question
        text += "<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
        let blocks = behind.flatMap(\.blocks)
        let (ids, positions) = expandVideo(qwen.tokenize(text), blocks: blocks, start: snap.position)
        let t1 = Date()
        let caches = Self.fork(snap.caches)
        let feats = blocks.isEmpty ? nil : concatenated(blocks.map(\.features), axis: 0)
        let hidden = qwen.model.prefill(qwen.model.inputEmbeddings(tokenIDs: ids, features: feats),
                                        positions: positions, caches: caches)
        var first = qwen.model.logits(hidden).reshaped(-1)
        let actions = ["</Silence>", "</Standby>", "</Observe>", "</Summary>"].compactMap {
            let t = qwen.tokenize($0)
            return t.count == 1 ? Int32(t[0]) : nil
        }
        if !actions.isEmpty { first[MLXArray(actions)] = MLXArray(Float(-1e9)).asType(first.dtype) }
        var current = first.argMax()
        eval(current)
        let tFirst = Date()
        var result = QwenAnswer(text: "", promptTokens: ids.count, visualTokens: feats?.dim(0) ?? 0,
                                tokenizeSeconds: t1.timeIntervalSince(t0), prefillSeconds: tFirst.timeIntervalSince(t1))
        let eos = Set(qwen.spec.eosTokenIds)
        var generated: [Int] = []
        var position = positions.next
        while true {
            let next = current.item(Int.self)
            if eos.contains(next) { break }
            generated.append(next)
            if generated.count >= maxNewTokens { break }
            let upcoming = qwen.model.logits(qwen.model.step(current, position: position, caches: caches))
                .reshaped(-1).argMax()
            position += 1
            asyncEval(upcoming)
            if let onText { onText(Self.answerText(qwen.decode(generated))) }
            current = upcoming
        }
        result.decodeSeconds = Date().timeIntervalSince(tFirst)
        result.generatedTokens = generated.count
        result.raw = StreamQwen.visible(qwen.decode(generated))
        result.text = Self.answerText(result.raw)
        onText?(result.text)
        return (result, snap, behind.count)
    }

    /// The answer without the stream's action tag (</Response>) the model may start it with.
    static func answerText(_ s: String) -> String {
        var t = StreamQwen.visible(s)
        for tag in ["</Response>", "Overall Summary:"] where t.hasPrefix(tag) {
            t = t.dropFirst(tag.count).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return t
    }

    /// The runner's chat for `turns` up to the generation prompt (the last turn's answer is not yet known).
    private func render(_ turns: [Turn]) -> String {
        let system = prompt == nil ? qwen.spec.captionSystem ?? "" : qwen.spec.streamSystem
        var s = "<|im_start|>system\n\(system)<|im_end|>\n"
        for (j, t) in turns.enumerated() {
            let first = prompt.map { $0 + "\n" + t.visual }
                ?? (qwen.spec.captionInstruction ?? "<VIDEO_CONTEXT>").replacingOccurrences(of: "<VIDEO_CONTEXT>",
                                                                                          with: t.visual)
            s += "<|im_start|>user\n" + (j == 0 ? first : t.visual) + "<|im_end|>\n"
            s += "<|im_start|>assistant\n"
            if let a = t.answer { s += a + "<|im_end|>\n" }
        }
        return s
    }

    /// Token ids with each video placeholder expanded to its block's tokens, and M-RoPE positions from `start`.
    private func expandVideo(_ ids: [Int], blocks: [QwenFrame], start: Int) -> ([Int32], MRopePositions) {
        let pad = qwen.model.config.videoTokenID
        var out: [Int32] = []
        var pos = MRopePositions(start: start)
        var bi = 0
        for id in ids {
            if id == pad {
                let b = blocks[bi]
                bi += 1
                out += [Int32](repeating: Int32(pad), count: b.tokens)
                pos.vision(rows: b.rows, cols: b.cols)
            } else {
                out.append(Int32(id))
                pos.text(1)
            }
        }
        precondition(bi == blocks.count, "\(bi) video placeholders for \(blocks.count) blocks")
        return (out, pos)
    }
}

/// The caption memory (PHCM, `proactive/stream_qwen.py` `memorize`): what the caption stream wrote, as the events a
/// question reads (`render_memory`): details (</Observe>) and summaries of finished segments (</Summary>, from the
/// segment's first detail to the second it was written).
public final class CaptionMemory: @unchecked Sendable {
    public struct Event: Sendable {
        public enum Kind: String, Sendable { case observe, summary }
        public let kind: Kind
        public let start: Double
        public let end: Double
        public let text: String
    }

    private let lock = NSLock()
    private var events: [Event] = []
    private var firstObserve: Double?
    private var fallback = 0.0

    public init() {}

    public var all: [Event] { lock.withLock { events } }
    public var count: Int { lock.withLock { events.count } }

    public func clear() {
        lock.withLock {
            events.removeAll()
            firstObserve = nil
            fallback = 0
        }
    }

    /// The `_parse` of one update's answer for second [a, b): the event it adds, if any.
    @discardableResult
    public func add(_ answer: String, second a: Double, end b: Double) -> Event? {
        let trimmed = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        func body(_ tag: String) -> String? {
            guard trimmed.hasPrefix(tag) else { return nil }
            let t = trimmed.dropFirst(tag.count).trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : t
        }
        return lock.withLock {
            if let t = body("</Observe>") {
                if firstObserve == nil { firstObserve = a }
                let e = Event(kind: .observe, start: a, end: b, text: t)
                events.append(e)
                return e
            }
            if let t = body("</Summary>") {
                let e = Event(kind: .summary, start: min(firstObserve ?? fallback, b), end: b, text: t)
                events.append(e)
                fallback = b
                firstObserve = nil
                return e
            }
            return nil
        }
    }

    /// `_memory_text`: the memory block of a question's prompt (summaries, then details, each "[a s-b s] text").
    public func text(preamble: String, postamble: String) -> String {
        Self.text(all, preamble: preamble, postamble: postamble)
    }

    public static func text(_ ev: [Event], preamble: String, postamble: String) -> String {
        func t(_ v: Double) -> String {
            v == v.rounded() ? String(Int(v)) : String(format: "%.2f", v).replacingOccurrences(
                of: "0+$", with: "", options: .regularExpression).replacingOccurrences(
                    of: "\\.$", with: "", options: .regularExpression)
        }
        let line = { (e: Event) in "[\(t(e.start))s-\(t(e.end))s] \(e.text)" }
        return ([preamble, "<SEMANTIC_SUMMARIES>"] + ev.filter { $0.kind == .summary }.map(line)
            + ["</SEMANTIC_SUMMARIES>", "<EVENT_CAPTIONS>"] + ev.filter { $0.kind == .observe }.map(line)
            + ["</EVENT_CAPTIONS>", postamble]).joined(separator: "\n")
    }
}

/// One second of video through the vision tower: its frames (fps of them) resized, in two-frame temporal blocks
/// (the last frame repeated to fill a block), with the time labels the model was trained on.
public struct QwenSecond: @unchecked Sendable {
    public let second: Int
    public let frames: Int
    let blocks: [QwenFrame]
    let visual: String
    /// Resize + patches, and the vision tower (evaluated).
    public let prep: Double
    public let vision: Double

    public init(qwen: StreamQwen, size: QwenFrameSize, second a: Int, frames: [RGBImage], times: [Double]) {
        let t0 = Date()
        let tp = qwen.model.config.vision.temporalPatchSize
        var fs = frames.map { qwen.sized($0, size) }
        var ts = times
        while fs.count % tp != 0 {
            fs.append(fs[fs.count - 1])
            ts.append(ts[ts.count - 1])
        }
        var patches: [MLXArray] = []
        var grid = (0, 0)
        for i in stride(from: 0, to: fs.count, by: tp) {
            let (x, gh, gw) = qwen.patches(Array(fs[i ..< i + tp]))
            patches.append(x)
            grid = (gh, gw)
        }
        let batch = stacked(patches, axis: 0)
        eval(batch)
        let t1 = Date()
        let feats = qwen.model.vision(batch, gridH: grid.0, gridW: grid.1)
        eval(feats)
        let m = qwen.model.config.vision.spatialMergeSize
        var blocks: [QwenFrame] = []
        var visual = ""
        for (j, i) in stride(from: 0, to: fs.count, by: tp).enumerated() {
            let mean = ts[i ..< i + tp].reduce(0, +) / Double(tp)
            visual += String(format: "<%.1f seconds>", mean) + "<|vision_start|><|video_pad|><|vision_end|>"
            blocks.append(QwenFrame(time: mean, features: feats[j], rows: grid.0 / m, cols: grid.1 / m,
                                    width: fs[i].width, height: fs[i].height, prepSeconds: 0, visionSeconds: 0))
        }
        visual += String(format: "<%.1f-%.1f seconds>", Double(a), Double(a + 1))
        second = a
        self.frames = frames.count
        self.blocks = blocks
        self.visual = visual
        prep = t1.timeIntervalSince(t0)
        vision = Date().timeIntervalSince(t1)
    }
}
