// Prompt text of the evaluated configuration (efficientstream/prompt.py) and the MiniCPM-V-4.6 chat template
// (backbone/chat_template.jinja with enable_thinking=False), with the image placeholders of its processor.

import Foundation

public enum PromptText {
    public static let system =
        "You are an expert multimodal assistant for streaming video QA. Answer the question "
        + "using the video frames shown, which cover the stream up to the current moment.\n"
        + "Place your final answer inside <answer> and </answer>."

    public static let finalView =
        "This is the final and richest view (the full timeline). <need_more_context/> is NO LONGER allowed. "
        + "Give your best-effort final answer inside <answer> and </answer>."

    public static let mcInstruction =
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer></answer> tags."

    public static let oeHead = "Here is the question:"
    public static let oeInstruction =
        "Give a COMPLETE, self-contained answer (a full sentence or two) inside <answer> and </answer>; "
        + "do not answer with a single word or refuse unless the question is truly unanswerable."

    public static let answerOpen = "<answer>"
    public static let answerClose = "</answer>"

    /// `question_block`: the text after the last image. Options are full strings such as "A. a red car".
    public static func questionBlock(_ question: String, options: [String]?) -> String {
        if let options, !options.isEmpty {
            return "\n" + question + "\n" + options.joined(separator: "\n") + "\n\n" + finalView + "\n"
                + mcInstruction
        }
        return "\n" + oeHead + "\n" + question + "\n\n" + finalView + "\n" + oeInstruction
    }
}

/// One item of the user turn: text, or an image (rendered as the image token, then expanded).
public enum PromptPart: Sendable {
    case text(String)
    case image(placeholder: String)
}

/// Special tokens and ids from the exported `prompt.json`.
public struct PromptSpec: Codable, Sendable {
    public var imageToken: String
    public var imageStartToken: String
    public var imageEndToken: String
    public var sliceStartToken: String
    public var sliceEndToken: String
    public var imageIdStartToken: String
    public var imageIdEndToken: String
    public var specialTokens: [String: Int]
    public var eosTokenIds: [Int]
    public var answerOpenIds: [Int]
    public var letterIds: [Int]
    public var imageTokenDivisor: Int

    enum CodingKeys: String, CodingKey {
        case imageToken = "image_token"
        case imageStartToken = "image_start_token"
        case imageEndToken = "image_end_token"
        case sliceStartToken = "slice_start_token"
        case sliceEndToken = "slice_end_token"
        case imageIdStartToken = "image_id_start_token"
        case imageIdEndToken = "image_id_end_token"
        case specialTokens = "special_tokens"
        case eosTokenIds = "eos_token_ids"
        case answerOpenIds = "answer_open_ids"
        case letterIds = "letter_ids"
        case imageTokenDivisor = "image_token_divisor"
    }

    public static func load(_ url: URL) throws -> PromptSpec {
        try JSONDecoder().decode(PromptSpec.self, from: Data(contentsOf: url))
    }

    /// The processor's `replace_image_token` (without the image-id prefix): the overview, then the slice rows.
    public func placeholder(overviewTokens: Int, sliceTokens: Int, rows: Int, cols: Int) -> String {
        var s = imageStartToken + String(repeating: imageToken, count: overviewTokens) + imageEndToken
        if rows > 0 && cols > 0 {
            let slice = sliceStartToken + String(repeating: imageToken, count: sliceTokens) + sliceEndToken
            s += Array(repeating: String(repeating: slice, count: cols), count: rows).joined(separator: "\n")
        }
        return s
    }

    /// The full prompt string: chat template (system + user turn + empty thinking block) with every image
    /// replaced by `<image_id>i</image_id>` + its placeholder (`_prepend_local_ids`).
    public func render(parts: [PromptPart]) -> String {
        let items: [String] = parts.map {
            switch $0 {
            case .text(let t): return t
            case .image: return imageToken
            }
        }
        let user = items.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        let system = PromptText.system.trimmingCharacters(in: .whitespacesAndNewlines)
        let chat = "<|im_start|>system\n" + system + "<|im_end|>\n" + "<|im_start|>user\n" + user + "<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let placeholders: [String] = parts.compactMap {
            if case .image(let p) = $0 { return p }
            return nil
        }
        let pieces = chat.components(separatedBy: imageToken)
        precondition(pieces.count == placeholders.count + 1, "one image token per image")
        var out = pieces[0]
        for (i, p) in placeholders.enumerated() {
            out += imageIdStartToken + "\(i)" + imageIdEndToken + p + pieces[i + 1]
        }
        return out
    }
}
