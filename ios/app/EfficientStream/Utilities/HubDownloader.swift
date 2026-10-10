import Foundation

/// Downloads a folder of a Hugging Face model repo (e.g. the `ios/` folder holding the exported models) into a
/// local directory, file by file, with an optional access token for private repos.
@Observable
@MainActor
final class HubDownloader {
    private(set) var isRunning = false
    private(set) var progress: Double = 0
    private(set) var status = ""

    struct Entry: Decodable {
        let type: String
        let path: String
        let size: Int64?
    }

    func download(repo: String, folder: String, token: String, to destination: URL) async throws {
        isRunning = true
        defer { isRunning = false }
        let auth = token.trimmingCharacters(in: .whitespacesAndNewlines)
        func request(_ url: URL) -> URLRequest {
            var r = URLRequest(url: url)
            if !auth.isEmpty { r.setValue("Bearer \(auth)", forHTTPHeaderField: "Authorization") }
            return r
        }
        status = "Listing \(repo)/\(folder)…"
        let api = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main/\(folder)?recursive=true")!
        let (data, resp) = try await URLSession.shared.data(for: request(api))
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw NSError(domain: "Hub", code: (resp as? HTTPURLResponse)?.statusCode ?? -1, userInfo: [
                NSLocalizedDescriptionKey: "Could not list \(repo)/\(folder) (HTTP \((resp as? HTTPURLResponse)?.statusCode ?? -1)). Private repos need a token."
            ])
        }
        let files = try JSONDecoder().decode([Entry].self, from: data).filter { $0.type == "file" }
        let total = max(1, files.reduce(Int64(0)) { $0 + ($1.size ?? 0) })
        var done: Int64 = 0
        let prefix = folder.isEmpty ? "" : folder + "/"
        for f in files {
            let rel = f.path.hasPrefix(prefix) ? String(f.path.dropFirst(prefix.count)) : f.path
            let target = destination.appendingPathComponent(rel)
            if let a = try? FileManager.default.attributesOfItem(atPath: target.path),
                (a[.size] as? Int64) == f.size
            {
                done += f.size ?? 0
                continue
            }
            status = "Downloading \(rel)"
            let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(f.path)")!
            let (tmp, r) = try await URLSession.shared.download(for: request(url))
            guard (r as? HTTPURLResponse)?.statusCode == 200 else {
                throw NSError(domain: "Hub", code: -2, userInfo: [NSLocalizedDescriptionKey: "Download failed: \(rel)"])
            }
            try FileManager.default.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: tmp, to: target)
            done += f.size ?? 0
            progress = Double(done) / Double(total)
        }
        progress = 1
        status = "Done"
    }
}
