// Arm's-length Apple Foundation Models narrator.
//
// Reads a prompt on STDIN, asks the on-device model to respond, prints the reply to
// STDOUT. Keeps the LLM call out-of-process (the Python engine shells out to this), so
// the portable narrator is swappable and the engine never links a Swift framework.
//
// Local-first / zero-egress: Foundation Models runs ON DEVICE — no third-party egress,
// no API key. Claude is a separate optional narrator invoked through the local CLI.
//
// Build:  swiftc -O -o foundation_narrate foundation_narrate.swift
// Usage:  echo "PROMPT" | ./foundation_narrate
// Exit 0 + text on success; exit 2 + diagnostic on stderr if the model is unavailable.

import Foundation

#if canImport(FoundationModels)
import FoundationModels

@available(macOS 26.0, *)
func run() async -> Int32 {
    let prompt = String(data: FileHandle.standardInput.readDataToEndOfFile(),
                        encoding: .utf8) ?? ""
    if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        FileHandle.standardError.write("empty prompt\n".data(using: .utf8)!)
        return 2
    }

    let model = SystemLanguageModel.default
    switch model.availability {
    case .available:
        break
    case .unavailable(let reason):
        FileHandle.standardError.write(
            "on-device model unavailable: \(reason)\n".data(using: .utf8)!)
        return 2
    @unknown default:
        FileHandle.standardError.write("on-device model unavailable\n"
            .data(using: .utf8)!)
        return 2
    }
    let session = LanguageModelSession(model: model)

    do {
        // A low temperature keeps the narration grounded and steady (we are not after
        // creative variety — we want a faithful, warm reflection from the given notes).
        let options = GenerationOptions(temperature: 0.6)
        let response = try await session.respond(to: prompt, options: options)
        FileHandle.standardOutput.write(response.content.data(using: .utf8)!)
        return 0
    } catch {
        FileHandle.standardError.write(
            "generation failed: \(error)\n".data(using: .utf8)!)
        return 2
    }
}

if #available(macOS 26.0, *) {
    exit(await run())
} else {
    FileHandle.standardError.write(
        "FoundationModels requires macOS 26+\n".data(using: .utf8)!)
    exit(2)
}
#else
FileHandle.standardError.write(
    "FoundationModels framework not available in this SDK\n".data(using: .utf8)!)
exit(2)
#endif
