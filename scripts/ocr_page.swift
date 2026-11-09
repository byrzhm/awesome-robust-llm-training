// ocr_page.swift — 用 macOS Vision 框架对图片做 OCR（只读工具，用于读取论文中整页图片形式的页面）
// 用法: swift scripts/ocr_page.swift <image1> [image2 ...]
import Foundation
import Vision
import AppKit

func ocr(imagePath: String) {
    guard let image = NSImage(contentsOfFile: imagePath),
          let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        FileHandle.standardError.write("ERROR: cannot load \(imagePath)\n".data(using: .utf8)!)
        return
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try handler.perform([request])
        guard let observations = request.results else { return }
        var lines: [(String, CGFloat, CGFloat)] = []
        for obs in observations {
            if let candidate = obs.topCandidates(1).first {
                // VNRecognizedTextObservation.boundingBox，左下角坐标系，转成顶部距离以便按版面排序
                let minY = 1 - obs.boundingBox.minY
                let midX = (obs.boundingBox.minX + obs.boundingBox.maxX) / 2
                lines.append((candidate.string, minY, midX))
            }
        }
        lines.sort { $0.1 < $1.1 }
        print("===== \(imagePath) =====")
        for (text, y, x) in lines {
            print(String(format: "[y=%.3f x=%.3f] %@", y, x, text))
        }
    } catch {
        FileHandle.standardError.write("ERROR: \(error)\n".data(using: .utf8)!)
    }
}

for path in CommandLine.arguments.dropFirst() {
    ocr(imagePath: path)
}
