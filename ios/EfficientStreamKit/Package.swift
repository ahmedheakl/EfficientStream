// swift-tools-version: 6.0
// EfficientStreamKit: the EfficientStream pipeline (bounded frame store, MobileCLIP2-S0 frame index,
// MiniCPM-V-4.6 answerer) for iOS and macOS, on MLX and Core ML.
import PackageDescription

let package = Package(
    name: "EfficientStreamKit",
    platforms: [.iOS("18.2"), .macOS("15.0")],
    products: [
        .library(name: "EfficientStreamKit", targets: ["EfficientStreamKit"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.3.5")),
    ],
    targets: [
        .target(
            name: "EfficientStreamKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // only QwenVerify: the independent MLX port of Qwen3.5 to check ours against on the phone
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // Checks the Swift port against tensors dumped from the PyTorch package (tools/dump_reference.py).
        .executableTarget(
            name: "es-verify",
            dependencies: ["EfficientStreamKit"]
        ),
    ]
)
