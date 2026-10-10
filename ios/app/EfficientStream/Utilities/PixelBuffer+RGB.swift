import EfficientStreamKit
import UIKit

extension RGBImage {
    var uiImage: UIImage? { cgImage().map { UIImage(cgImage: $0) } }
}
