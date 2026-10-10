// Memory and thermal state of the process, for measurements.

import Foundation

public enum DeviceStats {
    /// (current, peak) physical footprint in MB: the number iOS compares with the app's memory limit.
    public static func memoryMB() -> (current: Double, peak: Double) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (0, 0) }
        return (Double(info.phys_footprint) / 1_048_576, Double(info.ledger_phys_footprint_peak) / 1_048_576)
    }

    public static var thermalState: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// e.g. "iPhone18,1" or "Mac16,10".
    public static var model: String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.machine", &buf, &size, nil, 0)
        let machine = String(cString: buf)
        if machine.hasPrefix("arm64") || machine.hasPrefix("x86") {
            sysctlbyname("hw.model", nil, &size, nil, 0)
            buf = [CChar](repeating: 0, count: max(size, 1))
            sysctlbyname("hw.model", &buf, &size, nil, 0)
            return String(cString: buf)
        }
        return machine
    }
}
