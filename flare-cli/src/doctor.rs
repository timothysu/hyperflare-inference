/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host preflight diagnostics for multi-GPU tensor-parallel serving over PCIe on Linux.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    path::{Path, PathBuf},
};

use serde::Serialize;

use crate::{cli::Doctor, inspect::Canvas};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const NVIDIA_PARAMS_PATH: &str = "/proc/driver/nvidia/params";
const CMDLINE_PATH: &str = "/proc/cmdline";
const SYSFS_ROOT: &str = "/sys";

const EXPECTED_REGISTRY_DWORDS: [(&str, &str, &[u64], &str); 4] = [
    (
        "nvidia.registry_dwords.force_p2p",
        "ForceP2P",
        &[0x11],
        "0x11",
    ),
    (
        "nvidia.registry_dwords.rm_force_p2p_type",
        "RMForceP2PType",
        &[1],
        "1",
    ),
    (
        "nvidia.registry_dwords.rm_pcie_p2p_type",
        "RMPcieP2PType",
        &[1, 2],
        "1 or 2",
    ),
    (
        "nvidia.registry_dwords.grdma_pci_topo_check_override",
        "GrdmaPciTopoCheckOverride",
        &[1],
        "1",
    ),
];

const NVIDIA_MODPROBE_REMEDIATION: &str = "Configure /etc/modprobe.d/nvidia.conf with: \
options nvidia NVreg_EnablePCIERelaxedOrderingMode=1 NVreg_EnableResizableBar=1 \
NVreg_RegistryDwords=\"ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1\" \
and reload the driver or reboot.";

const IOMMU_REMEDIATION: &str =
    "Set iommu=pt (or iommu=off / amd_iommu=off / intel_iommu=off) on the kernel command line and reboot.";

/// Outcome of one diagnostic check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Fail,
    Unverified,
    NotApplicable,
}

impl CheckStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Unverified => "UNVERIFIED",
            Self::NotApplicable => "N/A",
        }
    }

    fn styled(self, color: bool) -> String {
        let label = self.label();
        if !color {
            return label.to_owned();
        }
        let code = match self {
            Self::Pass => "1;32",
            Self::Fail => "1;31",
            Self::Unverified => "1;33",
            Self::NotApplicable => "2",
        };
        format!("\x1b[{code}m{label}\x1b[0m")
    }
}

/// GPU peer interconnect classification for the inspected GPUs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Interconnect {
    Pcie,
    NvLink,
    Mixed,
    SingleGpu,
    NoGpus,
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    Unknown,
}

impl Interconnect {
    fn description(self) -> &'static str {
        match self {
            Self::Pcie => "PCIe",
            Self::NvLink => "NVLink",
            Self::Mixed => "Mixed (NVLink + PCIe)",
            Self::SingleGpu => "Single GPU",
            Self::NoGpus => "No GPUs detected",
            Self::Unknown => "Unknown",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct InspectedGpu {
    pub index: u32,
    pub name: String,
    pub pci_bus_id: String,
    pub uuid: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Check {
    pub id: String,
    pub name: String,
    pub status: CheckStatus,
    pub expected: Option<String>,
    pub actual: Option<String>,
    pub detail: String,
    pub remediation: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Report {
    pub platform: &'static str,
    pub interconnect: Interconnect,
    pub pcie_tp_checks_required: bool,
    pub gpus: Vec<InspectedGpu>,
    pub bridges: Vec<String>,
    pub notes: Vec<String>,
    pub checks: Vec<Check>,
    pub ok: bool,
    pub all_verified: bool,
}

impl Report {
    fn finalize(mut self) -> Self {
        self.ok = !self
            .checks
            .iter()
            .any(|check| check.status == CheckStatus::Fail);
        self.all_verified = self.ok
            && !self
                .checks
                .iter()
                .any(|check| check.status == CheckStatus::Unverified);
        self
    }
}

/// Run `flare doctor` and return whether all verified checks passed (`true` = exit 0).
pub fn run(args: Doctor) -> Result<bool> {
    let report = diagnose(&args.gpus)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let canvas = Canvas::terminal(args.plain);
        print!("{}", render(&report, canvas));
    }
    Ok(report.ok)
}

/// Enforce Linux PCIe TP>1 basic checks (NVIDIA driver params and IOMMU, without ACS) during
/// `flare run` immediately after CUDA GPU initialization and before model memory allocation.
///
/// Checks `/proc/driver/nvidia/params` and Linux IOMMU first so healthy hosts avoid loading NVML.
/// Only when a check fails does it query NVML for pairwise NVLink P2P on the initialized PCI bus
/// IDs, returning `Ok(())` when all pairs are linked by NVLink and an aborting error otherwise.
#[cfg(target_os = "linux")]
pub fn check_pcie_tp_after_gpu_init(pci_bus_ids: &[String]) -> std::result::Result<(), String> {
    if pci_bus_ids.len() <= 1 {
        return Ok(());
    }
    let gpu_bdfs: Vec<String> = pci_bus_ids
        .iter()
        .map(|raw| normalize_pci_bdf(raw).unwrap_or_else(|| raw.trim().to_ascii_lowercase()))
        .collect();
    let failed = evaluate_run_basic_checks(
        &gpu_bdfs,
        Path::new(SYSFS_ROOT),
        Path::new(NVIDIA_PARAMS_PATH),
        Path::new(CMDLINE_PATH),
    );
    if failed.is_empty() {
        return Ok(());
    }
    if all_pairs_nvlink_by_pci_bdfs(&gpu_bdfs) == Some(true) {
        return Ok(());
    }
    Err(format_run_check_error(&failed))
}

#[cfg(not(target_os = "linux"))]
pub fn check_pcie_tp_after_gpu_init(_pci_bus_ids: &[String]) -> std::result::Result<(), String> {
    Ok(())
}

fn diagnose(requested_gpus: &[i32]) -> Result<Report> {
    #[cfg(target_os = "linux")]
    {
        let inherited = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        let probe = probe_linux_gpus(requested_gpus, inherited.as_deref())?;
        Ok(evaluate_linux_host(
            probe,
            Path::new(SYSFS_ROOT),
            Path::new(NVIDIA_PARAMS_PATH),
            Path::new(CMDLINE_PATH),
        ))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = requested_gpus;
        Ok(Report {
            platform: std::env::consts::OS,
            interconnect: Interconnect::Unknown,
            pcie_tp_checks_required: false,
            gpus: Vec::new(),
            bridges: Vec::new(),
            notes: vec![
                "PCIe tensor-parallel host preflight checks are implemented for Linux only."
                    .into(),
            ],
            checks: vec![Check {
                id: "host.platform".into(),
                name: "Linux PCIe TP preflight".into(),
                status: CheckStatus::NotApplicable,
                expected: None,
                actual: Some(std::env::consts::OS.into()),
                detail: "Checks inspect /proc/driver/nvidia/params, Linux IOMMU groups, and sysfs PCIe ACS registers.".into(),
                remediation: None,
            }],
            ok: true,
            all_verified: true,
        })
    }
}

#[derive(Clone, Debug)]
struct GpuProbe {
    gpus: Vec<InspectedGpu>,
    interconnect: Interconnect,
    notes: Vec<String>,
}

fn evaluate_linux_host(
    probe: GpuProbe,
    sysfs_root: &Path,
    nvidia_params_path: &Path,
    cmdline_path: &Path,
) -> Report {
    let gpu_bdfs: Vec<String> = probe.gpus.iter().map(|g| g.pci_bus_id.clone()).collect();
    let ancestor_chains = collect_ancestor_chains(sysfs_root, &gpu_bdfs);
    let bridges = relevant_bridges(&ancestor_chains);

    let pcie_tp_checks_required = matches!(
        probe.interconnect,
        Interconnect::Pcie | Interconnect::Mixed
    );

    let mut notes = probe.notes;
    if probe.interconnect == Interconnect::NvLink {
        notes.push(
            "All selected GPU pairs have NVLink P2P; PCIe TP>1 tuning checks are not required."
                .into(),
        );
        return Report {
            platform: "linux",
            interconnect: probe.interconnect,
            pcie_tp_checks_required: false,
            gpus: probe.gpus,
            bridges,
            notes,
            checks: vec![Check {
                id: "pcie.interconnect".into(),
                name: "PCIe TP>1 preflight".into(),
                status: CheckStatus::NotApplicable,
                expected: None,
                actual: Some("NVLink".into()),
                detail: "Selected GPUs are connected over NVLink.".into(),
                remediation: None,
            }],
            ok: true,
            all_verified: true,
        };
    }

    if matches!(
        probe.interconnect,
        Interconnect::SingleGpu | Interconnect::NoGpus
    ) {
        notes.push(
            "Fewer than two GPUs selected; reporting host readiness for PCIe TP>1 serving.".into(),
        );
    }

    let params_text = std::fs::read_to_string(nvidia_params_path);
    let params_result = params_text
        .as_deref()
        .map_err(|error| format!("cannot read {}: {error}", nvidia_params_path.display()));
    let mut checks = check_nvidia_params(params_result);

    let cmdline = std::fs::read_to_string(cmdline_path).ok();
    checks.push(check_iommu(sysfs_root, &gpu_bdfs, cmdline.as_deref()));
    checks.push(check_acs(sysfs_root, &bridges));

    Report {
        platform: "linux",
        interconnect: probe.interconnect,
        pcie_tp_checks_required,
        gpus: probe.gpus,
        bridges,
        notes,
        checks,
        ok: false,
        all_verified: false,
    }
    .finalize()
}

/// Evaluate the basic `flare run` Linux PCIe TP>1 checks (NVIDIA driver params and IOMMU, no ACS)
/// and return any failed checks.
pub(crate) fn evaluate_run_basic_checks(
    gpu_bdfs: &[String],
    sysfs_root: &Path,
    nvidia_params_path: &Path,
    cmdline_path: &Path,
) -> Vec<Check> {
    let params_text = std::fs::read_to_string(nvidia_params_path);
    let params_result = params_text
        .as_deref()
        .map_err(|error| format!("cannot read {}: {error}", nvidia_params_path.display()));
    let mut checks = check_nvidia_params(params_result);
    let cmdline = std::fs::read_to_string(cmdline_path).ok();
    checks.push(check_iommu(sysfs_root, gpu_bdfs, cmdline.as_deref()));
    checks
        .into_iter()
        .filter(|check| check.status == CheckStatus::Fail)
        .collect()
}

/// Format the aborting error message for `flare run` when one or more basic PCIe TP>1 checks fail.
pub(crate) fn format_run_check_error(failed: &[Check]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "PCIe TP>1 performance checks failed ({} issue{}):",
        failed.len(),
        if failed.len() == 1 { "" } else { "s" }
    );
    for check in failed {
        let _ = writeln!(out, "  - {}: {}", check.name, check.detail);
    }
    let _ = write!(
        out,
        "Run `flare doctor` for details and remediation, or pass `--bypass-performance-check` to override."
    );
    out
}

/// Check via NVML whether every pair of the given PCI bus IDs has NVLink P2P, short-circuiting on
/// the first non-NVLink pair. Returns `None` when NVML cannot be loaded or queried.
#[cfg(target_os = "linux")]
fn all_pairs_nvlink_by_pci_bdfs(gpu_bdfs: &[String]) -> Option<bool> {
    use std::ffi::{CString, c_char, c_int, c_uint, c_void};

    type Device = *mut c_void;
    const NVML_P2P_CAPS_INDEX_NVLINK: c_uint = 2;
    const NVML_P2P_STATUS_OK: c_uint = 0;

    if gpu_bdfs.len() <= 1 {
        return Some(false);
    }
    // SAFETY: Only the system NVML library is loaded and kept alive for the duration of this call.
    let library = unsafe { libloading::Library::new("libnvidia-ml.so.1") }.ok()?;
    // SAFETY: Function signatures match nvml.h; shutdown balances init before library drop.
    unsafe {
        let init = library
            .get::<unsafe extern "C" fn() -> c_int>(b"nvmlInit_v2\0")
            .ok()?;
        let shutdown = library
            .get::<unsafe extern "C" fn() -> c_int>(b"nvmlShutdown\0")
            .ok()?;
        let get_by_pci = library
            .get::<unsafe extern "C" fn(*const c_char, *mut Device) -> c_int>(
                b"nvmlDeviceGetHandleByPciBusId_v2\0",
            )
            .ok()?;
        let get_p2p = library
            .get::<unsafe extern "C" fn(Device, Device, c_uint, *mut c_uint) -> c_int>(
                b"nvmlDeviceGetP2PStatus\0",
            )
            .ok()?;
        if init() != 0 {
            return None;
        }
        let result = (|| -> Option<bool> {
            let mut handles = Vec::with_capacity(gpu_bdfs.len());
            for bdf in gpu_bdfs {
                let c_bdf = CString::new(bdf.as_str()).ok()?;
                let mut device: Device = std::ptr::null_mut();
                if get_by_pci(c_bdf.as_ptr(), &mut device) != 0 {
                    return None;
                }
                handles.push(device);
            }
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    let mut status: c_uint = 0;
                    if get_p2p(
                        handles[i],
                        handles[j],
                        NVML_P2P_CAPS_INDEX_NVLINK,
                        &mut status,
                    ) != 0
                        || status != NVML_P2P_STATUS_OK
                    {
                        return Some(false);
                    }
                }
            }
            Some(true)
        })();
        let _ = shutdown();
        result
    }
}

/// Render a human-readable `flare doctor` report into `canvas`.
pub fn render(report: &Report, mut canvas: Canvas) -> String {
    let color = canvas.is_color();
    canvas.open("Flare doctor · PCIe TP>1 preflight");
    canvas.row("platform", report.platform);
    canvas.row("interconnect", report.interconnect.description());
    canvas.row(
        "gpus",
        &if report.gpus.is_empty() {
            "none detected".to_owned()
        } else {
            format!("{} selected", report.gpus.len())
        },
    );
    if !report.bridges.is_empty() {
        canvas.row(
            "pcie bridges",
            &format!("{} on P2P paths", report.bridges.len()),
        );
    }
    for note in &report.notes {
        canvas.note(note);
    }

    if !report.gpus.is_empty() {
        let rows: Vec<Vec<String>> = report
            .gpus
            .iter()
            .map(|gpu| vec![gpu.index.to_string(), gpu.pci_bus_id.clone(), gpu.name.clone()])
            .collect();
        canvas.open("Selected GPUs");
        canvas.table(&["GPU", "PCI Bus ID", "Model"], &rows);
    }

    canvas.open("Preflight checks");
    let rows: Vec<Vec<String>> = report
        .checks
        .iter()
        .map(|check| {
            vec![
                check.status.styled(color),
                check.name.clone(),
                check.detail.clone(),
            ]
        })
        .collect();
    canvas.table(&["Status", "Check", "Detail"], &rows);

    let mut remediations = Vec::new();
    for check in &report.checks {
        if matches!(check.status, CheckStatus::Fail | CheckStatus::Unverified)
            && let Some(hint) = &check.remediation
            && !remediations.contains(hint)
        {
            remediations.push(hint.clone());
        }
    }
    if !remediations.is_empty() {
        canvas.open("Remediation");
        for hint in &remediations {
            canvas.note(hint);
        }
    }

    let passed = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Pass)
        .count();
    let failed = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .count();
    let unverified = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Unverified)
        .count();
    let na = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::NotApplicable)
        .count();

    let mut summary = Vec::new();
    if passed > 0 {
        summary.push(format!("{passed} passed"));
    }
    if failed > 0 {
        summary.push(format!("{failed} failed"));
    }
    if unverified > 0 {
        summary.push(format!("{unverified} unverified"));
    }
    if na > 0 {
        summary.push(format!("{na} not applicable"));
    }
    canvas.open("Summary");
    canvas.row("result", &summary.join(", "));
    if failed > 0 {
        canvas.row("status", &CheckStatus::Fail.styled(color));
    } else if unverified > 0 {
        canvas.row(
            "status",
            &format!(
                "{} (run with sudo to complete privileged checks)",
                CheckStatus::Unverified.styled(color)
            ),
        );
    } else {
        canvas.row("status", &CheckStatus::Pass.styled(color));
    }

    canvas.finish()
}

/// Parse `/proc/driver/nvidia/params` into `key -> value`, stripping outer double quotes.
pub(crate) fn parse_nvidia_params(text: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        let value = value.trim();
        let unquoted = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(value);
        map.insert(key.to_owned(), unquoted.to_owned());
    }
    map
}

/// Parse semicolon-delimited `Key=Value` entries from `RegistryDwords`.
pub(crate) fn parse_registry_dwords(raw: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for entry in raw.split(';') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let Some((key, value)) = entry.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if !key.is_empty() {
            map.insert(key.to_owned(), value.to_owned());
        }
    }
    map
}

/// Parse a decimal or `0x`-prefixed hexadecimal unsigned integer.
pub(crate) fn parse_u64_param(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

/// Evaluate `/proc/driver/nvidia/params` top-level keys and `RegistryDwords`.
pub(crate) fn check_nvidia_params(params_text: std::result::Result<&str, String>) -> Vec<Check> {
    let Ok(text) = params_text else {
        let error = params_text.unwrap_err();
        return vec![Check {
            id: "nvidia.params".into(),
            name: "NVIDIA driver params".into(),
            status: CheckStatus::Fail,
            expected: Some(NVIDIA_PARAMS_PATH.into()),
            actual: None,
            detail: error,
            remediation: Some(format!(
                "Ensure the NVIDIA kernel module is loaded. {NVIDIA_MODPROBE_REMEDIATION}"
            )),
        }];
    };

    let params = parse_nvidia_params(text);
    let mut checks = Vec::with_capacity(2 + EXPECTED_REGISTRY_DWORDS.len());

    for (id, key, expected_val, expected_display) in [
        (
            "nvidia.enable_pcie_relaxed_ordering_mode",
            "EnablePCIERelaxedOrderingMode",
            1u64,
            "1",
        ),
        (
            "nvidia.enable_resizable_bar",
            "EnableResizableBar",
            1u64,
            "1",
        ),
    ] {
        let actual = params.get(key).cloned();
        let matches = actual
            .as_deref()
            .and_then(parse_u64_param)
            .is_some_and(|v| v == expected_val);
        checks.push(Check {
            id: id.into(),
            name: format!("NVIDIA {key}"),
            status: if matches {
                CheckStatus::Pass
            } else {
                CheckStatus::Fail
            },
            expected: Some(expected_display.into()),
            actual: actual.clone(),
            detail: if matches {
                format!("{key}: {}", actual.as_deref().unwrap_or(expected_display))
            } else {
                format!(
                    "{key} is {}; expected {expected_display}",
                    actual.as_deref().unwrap_or("absent")
                )
            },
            remediation: (!matches).then(|| NVIDIA_MODPROBE_REMEDIATION.into()),
        });
    }

    let dwords_raw = params.get("RegistryDwords").map(String::as_str).unwrap_or("");
    let dwords = parse_registry_dwords(dwords_raw);
    for (id, key, expected_vals, expected_display) in EXPECTED_REGISTRY_DWORDS {
        let actual = dwords.get(key).cloned();
        let matches = actual
            .as_deref()
            .and_then(parse_u64_param)
            .is_some_and(|v| expected_vals.contains(&v));
        checks.push(Check {
            id: id.into(),
            name: format!("RegistryDwords {key}"),
            status: if matches {
                CheckStatus::Pass
            } else {
                CheckStatus::Fail
            },
            expected: Some(expected_display.into()),
            actual: actual.clone(),
            detail: if matches {
                format!("{key}={}", actual.as_deref().unwrap_or(expected_display))
            } else {
                format!(
                    "{key} is {} in RegistryDwords; expected {expected_display}",
                    actual.as_deref().unwrap_or("unset")
                )
            },
            remediation: (!matches).then(|| NVIDIA_MODPROBE_REMEDIATION.into()),
        });
    }

    checks
}

/// Evaluate whether IOMMU is `off` or `pt` (identity passthrough) for the selected GPUs.
pub(crate) fn check_iommu(
    sysfs_root: &Path,
    gpu_bdfs: &[String],
    cmdline: Option<&str>,
) -> Check {
    let cmdline_flags = cmdline.map(extract_iommu_cmdline_flags).unwrap_or_default();
    let cmdline_suffix = if cmdline_flags.is_empty() {
        String::new()
    } else {
        format!(" (cmdline: {})", cmdline_flags.join(" "))
    };

    let mut group_types: Vec<(String, Option<String>)> = Vec::new();
    if !gpu_bdfs.is_empty() {
        for bdf in gpu_bdfs {
            let group_path = sysfs_root
                .join("bus/pci/devices")
                .join(bdf)
                .join("iommu_group");
            if group_path.exists() {
                let kind = std::fs::read_to_string(group_path.join("type"))
                    .ok()
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty());
                group_types.push((bdf.clone(), kind));
            }
        }
    } else {
        let groups_dir = sysfs_root.join("kernel/iommu_groups");
        if let Ok(entries) = std::fs::read_dir(&groups_dir) {
            let mut names: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            for name in names {
                let kind = std::fs::read_to_string(groups_dir.join(&name).join("type"))
                    .ok()
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty());
                group_types.push((format!("group {name}"), kind));
            }
        }
    }

    if group_types.is_empty() {
        // When no GPU has an iommu_group (and no global iommu_groups exist), IOMMU is off.
        return Check {
            id: "kernel.iommu".into(),
            name: "IOMMU mode".into(),
            status: CheckStatus::Pass,
            expected: Some("pt or off".into()),
            actual: Some("off".into()),
            detail: format!("IOMMU is off (no active IOMMU groups){cmdline_suffix}"),
            remediation: None,
        };
    }

    let mut non_identity = Vec::new();
    let mut missing_type = Vec::new();
    let mut identity_count = 0usize;
    for (subject, kind) in &group_types {
        match kind.as_deref() {
            Some("identity") => identity_count += 1,
            Some(other) => non_identity.push(format!("{subject}={other}")),
            None => missing_type.push(subject.clone()),
        }
    }

    if !non_identity.is_empty() {
        let actual_kinds: BTreeSet<&str> = group_types
            .iter()
            .filter_map(|(_, k)| k.as_deref())
            .filter(|&k| k != "identity")
            .collect();
        let actual = actual_kinds.into_iter().collect::<Vec<_>>().join(", ");
        return Check {
            id: "kernel.iommu".into(),
            name: "IOMMU mode".into(),
            status: CheckStatus::Fail,
            expected: Some("pt or off".into()),
            actual: Some(actual),
            detail: format!(
                "IOMMU is translating DMA ({}){cmdline_suffix}; expected pt (identity) or off",
                non_identity.join(", ")
            ),
            remediation: Some(IOMMU_REMEDIATION.into()),
        };
    }

    if missing_type.is_empty() && identity_count == group_types.len() {
        return Check {
            id: "kernel.iommu".into(),
            name: "IOMMU mode".into(),
            status: CheckStatus::Pass,
            expected: Some("pt or off".into()),
            actual: Some("pt (identity)".into()),
            detail: format!(
                "IOMMU is in passthrough mode (identity on {} group{}){cmdline_suffix}",
                identity_count,
                if identity_count == 1 { "" } else { "s" }
            ),
            remediation: None,
        };
    }

    // Fallback for kernels without /sys/kernel/iommu_groups/*/type.
    let cmdline_pt = cmdline_flags
        .iter()
        .any(|flag| flag == "iommu=pt" || flag == "iommu.passthrough=1");
    let cmdline_strict = cmdline_flags
        .iter()
        .any(|flag| flag == "iommu.passthrough=0" || flag == "iommu=force");
    if cmdline_pt && !cmdline_strict {
        Check {
            id: "kernel.iommu".into(),
            name: "IOMMU mode".into(),
            status: CheckStatus::Pass,
            expected: Some("pt or off".into()),
            actual: Some("pt".into()),
            detail: format!("IOMMU passthrough requested on kernel command line{cmdline_suffix}"),
            remediation: None,
        }
    } else if cmdline_strict {
        Check {
            id: "kernel.iommu".into(),
            name: "IOMMU mode".into(),
            status: CheckStatus::Fail,
            expected: Some("pt or off".into()),
            actual: Some("enabled".into()),
            detail: format!("IOMMU groups are active with non-passthrough cmdline{cmdline_suffix}"),
            remediation: Some(IOMMU_REMEDIATION.into()),
        }
    } else {
        Check {
            id: "kernel.iommu".into(),
            name: "IOMMU mode".into(),
            status: CheckStatus::Unverified,
            expected: Some("pt or off".into()),
            actual: Some("active (group type unavailable)".into()),
            detail: format!(
                "IOMMU groups are present, but sysfs group type is unavailable and kernel cmdline does not specify iommu=pt{cmdline_suffix}"
            ),
            remediation: Some(IOMMU_REMEDIATION.into()),
        }
    }
}

fn extract_iommu_cmdline_flags(cmdline: &str) -> Vec<String> {
    cmdline
        .split_whitespace()
        .filter(|token| {
            token.starts_with("iommu=")
                || token.starts_with("iommu.")
                || token.starts_with("amd_iommu=")
                || token.starts_with("intel_iommu=")
                || token.starts_with("arm-smmu.")
        })
        .map(str::to_owned)
        .collect()
}

/// Normalize a PCI Bus ID such as `00000000:0F:00.0` to Linux sysfs form `0000:0f:00.0`.
pub(crate) fn normalize_pci_bdf(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let mut parts = trimmed.split(':');
    let domain_str = parts.next()?;
    let bus_str = parts.next()?;
    let dev_fn = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let (dev_str, fn_str) = dev_fn.split_once('.')?;
    if domain_str.is_empty()
        || bus_str.len() != 2
        || dev_str.len() != 2
        || fn_str.len() != 1
    {
        return None;
    }
    let domain = u32::from_str_radix(domain_str, 16).ok()?;
    let bus = u8::from_str_radix(bus_str, 16).ok()?;
    let dev = u8::from_str_radix(dev_str, 16).ok()?;
    let func = u8::from_str_radix(fn_str, 16).ok()?;
    if func > 7 {
        return None;
    }
    Some(format!("{domain:04x}:{bus:02x}:{dev:02x}.{func:x}"))
}

/// Extract upstream PCI bridge BDFs (root-to-parent order) from a canonical sysfs PCI device path.
pub(crate) fn pci_ancestor_bdfs(canonical_device_path: &Path) -> Vec<String> {
    let mut components: Vec<String> = canonical_device_path
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .filter_map(normalize_pci_bdf)
        .collect();
    // Drop the endpoint GPU itself; only ancestor root ports and switch bridges remain.
    if !components.is_empty() {
        components.pop();
    }
    components
}

fn collect_ancestor_chains(sysfs_root: &Path, gpu_bdfs: &[String]) -> Vec<Vec<String>> {
    gpu_bdfs
        .iter()
        .filter_map(|bdf| {
            let link = sysfs_root.join("bus/pci/devices").join(bdf);
            let canonical = std::fs::canonicalize(&link).ok()?;
            Some(pci_ancestor_bdfs(&canonical))
        })
        .collect()
}

/// Determine the PCIe bridges/switches on the P2P paths connecting `ancestor_chains`.
///
/// For two or more GPUs, computes their global lowest common ancestor (LCA) in the PCIe tree and
/// returns the LCA plus every bridge beneath it along each GPU's path. Bridges strictly above the
/// LCA are never traversed by P2P traffic among the selected GPUs. When GPUs sit under different
/// root ports (empty common prefix), every ancestor bridge up to and including each root port is
/// included. For a single GPU, all of its ancestor bridges are returned.
pub(crate) fn relevant_bridges(ancestor_chains: &[Vec<String>]) -> Vec<String> {
    if ancestor_chains.is_empty() {
        return Vec::new();
    }
    let start_index = if ancestor_chains.len() >= 2 {
        let min_len = ancestor_chains.iter().map(Vec::len).min().unwrap_or(0);
        let mut lcp = 0;
        while lcp < min_len
            && ancestor_chains
                .iter()
                .all(|chain| chain[lcp] == ancestor_chains[0][lcp])
        {
            lcp += 1;
        }
        lcp.saturating_sub(1)
    } else {
        0
    };

    let mut seen = BTreeSet::new();
    let mut bridges = Vec::new();
    for chain in ancestor_chains {
        for bdf in chain.iter().skip(start_index) {
            if seen.insert(bdf.clone()) {
                bridges.push(bdf.clone());
            }
        }
    }
    bridges
}

/// PCIe Extended Capability ID for Access Control Services (ACS).
const PCIE_EXT_CAP_ID_ACS: u16 = 0x000d;
/// ACS Control bits that redirect or restrict PCIe P2P traffic:
/// ReqRedir (bit 2), CmpltRedir (bit 3), UpstreamFwd (bit 4), EgressCtrl (bit 5).
const PCI_ACS_P2P_REDIRECT_MASK: u16 = 0x003c;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AcsRegisters {
    pub offset: usize,
    pub control: u16,
}

impl AcsRegisters {
    pub(crate) fn p2p_redirect_bits(self) -> u16 {
        self.control & PCI_ACS_P2P_REDIRECT_MASK
    }

    pub(crate) fn enabled_flags(self) -> Vec<&'static str> {
        const FLAGS: [(u16, &str); 7] = [
            (0x0001, "SrcValid+"),
            (0x0002, "TransBlk+"),
            (0x0004, "ReqRedir+"),
            (0x0008, "CmpltRedir+"),
            (0x0010, "UpstreamFwd+"),
            (0x0020, "EgressCtrl+"),
            (0x0040, "DirectTrans+"),
        ];
        FLAGS
            .iter()
            .filter_map(|&(mask, name)| (self.control & mask != 0).then_some(name))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AcsConfigRead {
    /// PCIe extended config space (>= 0x100) was not readable (typically needs CAP_SYS_ADMIN).
    ExtendedSpaceUnavailable,
    /// Extended config space was readable and this bridge has no ACS extended capability.
    NotPresent,
    /// Bridge exposes the ACS extended capability with the decoded control register.
    Present(AcsRegisters),
}

/// Walk the PCIe Extended Capability linked list starting at offset 0x100 using targeted reads.
///
/// Only the 4-byte extended capability headers along the chain and the 2-byte ACS Control register
/// at `cap + 0x06` are read. Dumping the full 4 KiB config space is unsafe on PCIe switches (such
/// as Broadcom PEX890xx) because vendor-specific registers in extended config space can have read
/// side effects that trigger Downstream Port Containment (DPC).
pub(crate) fn parse_acs_capability_with<F>(config_len: usize, mut read_at: F) -> AcsConfigRead
where
    F: FnMut(usize, &mut [u8]) -> std::io::Result<()>,
{
    if config_len < 0x104 {
        return AcsConfigRead::ExtendedSpaceUnavailable;
    }
    let mut visited = BTreeSet::new();
    let mut offset: usize = 0x100;
    while offset >= 0x100 && offset + 4 <= config_len {
        if !visited.insert(offset) {
            break;
        }
        let mut hdr = [0u8; 4];
        if read_at(offset, &mut hdr).is_err() {
            return AcsConfigRead::ExtendedSpaceUnavailable;
        }
        let header = u32::from_le_bytes(hdr);
        if header == 0 || header == u32::MAX {
            break;
        }
        let cap_id = (header & 0xffff) as u16;
        let next = ((header >> 20) & 0xfff) as usize;
        if cap_id == PCIE_EXT_CAP_ID_ACS {
            if offset + 8 > config_len {
                return AcsConfigRead::ExtendedSpaceUnavailable;
            }
            let mut ctl = [0u8; 2];
            if read_at(offset + 6, &mut ctl).is_err() {
                return AcsConfigRead::ExtendedSpaceUnavailable;
            }
            let control = u16::from_le_bytes(ctl);
            return AcsConfigRead::Present(AcsRegisters { offset, control });
        }
        if next == 0 || next < 0x100 || next % 4 != 0 {
            break;
        }
        offset = next;
    }
    AcsConfigRead::NotPresent
}

#[cfg(unix)]
fn read_acs_capability_from_path(config_path: &Path) -> AcsConfigRead {
    use std::os::unix::fs::FileExt;

    let Ok(file) = std::fs::File::open(config_path) else {
        return AcsConfigRead::ExtendedSpaceUnavailable;
    };
    let Ok(metadata) = file.metadata() else {
        return AcsConfigRead::ExtendedSpaceUnavailable;
    };
    let len = usize::try_from(metadata.len()).unwrap_or(0);
    // Issue single aligned pread64 calls for capability headers (4 B) and ACSCtl (2 B).
    parse_acs_capability_with(len, |offset, buf| file.read_exact_at(buf, offset as u64))
}

#[cfg(not(unix))]
fn read_acs_capability_from_path(config_path: &Path) -> AcsConfigRead {
    use std::io::{Read, Seek, SeekFrom};

    let Ok(mut file) = std::fs::File::open(config_path) else {
        return AcsConfigRead::ExtendedSpaceUnavailable;
    };
    let Ok(metadata) = file.metadata() else {
        return AcsConfigRead::ExtendedSpaceUnavailable;
    };
    let len = usize::try_from(metadata.len()).unwrap_or(0);
    parse_acs_capability_with(len, |offset, buf| {
        file.seek(SeekFrom::Start(offset as u64))?;
        file.read_exact(buf)
    })
}

/// Check PCIe ACS control registers on every bridge in `bridges`.
pub(crate) fn check_acs(sysfs_root: &Path, bridges: &[String]) -> Check {
    if bridges.is_empty() {
        return Check {
            id: "pcie.acs".into(),
            name: "PCIe ACS".into(),
            status: CheckStatus::Unverified,
            expected: Some("disabled on P2P bridges".into()),
            actual: None,
            detail: "No PCIe ancestor bridges found for the selected GPUs.".into(),
            remediation: None,
        };
    }

    let mut enabled: Vec<(String, AcsRegisters)> = Vec::new();
    let mut unreadable: Vec<String> = Vec::new();
    let mut disabled_count = 0usize;
    let mut no_acs_count = 0usize;

    for bdf in bridges {
        let config_path = sysfs_root
            .join("bus/pci/devices")
            .join(bdf)
            .join("config");
        match read_acs_capability_from_path(&config_path) {
            AcsConfigRead::ExtendedSpaceUnavailable => unreadable.push(bdf.clone()),
            AcsConfigRead::NotPresent => no_acs_count += 1,
            AcsConfigRead::Present(regs) => {
                if regs.p2p_redirect_bits() != 0 {
                    enabled.push((bdf.clone(), regs));
                } else {
                    disabled_count += 1;
                }
            }
        }
    }

    if !enabled.is_empty() {
        let offending: Vec<String> = enabled
            .iter()
            .map(|(bdf, regs)| {
                format!(
                    "{bdf} (ACSCtl={:#06x}: {})",
                    regs.control,
                    regs.enabled_flags().join(" ")
                )
            })
            .collect();
        let setpci_cmds: Vec<String> = enabled
            .iter()
            .map(|(bdf, _)| format!("setpci -s {bdf} ECAP_ACS+0x6.w=0000"))
            .collect();
        return Check {
            id: "pcie.acs".into(),
            name: "PCIe ACS".into(),
            status: CheckStatus::Fail,
            expected: Some("disabled on P2P bridges".into()),
            actual: Some(format!("enabled on {} bridge(s)", enabled.len())),
            detail: format!("ACS P2P redirection enabled on {}", offending.join(", ")),
            remediation: Some(format!(
                "Disable ACS in BIOS/firmware, add pci=disable_acs_redir=... to the kernel command line, or run as root: {}",
                setpci_cmds.join("; ")
            )),
        };
    }

    if !unreadable.is_empty() {
        return Check {
            id: "pcie.acs".into(),
            name: "PCIe ACS".into(),
            status: CheckStatus::Unverified,
            expected: Some("disabled on P2P bridges".into()),
            actual: Some("unreadable without root".into()),
            detail: format!(
                "Cannot read PCIe extended config space for {} of {} bridge(s) without root / CAP_SYS_ADMIN ({})",
                unreadable.len(),
                bridges.len(),
                unreadable.join(", ")
            ),
            remediation: Some(
                "Run `sudo flare doctor` to inspect PCIe ACS extended capability registers.".into(),
            ),
        };
    }

    Check {
        id: "pcie.acs".into(),
        name: "PCIe ACS".into(),
        status: CheckStatus::Pass,
        expected: Some("disabled on P2P bridges".into()),
        actual: Some("disabled".into()),
        detail: format!(
            "Disabled across {} bridge(s) ({disabled_count} with ACS capability disabled, {no_acs_count} without ACS)",
            bridges.len()
        ),
        remediation: None,
    }
}

#[cfg(target_os = "linux")]
fn probe_linux_gpus(requested: &[i32], inherited: Option<&str>) -> Result<GpuProbe> {
    match probe_nvml_gpus(requested, inherited) {
        Ok(probe) => Ok(probe),
        Err(nvml_error) => {
            // Invalid --gpus / CUDA_VISIBLE_DEVICES selections are user errors, not driver absences.
            if !requested.is_empty() {
                crate::environment::visibility(requested, inherited)?;
            }
            let fallback = probe_proc_nvidia_gpus(Path::new("/proc/driver/nvidia/gpus"));
            if fallback.is_empty() {
                Ok(GpuProbe {
                    gpus: Vec::new(),
                    interconnect: Interconnect::NoGpus,
                    notes: vec![format!(
                        "NVML unavailable ({nvml_error}); no GPUs found under /proc/driver/nvidia/gpus."
                    )],
                })
            } else {
                let interconnect = if fallback.len() >= 2 {
                    Interconnect::Pcie
                } else {
                    Interconnect::SingleGpu
                };
                Ok(GpuProbe {
                    gpus: fallback,
                    interconnect,
                    notes: vec![format!(
                        "NVML unavailable ({nvml_error}); fell back to /proc/driver/nvidia/gpus (NVLink status unverified)."
                    )],
                })
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn probe_nvml_gpus(requested: &[i32], inherited: Option<&str>) -> Result<GpuProbe> {
    use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

    type Device = *mut c_void;

    #[repr(C)]
    struct PciInfo {
        _bus_id_legacy: [c_char; 16],
        _domain: c_uint,
        _bus: c_uint,
        _device: c_uint,
        _device_id: c_uint,
        _subsystem_id: c_uint,
        bus_id: [c_char; 32],
    }

    const NVML_P2P_CAPS_INDEX_NVLINK: c_uint = 2;
    const NVML_P2P_STATUS_OK: c_uint = 0;

    // Resolve explicit --gpus or inherited CUDA_VISIBLE_DEVICES before loading NVML.
    let explicit_selection: Option<Vec<String>> = if !requested.is_empty() {
        let visible = crate::environment::visibility(requested, inherited)?;
        Some(visible.split(',').map(|s| s.trim().to_owned()).collect())
    } else if let Some(mask) = inherited {
        let trimmed = mask.trim();
        if trimmed.is_empty() || trimmed == "-1" {
            return Ok(GpuProbe {
                gpus: Vec::new(),
                interconnect: Interconnect::NoGpus,
                notes: vec![format!("CUDA_VISIBLE_DEVICES={trimmed:?} exposes no GPUs.")],
            });
        }
        // Validate syntax using the shared visibility helper.
        crate::environment::visibility(&[0], Some(trimmed))?;
        Some(trimmed.split(',').map(|s| s.trim().to_owned()).collect())
    } else {
        None
    };

    // SAFETY: Only the system NVML library is loaded and kept alive for the duration of this call.
    let library = unsafe { libloading::Library::new("libnvidia-ml.so.1") }
        .map_err(|error| format!("cannot load libnvidia-ml.so.1: {error}"))?;

    // SAFETY: Function signatures and PciInfo match nvml.h; all output pointers are valid.
    unsafe {
        let init = library.get::<unsafe extern "C" fn() -> c_int>(b"nvmlInit_v2\0")?;
        let shutdown = library.get::<unsafe extern "C" fn() -> c_int>(b"nvmlShutdown\0")?;
        let code = init();
        if code != 0 {
            return Err(format!("nvmlInit_v2 failed with NVML error {code}").into());
        }
        let result = (|| -> Result<GpuProbe> {
            let get_count = library
                .get::<unsafe extern "C" fn(*mut c_uint) -> c_int>(b"nvmlDeviceGetCount_v2\0")?;
            let get_by_index = library.get::<unsafe extern "C" fn(c_uint, *mut Device) -> c_int>(
                b"nvmlDeviceGetHandleByIndex_v2\0",
            )?;
            let get_by_uuid = library
                .get::<unsafe extern "C" fn(*const c_char, *mut Device) -> c_int>(
                    b"nvmlDeviceGetHandleByUUID\0",
                )?;
            let get_name = library
                .get::<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>(
                    b"nvmlDeviceGetName\0",
                )?;
            let get_uuid = library
                .get::<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>(
                    b"nvmlDeviceGetUUID\0",
                )?;
            let get_pci = library.get::<unsafe extern "C" fn(Device, *mut PciInfo) -> c_int>(
                b"nvmlDeviceGetPciInfo_v3\0",
            )?;
            let get_p2p = library
                .get::<unsafe extern "C" fn(Device, Device, c_uint, *mut c_uint) -> c_int>(
                    b"nvmlDeviceGetP2PStatus\0",
                )
                .ok();

            let mut total: c_uint = 0;
            let code = get_count(&mut total);
            if code != 0 {
                return Err(format!("nvmlDeviceGetCount_v2 failed with NVML error {code}").into());
            }

            let mut handles: Vec<(u32, Device)> = Vec::new();
            if let Some(entries) = explicit_selection {
                // Build UUID lookup table only if a shortened UUID prefix is encountered.
                let mut all_uuids: Option<Vec<(u32, String, Device)>> = None;
                for (visible_idx, entry) in entries.iter().enumerate() {
                    let mut device: Device = std::ptr::null_mut();
                    let nvml_idx = if let Ok(index) = entry.parse::<u32>() {
                        let code = get_by_index(index, &mut device);
                        if code != 0 {
                            return Err(format!(
                                "GPU {index} is unavailable (nvmlDeviceGetHandleByIndex_v2 error {code})"
                            )
                            .into());
                        }
                        index
                    } else if entry.len() == 40 {
                        let c_uuid = CString::new(entry.as_str())?;
                        let code = get_by_uuid(c_uuid.as_ptr(), &mut device);
                        if code != 0 {
                            return Err(format!(
                                "GPU {entry} is unavailable (nvmlDeviceGetHandleByUUID error {code})"
                            )
                            .into());
                        }
                        visible_idx as u32
                    } else {
                        let list = match all_uuids.as_ref() {
                            Some(list) => list,
                            None => {
                                let mut list = Vec::with_capacity(total as usize);
                                for idx in 0..total {
                                    let mut candidate: Device = std::ptr::null_mut();
                                    if get_by_index(idx, &mut candidate) == 0 {
                                        let mut buf = [0 as c_char; 96];
                                        if get_uuid(candidate, buf.as_mut_ptr(), 96) == 0 {
                                            let uuid = CStr::from_ptr(buf.as_ptr())
                                                .to_string_lossy()
                                                .into_owned();
                                            list.push((idx, uuid, candidate));
                                        }
                                    }
                                }
                                all_uuids.insert(list)
                            }
                        };
                        let matches: Vec<_> = list
                            .iter()
                            .filter(|(_, uuid, _)| uuid.starts_with(entry.as_str()))
                            .collect();
                        let [matched] = matches.as_slice() else {
                            return Err(format!(
                                "CUDA_VISIBLE_DEVICES entry {entry:?} does not uniquely identify one GPU"
                            )
                            .into());
                        };
                        device = matched.2;
                        matched.0
                    };
                    handles.push((nvml_idx, device));
                }
            } else {
                for index in 0..total {
                    let mut device: Device = std::ptr::null_mut();
                    let code = get_by_index(index, &mut device);
                    if code != 0 {
                        return Err(format!(
                            "nvmlDeviceGetHandleByIndex_v2({index}) failed with NVML error {code}"
                        )
                        .into());
                    }
                    handles.push((index, device));
                }
            }

            let mut gpus = Vec::with_capacity(handles.len());
            for &(index, device) in &handles {
                let mut name_buf = [0 as c_char; 256];
                let name = if get_name(device, name_buf.as_mut_ptr(), name_buf.len() as c_uint) == 0
                {
                    CStr::from_ptr(name_buf.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("NVIDIA GPU {index}")
                };
                let mut uuid_buf = [0 as c_char; 96];
                let uuid = if get_uuid(device, uuid_buf.as_mut_ptr(), uuid_buf.len() as c_uint) == 0
                {
                    CStr::from_ptr(uuid_buf.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                } else {
                    String::new()
                };
                let mut pci: PciInfo = std::mem::zeroed();
                let code = get_pci(device, &mut pci);
                if code != 0 {
                    return Err(format!(
                        "nvmlDeviceGetPciInfo_v3({index}) failed with NVML error {code}"
                    )
                    .into());
                }
                let raw_bus_id = CStr::from_ptr(pci.bus_id.as_ptr()).to_string_lossy();
                let pci_bus_id = normalize_pci_bdf(&raw_bus_id)
                    .unwrap_or_else(|| raw_bus_id.to_ascii_lowercase());
                gpus.push(InspectedGpu {
                    index,
                    name,
                    pci_bus_id,
                    uuid,
                });
            }

            let interconnect = if handles.is_empty() {
                Interconnect::NoGpus
            } else if handles.len() == 1 {
                Interconnect::SingleGpu
            } else if let Some(get_p2p) = get_p2p {
                let mut nvlink_pairs = 0usize;
                let mut pcie_pairs = 0usize;
                for i in 0..handles.len() {
                    for j in (i + 1)..handles.len() {
                        let mut status: c_uint = 0;
                        let code =
                            get_p2p(handles[i].1, handles[j].1, NVML_P2P_CAPS_INDEX_NVLINK, &mut status);
                        if code == 0 && status == NVML_P2P_STATUS_OK {
                            nvlink_pairs += 1;
                        } else {
                            pcie_pairs += 1;
                        }
                    }
                }
                match (nvlink_pairs > 0, pcie_pairs > 0) {
                    (true, false) => Interconnect::NvLink,
                    (true, true) => Interconnect::Mixed,
                    (false, _) => Interconnect::Pcie,
                }
            } else {
                Interconnect::Pcie
            };

            Ok(GpuProbe {
                gpus,
                interconnect,
                notes: Vec::new(),
            })
        })();
        let _ = shutdown();
        result
    }
}

#[cfg(target_os = "linux")]
fn probe_proc_nvidia_gpus(root: &Path) -> Vec<InspectedGpu> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    let mut gpus = Vec::new();
    for (idx, dir) in dirs.into_iter().enumerate() {
        let Some(raw_bdf) = dir.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(pci_bus_id) = normalize_pci_bdf(raw_bdf) else {
            continue;
        };
        let info = std::fs::read_to_string(dir.join("information")).unwrap_or_default();
        let fields = parse_nvidia_params(&info);
        gpus.push(InspectedGpu {
            index: idx as u32,
            name: fields
                .get("Model")
                .cloned()
                .unwrap_or_else(|| format!("NVIDIA GPU {idx}")),
            pci_bus_id,
            uuid: fields.get("GPU UUID").cloned().unwrap_or_default(),
        });
    }
    gpus
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const HEALTHY_NVIDIA_PARAMS: &str = r#"
ResmanDebugLevel: 4294967295
EnablePCIERelaxedOrderingMode: 1
EnableResizableBar: 1
RegistryDwords: "ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1"
"#;

    #[test]
    fn nvidia_params_parser_and_checks_validate_all_required_fields() {
        let checks = check_nvidia_params(Ok(HEALTHY_NVIDIA_PARAMS));
        assert_eq!(checks.len(), 6);
        assert!(
            checks.iter().all(|c| c.status == CheckStatus::Pass),
            "{checks:#?}"
        );

        // Decimal ForceP2P=17 equals 0x11; RMPcieP2PType=1 (forced BAR1) and 2 (auto) are both
        // accepted; extra whitespace and extra keys (including redundant EnableResizableBar=1 in
        // RegistryDwords) are accepted.
        let variant = r#"
EnablePCIERelaxedOrderingMode: 1
EnableResizableBar: 1
RegistryDwords: "OtherKey=42; ForceP2P = 17 ; RMForceP2PType=0x1; RMPcieP2PType=1; GrdmaPciTopoCheckOverride=1; EnableResizableBar=1;"
"#;
        let checks = check_nvidia_params(Ok(variant));
        assert!(
            checks.iter().all(|c| c.status == CheckStatus::Pass),
            "{checks:#?}"
        );

        // Out-of-range RMPcieP2PType=3 or omitted RMPcieP2PType fails.
        for bad_val in [Some("3"), None] {
            let dwords = match bad_val {
                Some(v) => format!(
                    "ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType={v};GrdmaPciTopoCheckOverride=1"
                ),
                None => "ForceP2P=0x11;RMForceP2PType=1;GrdmaPciTopoCheckOverride=1".to_owned(),
            };
            let text = format!(
                "EnablePCIERelaxedOrderingMode: 1\nEnableResizableBar: 1\nRegistryDwords: \"{dwords}\"\n"
            );
            let checks = check_nvidia_params(Ok(&text));
            let by_id: BTreeMap<&str, &Check> =
                checks.iter().map(|c| (c.id.as_str(), c)).collect();
            assert_eq!(
                by_id["nvidia.registry_dwords.rm_pcie_p2p_type"].status,
                CheckStatus::Fail
            );
        }

        // Missing or wrong values fail individually with expected/actual details.
        let broken = r#"
EnablePCIERelaxedOrderingMode: 0
EnableResizableBar: 0
RegistryDwords: "ForceP2P=0x01;RMForceP2PType=1;RMPcieP2PType=0"
"#;
        let checks = check_nvidia_params(Ok(broken));
        let by_id: BTreeMap<&str, &Check> = checks.iter().map(|c| (c.id.as_str(), c)).collect();
        assert_eq!(
            by_id["nvidia.enable_pcie_relaxed_ordering_mode"].status,
            CheckStatus::Fail
        );
        assert_eq!(
            by_id["nvidia.enable_resizable_bar"].status,
            CheckStatus::Fail
        );
        assert_eq!(
            by_id["nvidia.registry_dwords.force_p2p"].status,
            CheckStatus::Fail
        );
        assert_eq!(
            by_id["nvidia.registry_dwords.rm_force_p2p_type"].status,
            CheckStatus::Pass
        );
        assert_eq!(
            by_id["nvidia.registry_dwords.rm_pcie_p2p_type"].status,
            CheckStatus::Fail
        );
        assert_eq!(
            by_id["nvidia.registry_dwords.grdma_pci_topo_check_override"].status,
            CheckStatus::Fail
        );
        assert!(
            !by_id.contains_key("nvidia.registry_dwords.enable_resizable_bar"),
            "EnableResizableBar is verified via the top-level driver param, not duplicated in RegistryDwords"
        );
        assert!(
            by_id["nvidia.registry_dwords.force_p2p"]
                .remediation
                .as_deref()
                .unwrap()
                .contains("NVreg_RegistryDwords")
        );

        let missing = check_nvidia_params(Err("cannot read /proc/driver/nvidia/params".into()));
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].status, CheckStatus::Fail);
    }

    #[test]
    fn pci_bdf_normalization_and_ancestor_lca_filtering() {
        assert_eq!(
            normalize_pci_bdf("00000000:0F:00.0").as_deref(),
            Some("0000:0f:00.0")
        );
        assert_eq!(
            normalize_pci_bdf("0000:05:00.0").as_deref(),
            Some("0000:05:00.0")
        );
        assert_eq!(normalize_pci_bdf("pci0000:00"), None);
        assert_eq!(normalize_pci_bdf("not-a-bdf"), None);

        let gpu0 = Path::new(
            "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.0/0000:04:00.0/0000:05:00.0",
        );
        let gpu1 = Path::new(
            "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.0/0000:04:10.0/0000:06:00.0",
        );
        let gpu2 = Path::new(
            "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:04.0/0000:07:00.0/0000:08:00.0/0000:09:00.0",
        );
        let gpu4 = Path::new(
            "/sys/devices/pci0000:16/0000:16:01.1/0000:17:00.0/0000:18:00.0/0000:19:00.0/0000:1a:00.0/0000:1b:00.0",
        );

        let c0 = pci_ancestor_bdfs(gpu0);
        let c1 = pci_ancestor_bdfs(gpu1);
        let c2 = pci_ancestor_bdfs(gpu2);
        let c4 = pci_ancestor_bdfs(gpu4);
        assert_eq!(
            c0,
            vec![
                "0000:00:01.1",
                "0000:01:00.0",
                "0000:02:00.0",
                "0000:03:00.0",
                "0000:04:00.0"
            ]
        );

        // GPU 0 and GPU 1 share switch 0000:03:00.0; upstream root port 0000:00:01.1 is excluded.
        assert_eq!(
            relevant_bridges(&[c0.clone(), c1.clone()]),
            vec!["0000:03:00.0", "0000:04:00.0", "0000:04:10.0"]
        );

        // GPUs 0, 1, 2 share top switch 0000:01:00.0; root port 0000:00:01.1 is still excluded.
        let b012 = relevant_bridges(&[c0.clone(), c1, c2]);
        assert!(b012.contains(&"0000:01:00.0".to_owned()));
        assert!(b012.contains(&"0000:04:00.0".to_owned()));
        assert!(b012.contains(&"0000:08:00.0".to_owned()));
        assert!(!b012.contains(&"0000:00:01.1".to_owned()));

        // GPUs 0 and 4 cross root ports: both root ports and all intermediate switches are included.
        let b04 = relevant_bridges(&[c0, c4]);
        assert!(b04.contains(&"0000:00:01.1".to_owned()));
        assert!(b04.contains(&"0000:16:01.1".to_owned()));
        assert!(b04.contains(&"0000:04:00.0".to_owned()));
        assert!(b04.contains(&"0000:1a:00.0".to_owned()));
    }

    fn make_pcie_config_with_acs(acs_offset: usize, acs_cap: u16, acs_ctl: u16) -> Vec<u8> {
        let mut config = vec![0u8; 4096];
        // Place a dummy AER capability (0x0001) at 0x100 pointing to acs_offset.
        if acs_offset > 0x100 {
            let hdr: u32 = 0x0001 | (1 << 16) | ((acs_offset as u32) << 20);
            config[0x100..0x104].copy_from_slice(&hdr.to_le_bytes());
        }
        let acs_hdr: u32 = (PCIE_EXT_CAP_ID_ACS as u32) | (1 << 16);
        config[acs_offset..acs_offset + 4].copy_from_slice(&acs_hdr.to_le_bytes());
        config[acs_offset + 4..acs_offset + 6].copy_from_slice(&acs_cap.to_le_bytes());
        config[acs_offset + 6..acs_offset + 8].copy_from_slice(&acs_ctl.to_le_bytes());
        config
    }

    fn parse_acs_slice(config: &[u8]) -> (AcsConfigRead, Vec<(usize, usize)>) {
        let mut reads = Vec::new();
        let result = parse_acs_capability_with(config.len(), |offset, buf| {
            reads.push((offset, buf.len()));
            let end = offset.checked_add(buf.len()).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "overflow")
            })?;
            if end > config.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "short read",
                ));
            }
            buf.copy_from_slice(&config[offset..end]);
            Ok(())
        });
        (result, reads)
    }

    #[test]
    fn acs_capability_parser_uses_targeted_reads_and_handles_edge_cases() {
        // Unprivileged / legacy config (< 0x104 bytes) performs no extended-space reads.
        let (result, reads) = parse_acs_slice(&[0u8; 64]);
        assert_eq!(result, AcsConfigRead::ExtendedSpaceUnavailable);
        assert!(reads.is_empty());

        // Simulated unprivileged sysfs file (st_size=4096, pread at 0x100 returns EOF).
        let unpriv = parse_acs_capability_with(4096, |offset, _| {
            assert_eq!(offset, 0x100);
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "unprivileged",
            ))
        });
        assert_eq!(unpriv, AcsConfigRead::ExtendedSpaceUnavailable);

        // No extended capabilities (all zero) reads only the single DWORD at 0x100.
        let (result, reads) = parse_acs_slice(&[0u8; 4096]);
        assert_eq!(result, AcsConfigRead::NotPresent);
        assert_eq!(reads, vec![(0x100, 4)]);

        // ACS present at 0x148 and disabled (0x0000): touches only headers at 0x100, 0x148 and ACSCtl at 0x14e.
        let disabled = make_pcie_config_with_acs(0x148, 0x001f, 0x0000);
        let (result, reads) = parse_acs_slice(&disabled);
        let AcsConfigRead::Present(regs) = result else {
            panic!("expected ACS present");
        };
        assert_eq!(regs.offset, 0x148);
        assert_eq!(regs.p2p_redirect_bits(), 0);
        assert_eq!(reads, vec![(0x100, 4), (0x148, 4), (0x14e, 2)]);

        // SrcValid+ alone (bit 0) does not trigger P2P redirection failure (matches pci=disable_acs_redir).
        let sv_only = make_pcie_config_with_acs(0x100, 0x001f, 0x0001);
        let (result, reads) = parse_acs_slice(&sv_only);
        let AcsConfigRead::Present(regs) = result else {
            panic!("expected ACS present");
        };
        assert_eq!(regs.p2p_redirect_bits(), 0);
        assert_eq!(regs.enabled_flags(), vec!["SrcValid+"]);
        assert_eq!(reads, vec![(0x100, 4), (0x106, 2)]);

        // ReqRedir+ CmpltRedir+ UpstreamFwd+ triggers failure.
        let enabled = make_pcie_config_with_acs(0x148, 0x001f, 0x001d);
        let (result, reads) = parse_acs_slice(&enabled);
        let AcsConfigRead::Present(regs) = result else {
            panic!("expected ACS present");
        };
        assert_ne!(regs.p2p_redirect_bits(), 0);
        assert_eq!(
            regs.enabled_flags(),
            vec!["SrcValid+", "ReqRedir+", "CmpltRedir+", "UpstreamFwd+"]
        );
        assert_eq!(reads, vec![(0x100, 4), (0x148, 4), (0x14e, 2)]);

        // Cyclic extended capability list terminates safely without infinite loop.
        let mut cyclic = vec![0u8; 4096];
        let hdr1: u32 = 0x0001 | (1 << 16) | (0x120 << 20);
        let hdr2: u32 = 0x0002 | (1 << 16) | (0x100 << 20);
        cyclic[0x100..0x104].copy_from_slice(&hdr1.to_le_bytes());
        cyclic[0x120..0x124].copy_from_slice(&hdr2.to_le_bytes());
        let (result, reads) = parse_acs_slice(&cyclic);
        assert_eq!(result, AcsConfigRead::NotPresent);
        assert_eq!(reads, vec![(0x100, 4), (0x120, 4)]);
    }

    #[test]
    fn iommu_check_distinguishes_off_identity_dma_and_old_kernels() {
        let dir = tempfile::tempdir().unwrap();
        let sysfs = dir.path();
        let gpu_dir = sysfs.join("bus/pci/devices/0000:05:00.0");
        fs::create_dir_all(&gpu_dir).unwrap();
        fs::create_dir_all(sysfs.join("kernel/iommu_groups")).unwrap();

        // Off: no iommu_group symlink.
        let off = check_iommu(
            sysfs,
            &["0000:05:00.0".into()],
            Some("BOOT_IMAGE=/vmlinuz iommu=off amd_iommu=off"),
        );
        assert_eq!(off.status, CheckStatus::Pass);
        assert_eq!(off.actual.as_deref(), Some("off"));
        assert!(off.detail.contains("iommu=off"));

        // Identity (pt): iommu_group/type == "identity".
        let group_dir = gpu_dir.join("iommu_group");
        fs::create_dir_all(&group_dir).unwrap();
        fs::write(group_dir.join("type"), "identity\n").unwrap();
        let pt = check_iommu(sysfs, &["0000:05:00.0".into()], Some("iommu=pt"));
        assert_eq!(pt.status, CheckStatus::Pass);
        assert_eq!(pt.actual.as_deref(), Some("pt (identity)"));

        // Translating DMA-FQ: must fail.
        fs::write(group_dir.join("type"), "DMA-FQ\n").unwrap();
        let dma = check_iommu(sysfs, &["0000:05:00.0".into()], Some(""));
        assert_eq!(dma.status, CheckStatus::Fail);
        assert_eq!(dma.actual.as_deref(), Some("DMA-FQ"));
        assert!(dma.detail.contains("0000:05:00.0=DMA-FQ"));

        // Older kernel without group/type: falls back to cmdline.
        fs::remove_file(group_dir.join("type")).unwrap();
        let fallback_pt = check_iommu(sysfs, &["0000:05:00.0".into()], Some("iommu=pt"));
        assert_eq!(fallback_pt.status, CheckStatus::Pass);
        let fallback_unknown = check_iommu(sysfs, &["0000:05:00.0".into()], Some("quiet"));
        assert_eq!(fallback_unknown.status, CheckStatus::Unverified);
    }

    #[test]
    fn acs_check_against_synthetic_sysfs_reports_pass_fail_and_unverified() {
        let dir = tempfile::tempdir().unwrap();
        let sysfs = dir.path();
        let b1 = "0000:04:00.0";
        let b2 = "0000:04:10.0";
        for b in [b1, b2] {
            fs::create_dir_all(sysfs.join("bus/pci/devices").join(b)).unwrap();
        }

        // Unprivileged (64-byte config reads) -> Unverified with sudo hint.
        for b in [b1, b2] {
            fs::write(sysfs.join("bus/pci/devices").join(b).join("config"), [0u8; 64]).unwrap();
        }
        let unverified = check_acs(sysfs, &[b1.into(), b2.into()]);
        assert_eq!(unverified.status, CheckStatus::Unverified);
        assert!(unverified.remediation.as_deref().unwrap().contains("sudo"));

        // Disabled across both bridges -> Pass.
        fs::write(
            sysfs.join("bus/pci/devices").join(b1).join("config"),
            make_pcie_config_with_acs(0x148, 0x001f, 0x0000),
        )
        .unwrap();
        fs::write(
            sysfs.join("bus/pci/devices").join(b2).join("config"),
            vec![0u8; 4096],
        )
        .unwrap();
        let pass = check_acs(sysfs, &[b1.into(), b2.into()]);
        assert_eq!(pass.status, CheckStatus::Pass);
        assert!(pass.detail.contains("2 bridge(s)"));

        // Enabled on b2 -> Fail with offending BDF and setpci command.
        fs::write(
            sysfs.join("bus/pci/devices").join(b2).join("config"),
            make_pcie_config_with_acs(0x148, 0x001f, 0x001d),
        )
        .unwrap();
        let fail = check_acs(sysfs, &[b1.into(), b2.into()]);
        assert_eq!(fail.status, CheckStatus::Fail);
        assert!(fail.detail.contains(b2));
        assert!(fail.detail.contains("ReqRedir+"));
        assert!(
            fail.remediation
                .as_deref()
                .unwrap()
                .contains("setpci -s 0000:04:10.0 ECAP_ACS+0x6.w=0000")
        );
    }

    #[test]
    fn nvlink_host_skips_pcie_checks_and_run_basic_checks_skip_acs_and_format_error() {
        let dir = tempfile::tempdir().unwrap();
        let nvlink_probe = GpuProbe {
            gpus: vec![
                InspectedGpu {
                    index: 0,
                    name: "NVIDIA B200".into(),
                    pci_bus_id: "0000:05:00.0".into(),
                    uuid: "GPU-1".into(),
                },
                InspectedGpu {
                    index: 1,
                    name: "NVIDIA B200".into(),
                    pci_bus_id: "0000:06:00.0".into(),
                    uuid: "GPU-2".into(),
                },
            ],
            interconnect: Interconnect::NvLink,
            notes: Vec::new(),
        };
        let report = evaluate_linux_host(
            nvlink_probe,
            dir.path(),
            &dir.path().join("missing-params"),
            &dir.path().join("missing-cmdline"),
        );
        assert!(report.ok);
        assert!(!report.pcie_tp_checks_required);
        assert_eq!(report.checks[0].status, CheckStatus::NotApplicable);

        // Basic `flare run` checks verify NVIDIA params and IOMMU only (no ACS check),
        // returning an error directing the operator to `flare doctor` and `--bypass-performance-check`.
        let params = dir.path().join("params");
        let cmdline = dir.path().join("cmdline");
        fs::write(
            &params,
            "EnablePCIERelaxedOrderingMode: 0\nEnableResizableBar: 1\nRegistryDwords: \"\"\n",
        )
        .unwrap();
        fs::write(&cmdline, "iommu=off\n").unwrap();
        let bdfs = vec!["0000:05:00.0".to_owned(), "0000:06:00.0".to_owned()];
        let failed = evaluate_run_basic_checks(&bdfs, dir.path(), &params, &cmdline);
        assert!(!failed.is_empty());
        assert!(failed.iter().all(|check| check.id != "pcie.acs"));
        let err = format_run_check_error(&failed);
        assert!(err.contains("EnablePCIERelaxedOrderingMode"), "{err}");
        assert!(err.contains("ForceP2P"), "{err}");
        assert!(err.contains("flare doctor"), "{err}");
        assert!(err.contains("--bypass-performance-check"), "{err}");

        // Healthy params + IOMMU pass `evaluate_run_basic_checks` even when ACS config is absent/enabled.
        fs::write(
            &params,
            "EnablePCIERelaxedOrderingMode: 1\nEnableResizableBar: 1\nRegistryDwords: \"ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1\"\n",
        )
        .unwrap();
        let healthy = evaluate_run_basic_checks(&bdfs, dir.path(), &params, &cmdline);
        assert!(healthy.is_empty(), "unexpected failures: {healthy:?}");

        // Full `flare doctor` still reports failures cleanly in plain + JSON.
        fs::write(
            &params,
            "EnablePCIERelaxedOrderingMode: 0\nEnableResizableBar: 1\nRegistryDwords: \"\"\n",
        )
        .unwrap();
        let pcie_probe = GpuProbe {
            gpus: vec![
                InspectedGpu {
                    index: 0,
                    name: "RTX PRO 6000".into(),
                    pci_bus_id: "0000:05:00.0".into(),
                    uuid: "GPU-1".into(),
                },
                InspectedGpu {
                    index: 1,
                    name: "RTX PRO 6000".into(),
                    pci_bus_id: "0000:06:00.0".into(),
                    uuid: "GPU-2".into(),
                },
            ],
            interconnect: Interconnect::Pcie,
            notes: Vec::new(),
        };
        let report = evaluate_linux_host(pcie_probe, dir.path(), &params, &cmdline);
        assert!(!report.ok);
        let plain = render(&report, Canvas::sized(100, false));
        assert!(plain.contains("Flare doctor"), "{plain}");
        assert!(plain.contains("FAIL"), "{plain}");
        assert!(plain.contains("Remediation"), "{plain}");
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["ok"], false);
        assert_eq!(json["interconnect"], "pcie");
    }
}
