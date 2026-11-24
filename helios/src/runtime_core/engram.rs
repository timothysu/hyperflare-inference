/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{
    format::{Memory, Core, invalid},
    protobuf::wire,
};
use crate::{
    Result,
    engram::{Index, Plane, Table},
};
use std::collections::HashSet;

/// Payload ownership follows the table layout; absent partitions retain shared storage.
pub(super) fn spans(
    index: &wire::ModelIndex,
    selected: Option<usize>,
) -> Result<Vec<(Option<usize>, &wire::Span)>> {
    let world = index
        .cluster
        .as_ref()
        .map_or(index.ranks.len(), |c| c.world as usize);
    let mut result = Vec::new();
    for table in index.engram.iter().flat_map(|e| &e.tables) {
        validate_shards(table, world)?;
        if table.shards.is_empty() {
            for span in [&table.weights, &table.scales] {
                result.push((
                    None,
                    span.as_ref()
                        .ok_or_else(|| invalid("missing table plane"))?,
                ));
            }
        } else {
            for shard in &table.shards {
                if selected.is_none_or(|rank| rank == shard.rank as usize) {
                    for span in [&shard.weights, &shard.scales] {
                        result.push((
                            Some(shard.rank as usize),
                            span.as_ref()
                                .ok_or_else(|| invalid("missing table partition plane"))?,
                        ));
                    }
                }
            }
        }
    }
    Ok(result)
}

fn validate_shards(table: &wire::EngramTable, world: usize) -> Result<()> {
    if table.shards.is_empty() {
        return Ok(());
    }
    if table.weights.is_some()
        || table.scales.is_some()
        || table.request_buffer.is_empty()
        || table.shards.len() != world
    {
        return Err(invalid("invalid partitioned embedding storage"));
    }
    let mut end = 0;
    for (rank, shard) in table.shards.iter().enumerate() {
        if shard.rank as usize != rank || shard.first_row != end || shard.rows == 0 {
            return Err(invalid(
                "embedding partitions must cover table rows in rank order",
            ));
        }
        end = end
            .checked_add(shard.rows)
            .ok_or_else(|| invalid("embedding partition overflow"))?;
        // Sibling partitions absent from a rank-selective copy carry neither plane; the
        // owning rank still requires both (decode and `spans` reject a missing plane).
        if shard.weights.is_none() != shard.scales.is_none() {
            return Err(invalid("missing embedding partition plane"));
        }
    }
    if end != table.rows {
        return Err(invalid("embedding partitions must cover all table rows"));
    }
    Ok(())
}

pub(super) fn decode<F: super::payload::Files>(
    value: wire::Engram,
    archive: &F,
    cores: &[Core],
    rank: usize,
    world: usize,
) -> Result<Index> {
    if value.compressed_vocabulary == 0
        || value
            .token_map
            .iter()
            .any(|n| *n >= value.compressed_vocabulary)
        || value.pad_token_id
            >= if value.token_map.is_empty() {
                value.compressed_vocabulary
            } else {
                value.token_map.len() as u32
            }
        || value.tables.is_empty()
    {
        return Err(invalid("invalid Engram vocabulary or table catalog"));
    }
    let mut layers = HashSet::new();
    let mut buffers = HashSet::new();
    let mut events = HashSet::new();
    let mut tables = Vec::new();
    for table in value.tables {
        validate_shards(&table, world)?;
        let gpu = !table.request_buffer.is_empty();
        let encoding = wire::engram_table::Encoding::try_from(table.encoding)
            .map_err(|_| invalid("unknown n-gram encoding"))?;
        let (block, row_bytes, scale_bytes) = match encoding {
            wire::engram_table::Encoding::Fp8E4m3E8m0
                if table.scale_block == 32
                    && table.embedding_width > 0
                    && table.embedding_width.is_multiple_of(32) =>
            {
                let width = table.embedding_width as usize;
                (32u32, width, width / 32)
            }
            wire::engram_table::Encoding::Nvfp4E4m3Global
                if table.global_scale.is_finite()
                    && table.global_scale > 0.0
                    && table.scale_block == 16
                    && table.embedding_width > 0
                    && table.embedding_width.is_multiple_of(16) =>
            {
                let width = table.embedding_width as usize;
                (16u32, width / 2, width / 16)
            }
            _ => return Err(invalid("invalid n-gram encoding, width, scale block, or global scale")),
        };
        let geometry = table
            .hash
            .as_ref()
            .ok_or_else(|| invalid("n-gram table lacks hash geometry"))?;
        let order = match wire::engram_hash::Method::try_from(geometry.method) {
            Ok(wire::engram_hash::Method::XorModulo)
                if !value.token_map.is_empty()
                    && matches!(geometry.multipliers.len(), 3 | 4) =>
            {
                geometry.multipliers.len()
            }
            Ok(wire::engram_hash::Method::EosXorModulo) if gpu && value.token_map.is_empty() => 3,
            _ => return Err(invalid("unsupported n-gram hash method or request source")),
        };
        let heads = (order - 1) * 8;
        if !layers.insert(table.layer)
            || table.rows == 0
            || table.scale_block != block
            || !events.insert(table.ready_event.clone())
            || table.ready_event.is_empty()
            || [
                &table.weight_buffer,
                &table.scale_buffer,
                &table.hash_buffer,
            ]
            .iter()
            .any(|name| name.is_empty() || !buffers.insert((*name).clone()))
        {
            return Err(invalid("invalid Engram table encoding or bindings"));
        }
        if gpu && !buffers.insert(table.request_buffer.clone()) {
            return Err(invalid("n-gram mailbox aliases another binding"));
        }
        let geometry = table
            .hash
            .ok_or_else(|| invalid("Engram table lacks hash geometry"))?;
        if geometry.orders != (2..=order as u32).flat_map(|n| [n; 8]).collect::<Vec<_>>()
            || geometry.multipliers.len() != order
            || geometry.primes.len() != heads
            || geometry.offsets.len() != heads
            || geometry.multipliers.iter().any(|m| {
                *m == 0
                    || m.checked_mul(value.compressed_vocabulary as u64)
                        .is_none_or(|n| n > i64::MAX as u64)
            })
            || geometry.primes.iter().zip(&geometry.offsets).any(|(p, o)| {
                *p < 2
                    || o.checked_add(*p)
                        .is_none_or(|n| n > table.rows || n > i64::MAX as u64)
            })
        {
            return Err(invalid("invalid Engram hash geometry"));
        }
        let (stored_rows, weights, scales) = if table.shards.is_empty() {
            (table.rows, table.weights.clone(), table.scales.clone())
        } else {
            let shard = table
                .shards
                .get(rank)
                .ok_or_else(|| invalid("embedding partition rank is absent"))?;
            (shard.rows, shard.weights.clone(), shard.scales.clone())
        };
        let plane = |span: Option<wire::Span>, row_bytes: usize| -> Result<Plane> {
            let span = span.ok_or_else(|| invalid("Engram table lacks a payload plane"))?;
            if stored_rows.checked_mul(row_bytes as u64) != Some(span.nbytes) {
                return Err(invalid("Engram payload geometry mismatch"));
            }
            Ok(Plane {
                file: archive.source_index(&span)?,
                offset: archive.source(&span)?.1,
                nbytes: usize::try_from(span.nbytes)
                    .map_err(|_| invalid("Engram plane exceeds host range"))?,
                row_bytes,
            })
        };
        let planes = [plane(weights, row_bytes)?, plane(scales, scale_bytes)?];
        for core in cores {
            if !core.events.contains(&table.ready_event) {
                return Err(invalid("rank lacks Engram readiness event"));
            }
            let execution = core
                .execution
                .as_ref()
                .ok_or_else(|| invalid("Engram rank lacks execution bindings"))?;
            let is_consumer = |kind: &str| {
                crate::execution::GraphKind::try_from(kind)
                    .is_ok_and(crate::engram::consumes)
            };
            let tokens = execution
                .graphs
                .iter()
                .filter(|g| is_consumer(g.kind.as_str()))
                .map(|g| g.packed_tokens())
                .max()
                .ok_or_else(|| invalid("Engram catalog lacks consumers"))?;
            for (name, dtype, columns) in [
                (&table.weight_buffer, "uint8", row_bytes as u64),
                (&table.scale_buffer, "uint8", scale_bytes as u64),
                (&table.hash_buffer, "int64", heads as u64),
            ] {
                let buffer = execution
                    .buffers
                    .iter()
                    .find(|b| &b.name == name)
                    .ok_or_else(|| invalid("rank lacks Engram input buffer"))?;
                let rows = if dtype == "uint8" {
                    tokens * heads as u64
                } else {
                    tokens
                };
                if buffer.dtype != dtype
                    || buffer.shape.len() != 2
                    || buffer.shape[0] < rows
                    || buffer.shape[1] != columns
                    || buffer.strides != [columns, 1]
                    || buffer.nbytes
                        != buffer.shape[0]
                            .checked_mul(columns * if dtype == "int64" { 8 } else { 1 })
                            .unwrap_or(0)
                {
                    return Err(invalid("invalid compact Engram buffer layout"));
                }
            }
            if gpu {
                let request = execution
                    .buffers
                    .iter()
                    .find(|b| b.name == table.request_buffer)
                    .ok_or_else(|| invalid("rank lacks n-gram request mailbox"))?;
                let reference = request
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("mailbox lacks storage"))?;
                if request.dtype != "int64"
                    || request.shape.len() != 1
                    || request.strides != [1]
                    || request.shape[0] < tokens * heads as u64 + 1
                    || request.nbytes != request.shape[0] * 8
                    || !reference.offset.is_multiple_of(8)
                    || !core.allocations.iter().any(|a| {
                        a.name == reference.allocation && matches!(a.memory, Memory::MappedHost)
                    })
                {
                    return Err(invalid("invalid mapped n-gram mailbox"));
                }
            }
            for graph in execution
                .graphs
                .iter()
                .filter(|g| is_consumer(g.kind.as_str()))
            {
                if !gpu
                    && !graph.input_bindings.iter().any(|b| {
                        b.buffer_name == table.hash_buffer
                            && b.nbytes >= graph.packed_tokens() * heads as u64 * 8
                    })
                {
                    return Err(invalid("Engram graph lacks hash input binding"));
                }
            }
        }
        tables.push(Table {
            planes,
            heads: geometry.primes.len(),
            weights: table.weight_buffer,
            scales: table.scale_buffer,
            hashes: table.hash_buffer,
            event: table.ready_event,
            request: gpu.then_some(table.request_buffer),
        });
    }
    Ok(Index { tables })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn partitioned() -> wire::EngramTable {
        wire::EngramTable {
            rows: 40,
            request_buffer: "requests".into(),
            shards: (0..4)
                .map(|rank| wire::EngramTableShard {
                    rank,
                    first_row: rank as u64 * 10,
                    rows: 10,
                    weights: Some(wire::Span {
                        nbytes: 2560,
                        ..Default::default()
                    }),
                    scales: Some(wire::Span {
                        nbytes: 80,
                        ..Default::default()
                    }),
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn table_partitions_cover_rows_exactly_once() {
        let table = partitioned();
        assert!(validate_shards(&table, 4).is_ok());
        assert!(validate_shards(&table, 2).is_err());
        for change in 0..7 {
            let mut invalid = table.clone();
            match change {
                0 => invalid.shards[1].first_row += 1,
                1 => invalid.shards[1].first_row -= 1,
                2 => invalid.shards[2].rank = 0,
                3 => invalid.shards[3].rows = 0,
                4 => invalid.weights = Some(wire::Span::default()),
                5 => invalid.shards[1].scales = None,
                _ => invalid.request_buffer.clear(),
            }
            assert!(validate_shards(&invalid, 4).is_err(), "mutation {change}");
        }
    }

    #[test]
    fn rank_selected_partitions_omit_sibling_planes_but_keep_their_rows() {
        let mut table = partitioned();
        for rank in [0, 2, 3] {
            table.shards[rank].weights = None;
            table.shards[rank].scales = None;
        }
        assert!(validate_shards(&table, 4).is_ok());
        let mut one_plane = table.clone();
        one_plane.shards[0].weights = Some(wire::Span::default());
        assert!(validate_shards(&one_plane, 4).is_err());
        let mut gap = table;
        gap.shards[2].first_row += 1;
        assert!(validate_shards(&gap, 4).is_err());
    }

    #[test]
    fn metadata_without_partitions_retains_shared_planes() {
        // EngramTable encoded before field 15 existed: rows=40, weight/scales spans.
        let bytes = [
            0x10, 0x28, 0x2a, 0x04, 0x08, 0x00, 0x10, 0x20, 0x32, 0x04, 0x08, 0x20, 0x10, 0x08,
        ];
        let table = wire::EngramTable::decode(bytes.as_slice()).unwrap();
        assert!(table.shards.is_empty());
        assert!(validate_shards(&table, 4).is_ok());
        let index = wire::ModelIndex {
            ranks: vec![wire::BundleIndex::default(); 4],
            engram: Some(wire::Engram {
                tables: vec![table],
                ..Default::default()
            }),
            ..Default::default()
        };
        for rank in 0..4 {
            let planes = spans(&index, Some(rank)).unwrap();
            assert_eq!(planes.len(), 2);
            assert!(planes.iter().all(|(owner, _)| owner.is_none()));
            assert_eq!((planes[0].1.offset, planes[0].1.nbytes), (0, 32));
            assert_eq!((planes[1].1.offset, planes[1].1.nbytes), (32, 8));
        }
    }

    #[test]
    fn engram_accepts_generalized_widths_and_order3_or_order4_hashing() {
        use super::super::{
            archive::PayloadFile,
            format::{Allocation, Context},
            payload::{Files, RecordReader},
        };
        use std::collections::HashMap;
        use std::sync::Arc;

        struct StubFiles {
            file: PayloadFile,
            path: std::path::PathBuf,
        }
        impl Files for StubFiles {
            fn span<'a>(&self, _: &'a Core, _: &str) -> Result<&'a wire::Span> {
                unreachable!()
            }
            fn source(&self, span: &wire::Span) -> Result<(&PayloadFile, u64)> {
                Ok((&self.file, span.offset))
            }
            fn read(&self, _: &Core, _: &str) -> Result<Vec<u8>> {
                unreachable!()
            }
            fn records<'a>(&'a self, _: &Core, _: &str) -> Result<RecordReader<'a>> {
                unreachable!()
            }
            fn path(&self) -> &std::path::Path {
                &self.path
            }
            fn source_index(&self, _: &wire::Span) -> Result<usize> {
                Ok(0)
            }
            fn asset_path(&self, _: &wire::Span) -> Result<std::path::PathBuf> {
                unreachable!()
            }
            fn release_page_cache(&self) {}
            fn retained_files(&self) -> Result<Vec<Arc<std::fs::File>>> {
                Ok(vec![])
            }
            fn direct_files(
                &self,
            ) -> Box<dyn Iterator<Item = Result<Arc<std::fs::File>>> + '_> {
                Box::new(std::iter::empty())
            }
            fn validate(&self, _: &Core) -> Result<()> {
                Ok(())
            }
            fn resource_reader<'a>(
                &'a self,
                _: &Core,
                _: &str,
            ) -> Result<Box<dyn std::io::Read + 'a>> {
                unreachable!()
            }
        }

        let path = std::env::temp_dir().join(format!("flare-engram-stub-{}", std::process::id()));
        std::fs::write(&path, [0u8; 16]).unwrap();
        let files = StubFiles {
            file: PayloadFile::new(Arc::new(std::fs::File::open(&path).unwrap()), 0, 16),
            path: path.clone(),
        };

        let make_core = |heads: u64, row_bytes: u64, scale_bytes: u64, with_mailbox: bool| {
            let mut buffers = vec![
                wire::Buffer {
                    name: "weights".into(),
                    dtype: "uint8".into(),
                    shape: vec![heads, row_bytes],
                    strides: vec![row_bytes, 1],
                    nbytes: heads * row_bytes,
                    reference: None,
                },
                wire::Buffer {
                    name: "scales".into(),
                    dtype: "uint8".into(),
                    shape: vec![heads, scale_bytes],
                    strides: vec![scale_bytes, 1],
                    nbytes: heads * scale_bytes,
                    reference: None,
                },
                wire::Buffer {
                    name: "hashes".into(),
                    dtype: "int64".into(),
                    shape: vec![1, heads],
                    strides: vec![heads, 1],
                    nbytes: heads * 8,
                    reference: None,
                },
            ];
            let mut allocations = vec![];
            let mut input_bindings = vec![];
            if with_mailbox {
                allocations.push(Allocation {
                    name: "mailbox_alloc".into(),
                    sizing: None,
                    nbytes: (heads as usize + 1) * 8,
                    alignment: 64,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: None,
                    memory: Memory::MappedHost,
                });
                buffers.push(wire::Buffer {
                    name: "requests".into(),
                    dtype: "int64".into(),
                    shape: vec![heads + 1],
                    strides: vec![1],
                    nbytes: (heads + 1) * 8,
                    reference: Some(wire::Reference {
                        allocation: "mailbox_alloc".into(),
                        offset: 0,
                    }),
                });
            } else {
                input_bindings.push(wire::InputBinding {
                    buffer_name: "hashes".into(),
                    host_offset: 0,
                    nbytes: heads * 8,
                });
            }
            Core {
                resources: HashMap::new(),
                architecture: [12, 0],
                allocations,
                modules: vec![],
                functions: vec![],
                graphs: vec![],
                reservations: vec![],
                context: Context::default(),
                events: vec!["engram_ready".into()],
                execution: Some(wire::Execution {
                    graphs: vec![wire::ExecutionGraph {
                        name: "decode1".into(),
                        kind: "decode".into(),
                        batch_capacity: 1,
                        token_capacity: 1,
                        input_bindings,
                        ..Default::default()
                    }],
                    buffers,
                    ..Default::default()
                }),
            }
        };

        let make_engram = |encoding: wire::engram_table::Encoding,
                           width: u32,
                           scale_block: u32,
                           global_scale: f32,
                           order: usize,
                           with_mailbox: bool| {
            let heads = (order - 1) * 8;
            let (row_bytes, scale_bytes) = match encoding {
                wire::engram_table::Encoding::Fp8E4m3E8m0 => (width as u64, (width / scale_block) as u64),
                _ => ((width / 2) as u64, (width / scale_block) as u64),
            };
            wire::Engram {
                token_map: vec![0, 1, 2],
                pad_token_id: 2,
                compressed_vocabulary: 3,
                tables: vec![wire::EngramTable {
                    layer: 1,
                    rows: 500,
                    embedding_width: width,
                    scale_block,
                    encoding: encoding as i32,
                    global_scale,
                    hash: Some(wire::EngramHash {
                        method: wire::engram_hash::Method::XorModulo as i32,
                        orders: (2..=order as u32).flat_map(|n| [n; 8]).collect(),
                        multipliers: vec![3; order],
                        primes: vec![17; heads],
                        offsets: (0..heads as u64).map(|i| i * 17).collect(),
                    }),
                    weights: Some(wire::Span {
                        filename: "w.bin".into(),
                        offset: 0,
                        nbytes: 500 * row_bytes,
                    }),
                    scales: Some(wire::Span {
                        filename: "s.bin".into(),
                        offset: 0,
                        nbytes: 500 * scale_bytes,
                    }),
                    weight_buffer: "weights".into(),
                    scale_buffer: "scales".into(),
                    hash_buffer: "hashes".into(),
                    ready_event: "engram_ready".into(),
                    request_buffer: if with_mailbox { "requests".into() } else { String::new() },
                    ..Default::default()
                }],
            }
        };

        // Generalized host-hash geometry: FP8_E4M3_E8M0 width 96, block 32 (96B + 3B), order-3 (16 heads).
        let idx = decode(
            make_engram(wire::engram_table::Encoding::Fp8E4m3E8m0, 96, 32, 0.0, 3, false),
            &files,
            &[make_core(16, 96, 3, false)],
            0,
            1,
        )
        .unwrap();
        assert_eq!(idx.tables.len(), 1);
        assert_eq!(idx.tables[0].heads, 16);
        assert_eq!(idx.tables[0].planes[0].row_bytes, 96);
        assert_eq!(idx.tables[0].planes[1].row_bytes, 3);
        assert!(idx.tables[0].request.is_none());

        // DeepSeek host-hash geometry: FP8_E4M3_E8M0 width 256, block 32 (256B + 8B), order-4 (24 heads).
        let ds_idx = decode(
            make_engram(wire::engram_table::Encoding::Fp8E4m3E8m0, 256, 32, 0.0, 4, false),
            &files,
            &[make_core(24, 256, 8, false)],
            0,
            1,
        )
        .unwrap();
        assert_eq!(ds_idx.tables[0].heads, 24);
        assert_eq!(ds_idx.tables[0].planes[0].row_bytes, 256);
        assert_eq!(ds_idx.tables[0].planes[1].row_bytes, 8);

        // Order-3 (16 heads) width-96 GPU-mailbox geometry for both FP8_E4M3_E8M0 and NVFP4_E4M3_GLOBAL.
        let fp8_mb = decode(
            make_engram(wire::engram_table::Encoding::Fp8E4m3E8m0, 96, 32, 0.0, 3, true),
            &files,
            &[make_core(16, 96, 3, true)],
            0,
            1,
        )
        .unwrap();
        assert_eq!(fp8_mb.tables[0].heads, 16);
        assert_eq!(fp8_mb.tables[0].planes[0].row_bytes, 96);
        assert_eq!(fp8_mb.tables[0].planes[1].row_bytes, 3);
        assert_eq!(fp8_mb.tables[0].request.as_deref(), Some("requests"));

        let fp4_mb = decode(
            make_engram(wire::engram_table::Encoding::Nvfp4E4m3Global, 96, 16, 1.5, 3, true),
            &files,
            &[make_core(16, 48, 6, true)],
            0,
            1,
        )
        .unwrap();
        assert_eq!(fp4_mb.tables[0].heads, 16);
        assert_eq!(fp4_mb.tables[0].planes[0].row_bytes, 48);
        assert_eq!(fp4_mb.tables[0].planes[1].row_bytes, 6);
        assert_eq!(fp4_mb.tables[0].request.as_deref(), Some("requests"));

        // Width not divisible by scale_block 32 is rejected with updated diagnostic text.
        let bad_width = decode(
            make_engram(wire::engram_table::Encoding::Fp8E4m3E8m0, 80, 32, 0.0, 3, false),
            &files,
            &[make_core(16, 80, 2, false)],
            0,
            1,
        )
        .unwrap_err();
        assert!(bad_width.to_string().contains("width"));

        // Unsupported order 2 is rejected.
        assert!(decode(
            make_engram(wire::engram_table::Encoding::Fp8E4m3E8m0, 96, 32, 0.0, 2, false),
            &files,
            &[make_core(8, 96, 3, false)],
            0,
            1,
        )
        .is_err());

        // ScoredPrefill is counted as an Engram consumer for both compact buffer sizing
        // and per-graph host-hash binding validation (matching crate::engram::consumes).
        let mut core_under = make_core(16, 96, 3, false);
        core_under.execution.as_mut().unwrap().graphs.push(wire::ExecutionGraph {
            name: "scored4".into(),
            kind: "scored_prefill".into(),
            batch_capacity: 1,
            token_capacity: 4,
            input_bindings: vec![wire::InputBinding {
                buffer_name: "hashes".into(),
                host_offset: 0,
                nbytes: 4 * 16 * 8,
            }],
            ..Default::default()
        });
        assert!(decode(
            make_engram(wire::engram_table::Encoding::Fp8E4m3E8m0, 96, 32, 0.0, 3, false),
            &files,
            &[core_under],
            0,
            1,
        )
        .is_err());

        // Sizing compact buffers to match scored_prefill succeeds, while omitting
        // scored_prefill's host hash binding is rejected.
        let make_scored_core = |with_binding: bool| {
            let mut core = make_core(16, 96, 3, false);
            for buf in &mut core.execution.as_mut().unwrap().buffers {
                if buf.name == "weights" {
                    buf.shape = vec![4 * 16, 96];
                    buf.nbytes = 4 * 16 * 96;
                } else if buf.name == "scales" {
                    buf.shape = vec![4 * 16, 3];
                    buf.nbytes = 4 * 16 * 3;
                } else if buf.name == "hashes" {
                    buf.shape = vec![4, 16];
                    buf.nbytes = 4 * 16 * 8;
                }
            }
            core.execution.as_mut().unwrap().graphs.push(wire::ExecutionGraph {
                name: "scored4".into(),
                kind: "scored_prefill".into(),
                batch_capacity: 1,
                token_capacity: 4,
                input_bindings: if with_binding {
                    vec![wire::InputBinding {
                        buffer_name: "hashes".into(),
                        host_offset: 0,
                        nbytes: 4 * 16 * 8,
                    }]
                } else {
                    vec![]
                },
                ..Default::default()
            });
            core
        };
        assert!(decode(
            make_engram(wire::engram_table::Encoding::Fp8E4m3E8m0, 96, 32, 0.0, 3, false),
            &files,
            &[make_scored_core(true)],
            0,
            1,
        )
        .is_ok());

        let missing_err = decode(
            make_engram(wire::engram_table::Encoding::Fp8E4m3E8m0, 96, 32, 0.0, 3, false),
            &files,
            &[make_scored_core(false)],
            0,
            1,
        )
        .unwrap_err();
        assert!(missing_err.to_string().contains("hash input binding"));

        std::fs::remove_file(&path).unwrap();
    }
}
