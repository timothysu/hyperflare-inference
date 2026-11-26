/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! In-graph asynchronous conditional grammar mask exchange for learned speculative decoding.
//!
//! Each wide `Speculative` graph publishes its proposed draft token IDs and submission
//! sequence into a mapped-host mailbox immediately after drafting and waits on an external
//! CUDA readiness event before target verification filtering. When a submission contains
//! one or more grammar-constrained lanes, a persistent host worker computes each lane's
//! conditional masks (`M_0 .. M_k`) and verification bounds in parallel with the target
//! forward, stages them in pinned host memory, and releases a generation-gated transfer
//! stream that uploads only the active constrained slots before recording the readiness
//! event. Unconstrained submissions record the readiness event directly without waiting on
//! the worker or issuing DMA transfers.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

use crate::{
    Error, Result,
    cuda::{Event, PinnedHost, Stream},
    execution::{Dtype, ExecutionManifest, GraphKind},
};

pub const EVENT_NAME: &str = "grammar_ready";
pub const MAILBOX_BUFFER: &str = "grammar_mailbox";
pub const MASKS_BUFFER: &str = "grammar_exchange_masks";
pub const MASK_COUNT_BUFFER: &str = "grammar_exchange_mask_count";
pub const VERIFY_EXTENT_BUFFER: &str = "grammar_exchange_verify_extent";
pub const EMIT_CAP_BUFFER: &str = "grammar_exchange_emit_cap";

const STAGING_FRAMES: usize = 2;

/// Wide speculative graphs publish draft IDs and wait on the grammar readiness event.
pub(crate) fn consumes(kind: GraphKind, token_capacity: usize) -> bool {
    kind == GraphKind::Speculative && token_capacity > 1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub slots: usize,
    pub max_width: usize,
    pub mask_words: usize,
}

impl Geometry {
    pub fn validate(&self) -> Result<()> {
        if self.slots == 0 || self.max_width < 2 || self.mask_words == 0 {
            return Err("invalid grammar exchange geometry".into());
        }
        self.slot_mask_bytes()?;
        self.total_mask_bytes()?;
        self.meta_bytes()?;
        Ok(())
    }

    pub fn max_drafts(&self) -> usize {
        self.max_width - 1
    }

    pub fn row_mask_bytes(&self) -> Result<usize> {
        self.mask_words
            .checked_mul(size_of::<i32>())
            .ok_or_else(|| "grammar exchange mask row overflows host size".into())
    }

    pub fn slot_mask_bytes(&self) -> Result<usize> {
        self.max_width
            .checked_mul(self.row_mask_bytes()?)
            .ok_or_else(|| "grammar exchange slot mask slice overflows host size".into())
    }

    pub fn total_mask_bytes(&self) -> Result<usize> {
        self.slots
            .checked_mul(self.slot_mask_bytes()?)
            .ok_or_else(|| "grammar exchange mask staging overflows host size".into())
    }

    pub fn meta_bytes(&self) -> Result<usize> {
        self.slots
            .checked_mul(size_of::<i32>())
            .ok_or_else(|| "grammar exchange metadata overflows host size".into())
    }
}

/// Inspect a manifest for the optional grammar-exchange buffers and validate their geometry.
pub(crate) fn detect_geometry(
    manifest: &ExecutionManifest,
    vocabulary: usize,
) -> Result<Option<Geometry>> {
    let expected_words = vocabulary.div_ceil(32);
    let has_any = [
        MAILBOX_BUFFER,
        MASKS_BUFFER,
        MASK_COUNT_BUFFER,
        VERIFY_EXTENT_BUFFER,
        EMIT_CAP_BUFFER,
    ]
    .iter()
    .any(|name| manifest.buffers.iter().any(|b| b.name == *name));
    if !has_any {
        return Ok(None);
    }
    let masks = manifest.buffer(MASKS_BUFFER)?;
    let mailbox = manifest.buffer(MAILBOX_BUFFER)?;
    let mask_count = manifest.buffer(MASK_COUNT_BUFFER)?;
    let verify_extent = manifest.buffer(VERIFY_EXTENT_BUFFER)?;
    let emit_cap = manifest.buffer(EMIT_CAP_BUFFER)?;
    if masks.dtype != Dtype::Int32 || masks.shape.len() != 3 {
        return Err("grammar_exchange_masks must be a contiguous 3D int32 buffer".into());
    }
    let slots = masks.shape[0];
    let max_width = masks.shape[1];
    let mask_words = masks.shape[2];
    if slots != manifest.request_capacity()
        || max_width < 2
        || mask_words != expected_words
        || masks.strides != [max_width * mask_words, mask_words, 1]
    {
        return Err("grammar_exchange_masks geometry disagrees with model contract".into());
    }
    let max_spec_width = manifest
        .graphs
        .iter()
        .filter(|g| consumes(g.kind, g.token_capacity))
        .map(|g| g.token_capacity)
        .max()
        .unwrap_or(0);
    if max_spec_width < 2 || max_width < max_spec_width {
        return Err("grammar_exchange_masks width does not cover speculative graphs".into());
    }
    if manifest
        .graphs
        .iter()
        .any(|g| g.kind == GraphKind::Mixed && g.token_capacity > 1)
    {
        return Err("grammar exchange does not yet support wide mixed graphs".into());
    }
    let expected_mailbox = slots
        .checked_mul(max_width - 1)
        .and_then(|n| n.checked_add(1))
        .ok_or("grammar mailbox size overflow")?;
    if mailbox.dtype != Dtype::Int64
        || mailbox.shape != [expected_mailbox]
        || mailbox.strides != [1]
    {
        return Err("grammar_mailbox must be int64[slots * (max_width - 1) + 1]".into());
    }
    for (name, buf) in [
        (MASK_COUNT_BUFFER, mask_count),
        (VERIFY_EXTENT_BUFFER, verify_extent),
        (EMIT_CAP_BUFFER, emit_cap),
    ] {
        if buf.dtype != Dtype::Int32 || buf.shape != [slots] || buf.strides != [1] {
            return Err(format!("{name} must be int32[{slots}]").into());
        }
    }
    let geometry = Geometry {
        slots,
        max_width,
        mask_words,
    };
    geometry.validate()?;
    Ok(Some(geometry))
}

#[cfg(feature = "grammar")]
pub struct LaneJob {
    pub slot: usize,
    pub draft_count: usize,
    pub grammar: crate::grammar::RequestGrammar,
}

#[cfg(not(feature = "grammar"))]
pub struct LaneJob {
    pub slot: usize,
    pub draft_count: usize,
}

struct Staging {
    masks: PinnedHost,
    /// Packed as `[mask_count; slots]`, `[verify_extent; slots]`, `[emit_cap; slots]`.
    meta: PinnedHost,
    ready: PinnedHost,
    error: Mutex<Option<Error>>,
}

impl Staging {
    fn signal(&self, generation: u32) {
        // SAFETY: `ready` is a 4-byte aligned mapped host allocation live until Shared drops.
        unsafe { &*(self.ready.address() as *const AtomicU32) }
            .store(generation, Ordering::Release);
    }
}

#[derive(Clone, Copy, Default)]
struct State {
    sequence: u64,
    generation: u32,
    submitted: usize,
    completed: usize,
}

impl State {
    fn prepare(&mut self, rank: usize, sequence: u64, ranks: usize) -> Result<bool> {
        if sequence == 0 || rank >= ranks {
            return Err("invalid grammar exchange submission identity".into());
        }
        let fresh = self.sequence != sequence;
        if fresh {
            if self.sequence > sequence
                || (self.sequence != 0 && self.completed != (1usize << ranks) - 1)
            {
                return Err("grammar exchange staging reused before all ranks completed".into());
            }
            *self = Self {
                sequence,
                generation: self.generation.wrapping_add(1),
                submitted: 0,
                completed: 0,
            };
        }
        if self.submitted & (1usize << rank) != 0 {
            return Err("rank submitted grammar exchange generation twice".into());
        }
        self.submitted |= 1usize << rank;
        Ok(fresh)
    }

    fn finish(&mut self, rank: usize, sequence: u64) -> Result<bool> {
        if self.sequence != sequence {
            return Ok(false);
        }
        if self.submitted & (1usize << rank) == 0 {
            return Err("grammar exchange completion does not match its submission".into());
        }
        if self.completed & (1usize << rank) != 0 {
            return Err("rank completed grammar exchange generation twice".into());
        }
        self.completed |= 1usize << rank;
        Ok(true)
    }
}

struct StagedSubmission {
    lanes: Option<Vec<LaneJob>>,
    active_slots: Vec<(usize, usize)>,
    remaining_ranks: usize,
}

struct WorkerJob {
    frame: usize,
    generation: u32,
    sequence: u64,
    mailbox_addr: usize,
    mailbox_flag: usize,
    lanes: Vec<LaneJob>,
}

pub(crate) struct Shared {
    geometry: Geometry,
    ranks: usize,
    staging: Arc<[Staging; STAGING_FRAMES]>,
    state: Mutex<[State; STAGING_FRAMES]>,
    staged: Mutex<HashMap<u64, StagedSubmission>>,
    sender: Option<mpsc::SyncSender<WorkerJob>>,
    worker: Option<JoinHandle<()>>,
    cancelled: Arc<AtomicBool>,
}

impl Drop for Shared {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Shared {
    pub fn new(geometry: Geometry, ranks: usize, context: usize, device: i32) -> Result<Arc<Self>> {
        geometry.validate()?;
        if ranks == 0 || ranks > 16 {
            return Err("invalid grammar exchange rank count".into());
        }
        let stream = Stream::open(context, device)?;
        let mask_bytes = geometry.total_mask_bytes()?;
        let meta_bytes = geometry
            .meta_bytes()?
            .checked_mul(3)
            .ok_or("grammar exchange metadata staging overflow")?;
        let mut staging_vec = Vec::with_capacity(STAGING_FRAMES);
        for _ in 0..STAGING_FRAMES {
            let mut masks = stream.mapped_host_buffer(mask_bytes)?;
            masks.bytes_mut().fill(0);
            let mut meta = stream.mapped_host_buffer(meta_bytes)?;
            meta.bytes_mut().fill(0);
            let mut ready = stream.mapped_host_buffer(4)?;
            ready.bytes_mut().fill(0);
            staging_vec.push(Staging {
                masks,
                meta,
                ready,
                error: Mutex::new(None),
            });
        }
        let staging: Arc<[Staging; STAGING_FRAMES]> = Arc::new(
            staging_vec
                .try_into()
                .map_err(|_| "invalid grammar exchange staging frame count")?,
        );
        let (sender, receiver) = mpsc::sync_channel::<WorkerJob>(STAGING_FRAMES);
        let buffers = staging.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let stop = cancelled.clone();
        let worker = thread::Builder::new()
            .name("flare-grammar-exchange".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let frame = &buffers[job.frame];
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute_job(geometry, &job, frame, &stop)
                    }))
                    .unwrap_or_else(|_| Err("grammar exchange worker panicked".into()));
                    if let Err(error) = result {
                        zero_meta(geometry, frame);
                        *frame.error.lock().unwrap() = Some(error);
                        stop.store(true, Ordering::Release);
                    }
                    // Always signal the generation flag so GPU streams can drain before reporting errors.
                    frame.signal(job.generation);
                }
            })
            .map_err(|e| format!("start grammar exchange worker: {e}"))?;
        Ok(Arc::new(Self {
            geometry,
            ranks,
            staging,
            state: Mutex::new(std::array::from_fn(|_| State::default())),
            staged: Mutex::new(HashMap::new()),
            sender: Some(sender),
            worker: Some(worker),
            cancelled,
        }))
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub fn stage_submission(&self, sequence: u64, lanes: Vec<LaneJob>) -> Result<()> {
        if sequence == 0 || lanes.is_empty() {
            return Err("grammar exchange requires a nonzero sequence and at least one lane".into());
        }
        let mut active_slots = Vec::with_capacity(lanes.len());
        for lane in &lanes {
            if lane.slot >= self.geometry.slots || lane.draft_count > self.geometry.max_drafts() {
                return Err("grammar exchange lane exceeds prepared capacity".into());
            }
            #[cfg(feature = "grammar")]
            if lane.grammar.mask_words() != self.geometry.mask_words {
                return Err("grammar mask width disagrees with grammar exchange storage".into());
            }
            active_slots.push((lane.slot, lane.draft_count));
        }
        let mut staged = self.staged.lock().unwrap();
        if staged.contains_key(&sequence) {
            return Err("duplicate staged grammar exchange submission".into());
        }
        staged.insert(
            sequence,
            StagedSubmission {
                lanes: Some(lanes),
                active_slots,
                remaining_ranks: self.ranks,
            },
        );
        Ok(())
    }

    pub fn discard_submission(&self, sequence: u64) {
        self.staged.lock().unwrap().remove(&sequence);
    }

    fn start(
        &self,
        rank: usize,
        sequence: u64,
        mailbox: (usize, usize),
    ) -> Result<Option<(u32, Vec<(usize, usize)>)>> {
        let (active_slots, taken_lanes) = {
            let mut staged = self.staged.lock().unwrap();
            let Some(entry) = staged.get_mut(&sequence) else {
                return Ok(None);
            };
            let slots = entry.active_slots.clone();
            let lanes = entry.lanes.take();
            entry.remaining_ranks = entry.remaining_ranks.saturating_sub(1);
            if entry.remaining_ranks == 0 {
                staged.remove(&sequence);
            }
            (slots, lanes)
        };
        let mut states = self.state.lock().unwrap();
        for staging in self.staging.iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let frame = sequence as usize % STAGING_FRAMES;
        let mut state = states[frame];
        let fresh = state.prepare(rank, sequence, self.ranks)?;
        if fresh {
            let lanes = taken_lanes.ok_or("grammar exchange lanes missing for fresh submission")?;
            self.sender
                .as_ref()
                .unwrap()
                .send(WorkerJob {
                    frame,
                    generation: state.generation,
                    sequence,
                    mailbox_addr: mailbox.0,
                    mailbox_flag: mailbox.1,
                    lanes,
                })
                .map_err(|_| "grammar exchange worker disconnected")?;
        }
        states[frame] = state;
        Ok(Some((state.generation, active_slots)))
    }

    fn finish(&self, rank: usize, sequence: u64) -> Result<()> {
        let mut states = self.state.lock().unwrap();
        let frame = sequence as usize % STAGING_FRAMES;
        if !states[frame].finish(rank, sequence)? {
            return Ok(());
        }
        if let Some(error) = self.staging[frame].error.lock().unwrap().clone() {
            return Err(error);
        }
        Ok(())
    }
}

fn zero_meta(geometry: Geometry, frame: &Staging) {
    // SAFETY: Worker exclusively owns `frame.meta` until `frame.signal(generation)`.
    let meta = unsafe {
        std::slice::from_raw_parts_mut(frame.meta.address() as *mut i32, geometry.slots * 3)
    };
    meta.fill(0);
}

#[cfg(feature = "grammar")]
fn execute_job(
    geometry: Geometry,
    job: &WorkerJob,
    frame: &Staging,
    cancelled: &AtomicBool,
) -> Result<()> {
    if cancelled.load(Ordering::Acquire) {
        return Err("grammar exchange cancelled".into());
    }
    // SAFETY: `mailbox_flag` points to the final int64 word of the mapped-host mailbox,
    // written by the GPU with system-scope release semantics after all draft stores.
    let ready = unsafe { &*(job.mailbox_flag as *const AtomicU64) };
    while ready.load(Ordering::Acquire) != job.sequence {
        if cancelled.load(Ordering::Acquire) {
            return Err("grammar exchange cancelled".into());
        }
        std::hint::spin_loop();
    }
    let max_drafts = geometry.max_drafts();
    let total_draft_words = geometry.slots * max_drafts;
    // SAFETY: GPU released all writes to `[0..slots * max_drafts)` before publishing `sequence`.
    let mailbox = unsafe {
        std::slice::from_raw_parts(job.mailbox_addr as *const i64, total_draft_words)
    };

    let evaluate_lane = |lane: &LaneJob| -> (usize, Option<crate::grammar::SpeculativeMasks>) {
        let base = lane.slot * max_drafts;
        let drafts = &mailbox[base..base + lane.draft_count];
        match lane.grammar.speculative_masks(drafts) {
            Ok(result) => (lane.slot, Some(result)),
            // Per-lane grammar error: zero bounds fail only this lane on GPU without poisoning peers.
            Err(_) => (lane.slot, None),
        }
    };

    let results: Vec<(usize, Option<crate::grammar::SpeculativeMasks>)> = if job.lanes.len() <= 1 {
        job.lanes.iter().map(evaluate_lane).collect()
    } else {
        thread::scope(|scope| {
            let handles: Vec<_> = job
                .lanes
                .iter()
                .map(|lane| scope.spawn(move || evaluate_lane(lane)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("grammar exchange lane worker panicked"))
                .collect()
        })
    };

    let slot_words = geometry.max_width * geometry.mask_words;
    // SAFETY: Worker exclusively owns `frame.masks` and `frame.meta` until `frame.signal`.
    let dst_masks = unsafe {
        std::slice::from_raw_parts_mut(
            frame.masks.address() as *mut i32,
            geometry.slots * slot_words,
        )
    };
    let dst_meta = unsafe {
        std::slice::from_raw_parts_mut(frame.meta.address() as *mut i32, geometry.slots * 3)
    };
    let (counts, rest) = dst_meta.split_at_mut(geometry.slots);
    let (extents, caps) = rest.split_at_mut(geometry.slots);
    for (slot, outcome) in results {
        match outcome {
            Some(spec) => {
                let offset = slot * slot_words;
                dst_masks[offset..offset + spec.masks.len()].copy_from_slice(&spec.masks);
                counts[slot] = spec.mask_count as i32;
                extents[slot] = spec.verify_extent as i32;
                caps[slot] = spec.emit_cap as i32;
            }
            None => {
                counts[slot] = 0;
                extents[slot] = 0;
                caps[slot] = 0;
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "grammar"))]
fn execute_job(
    _geometry: Geometry,
    _job: &WorkerJob,
    _frame: &Staging,
    _cancelled: &AtomicBool,
) -> Result<()> {
    Err("grammar exchange requires the grammar feature".into())
}

#[derive(Clone)]
pub struct Binding {
    pub(crate) shared: Arc<Shared>,
    pub(crate) rank: usize,
    pub(crate) event: usize,
}

impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarExchangeBinding")
            .field("rank", &self.rank)
            .finish()
    }
}

pub(crate) struct Transfers {
    binding: Binding,
    stream: Stream,
    flags: [usize; STAGING_FRAMES],
    mailbox: (usize, usize),
    masks: usize,
    mask_count: usize,
    verify_extent: usize,
    emit_cap: usize,
    row_mask_bytes: usize,
    slot_mask_bytes: usize,
    meta_bytes: usize,
}

impl Transfers {
    pub fn new(binding: Binding, manifest: &ExecutionManifest) -> Result<Self> {
        let geometry = binding.shared.geometry();
        let stream = Stream::open(manifest.context, manifest.device)?;
        let mut flags = [0; STAGING_FRAMES];
        for (flag, staging) in flags
            .iter_mut()
            .zip(binding.shared.staging.iter())
        {
            *flag = stream.mapped_address(&staging.ready)?;
        }
        let mailbox_buf = manifest.buffer(MAILBOX_BUFFER)?;
        let mailbox_host = stream.host_address(mailbox_buf.address, mailbox_buf.nbytes)?;
        let mailbox = (mailbox_host, mailbox_host + mailbox_buf.nbytes - size_of::<i64>());
        Ok(Self {
            binding,
            stream,
            flags,
            mailbox,
            masks: manifest.buffer(MASKS_BUFFER)?.address,
            mask_count: manifest.buffer(MASK_COUNT_BUFFER)?.address,
            verify_extent: manifest.buffer(VERIFY_EXTENT_BUFFER)?.address,
            emit_cap: manifest.buffer(EMIT_CAP_BUFFER)?.address,
            row_mask_bytes: geometry.row_mask_bytes()?,
            slot_mask_bytes: geometry.slot_mask_bytes()?,
            meta_bytes: geometry.meta_bytes()?,
        })
    }

    pub fn enqueue(&self, sequence: u64, inputs_ready: &Event) -> Result<()> {
        let Some((generation, active_slots)) =
            self.binding
                .shared
                .start(self.binding.rank, sequence, self.mailbox)?
        else {
            // Fast path for unconstrained batches: signal readiness immediately with zero DMA.
            // SAFETY: Event belongs to this context and outlives all queued work.
            unsafe {
                self.stream.record_borrowed(self.binding.event)?;
            }
            return Ok(());
        };
        let frame = sequence as usize % STAGING_FRAMES;
        let staging = &self.binding.shared.staging[frame];
        // Wait for the model stream to start this submission so DMA cannot overwrite
        // exchange buffers still read by any preceding graph.
        self.stream.wait(inputs_ready)?;
        // SAFETY: Staging buffers and event remain owned until close drains all streams.
        unsafe {
            self.stream.wait_value(self.flags[frame], generation)?;
            for (slot, draft_count) in active_slots {
                let slot_offset = slot * self.slot_mask_bytes;
                let copy_bytes = (draft_count + 1) * self.row_mask_bytes;
                self.stream
                    .upload(self.masks + slot_offset, &staging.masks, slot_offset, copy_bytes)?;
            }
            self.stream
                .upload(self.mask_count, &staging.meta, 0, self.meta_bytes)?;
            self.stream
                .upload(self.verify_extent, &staging.meta, self.meta_bytes, self.meta_bytes)?;
            self.stream
                .upload(self.emit_cap, &staging.meta, 2 * self.meta_bytes, self.meta_bytes)?;
            self.stream.record_borrowed(self.binding.event)?;
        }
        Ok(())
    }

    pub fn finish(&self, sequence: u64) -> Result<()> {
        self.binding.shared.finish(self.binding.rank, sequence)
    }

    pub fn cancel(&self) {
        self.binding.shared.cancelled.store(true, Ordering::Release);
    }

    pub fn drain(&self) -> Result<()> {
        self.stream.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_validates_bounds_and_byte_sizes() {
        let g = Geometry { slots: 16, max_width: 8, mask_words: 7760 };
        g.validate().unwrap();
        assert_eq!(g.max_drafts(), 7);
        assert_eq!(g.row_mask_bytes().unwrap(), 7760 * 4);
        assert_eq!(g.slot_mask_bytes().unwrap(), 8 * 7760 * 4);
        assert_eq!(g.meta_bytes().unwrap(), 16 * 4);

        assert!(Geometry { slots: 0, max_width: 8, mask_words: 1 }.validate().is_err());
        assert!(Geometry { slots: 1, max_width: 1, mask_words: 1 }.validate().is_err());
        assert!(Geometry { slots: 1, max_width: 8, mask_words: 0 }.validate().is_err());
    }

    #[test]
    fn staging_state_tracks_multi_rank_generations_and_rejects_reuse() {
        let mut s = State::default();
        assert!(s.prepare(0, 1, 2).unwrap());
        assert!(!s.prepare(1, 1, 2).unwrap());
        assert!(s.prepare(0, 1, 2).is_err());
        assert!(s.prepare(0, 3, 2).is_err());
        assert!(s.finish(0, 1).unwrap());
        assert!(s.finish(0, 1).is_err());
        assert!(s.prepare(0, 3, 2).is_err());
        assert!(s.finish(1, 1).unwrap());
        assert!(s.prepare(0, 3, 2).unwrap());
        assert_eq!(s.generation, 2);
        assert!(!s.finish(0, 1).unwrap());
    }
}
