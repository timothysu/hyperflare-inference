/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Retained duration estimates; zero denotes an unmeasured graph cost.

use super::ScheduledStep;
use crate::{
    Result,
    execution::{ExecutionTiming, GraphDescriptor, GraphKind},
};

#[cfg(test)]
thread_local! {
    /// Cost-table lookups on this thread, for tests that bound planning work.
    pub(super) static ENTRY_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn observed_duration_ns(timing: ExecutionTiming) -> Option<u64> {
    timing
        .gpu_ns
        .into_iter()
        .flatten()
        .max()
        .filter(|ns| *ns > 0)
        .map(|ns| if timing.queued { ns } else { ns.max(timing.wall_ns) })
}

/// Prefers the shortest positive measured duration, yielding `0` only when every
/// aggregated shape is unmeasured.
pub(super) fn min_measured_duration(current: Option<u64>, nanos: u64) -> Option<u64> {
    Some(match current {
        None | Some(0) => nanos,
        Some(old) if nanos == 0 => old,
        Some(old) => old.min(nanos),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key {
    pub drafts: usize,
    pub kind: GraphKind,
    pub batch: usize,
    pub tokens: usize,
    pub prefill: usize,
    pub packed: usize,
}

impl Key {
    pub fn from_step(step: &ScheduledStep) -> Self {
        Self { drafts: step.draft_tokens, kind: step.kind, batch: step.batch_capacity, tokens: step.token_capacity,
            prefill: step.prefill_capacity(), packed: step.packed_token_capacity }
    }
}

pub(super) struct PackedRoundShape {
    pub batch: usize,
    pub drafts: usize,
    pub width: usize,
    pub tokens: usize,
    pub nanos: u64,
}

/// Share of the context-slope statistics retained at each new observation.
const CONTEXT_SLOPE_RETENTION: f64 = 15.0 / 16.0;

struct Estimate {
    nanos: u64,
    /// Request context position at which `nanos` was observed; zero for graphs
    /// whose cost is not projected over context.
    position: u64,
    replays: ReplayTimes,
    observed_at_ns: u64,
    last_attempt_ns: u64,
}

#[derive(Default)]
struct ReplayTimes {
    samples: [(u64, u64); 9],
    len: usize,
    next: usize,
}

impl ReplayTimes {
    /// Records a duration with the context position it was observed at and returns
    /// the median duration with the position of the sample that supplied it.
    fn observe(&mut self, sample: u64, position: u64, fresh: bool) -> (u64, u64) {
        if !fresh {
            *self = Self::default();
        }
        self.samples[self.next] = (sample, position);
        self.next = (self.next + 1) % self.samples.len();
        self.len = (self.len + 1).min(self.samples.len());
        let mut ordered = self.samples;
        ordered[..self.len].sort_unstable();
        let middle = self.len / 2;
        if self.len % 2 == 0 {
            let (low, high) = (ordered[middle - 1], ordered[middle]);
            (((u128::from(low.0) + u128::from(high.0) + 1) / 2) as u64,
                ((u128::from(low.1) + u128::from(high.1)) / 2) as u64)
        } else {
            ordered[middle]
        }
    }
}

impl Estimate {
    fn fresh(&self, now_ns: u64, ttl_ns: u64) -> bool {
        self.nanos != 0 && now_ns.saturating_sub(self.observed_at_ns) < ttl_ns
    }
}

#[derive(Default)]
pub(super) struct ActionCosts {
    entries: Vec<(Key, Estimate)>,
    index: std::collections::HashMap<Key, usize>,
    packed_round_durations: [Vec<Option<u64>>; 7],
    /// Least-squares statistics, through the origin, of how far batch-one prefill
    /// samples depart from their graph's retained estimate against capacity times
    /// the context distance between them: the summed products and the summed squares.
    context_slope: (f64, f64),
    now_ns: u64,
    ttl_ns: u64,
}

impl ActionCosts {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub fn set_measured(&mut self, step: &ScheduledStep, nanos: u64) {
        let key = Key {
            drafts: step.draft_tokens,
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        let estimate = &mut self.entries[self.index[&key]].1;
        estimate.nanos = nanos;
        estimate.position = 0;
        estimate.replays.observe(nanos, 0, false);
        estimate.observed_at_ns = self.now_ns;
        self.refresh_packed_durations();
    }

    pub fn prepared(graphs: &[GraphDescriptor], token_capacity: usize) -> Result<Self> {
        let mut entries = Vec::with_capacity(graphs.len());
        let mut index = std::collections::HashMap::with_capacity(graphs.len());
        for graph in graphs {
            let key = Key {
                drafts: graph.draft_tokens,
                kind: graph.kind,
                batch: graph.batch_capacity,
                tokens: graph.token_capacity,
                prefill: graph.prefill_capacity,
                packed: graph.packed_token_capacity,
            };
            if index.insert(key, entries.len()).is_some() {
                return Err("prepared graphs require unique shapes".into());
            }
            entries.push((
                key,
                Estimate {
                    nanos: 0,
                    position: 0,
                    replays: ReplayTimes::default(),
                    observed_at_ns: 0,
                    last_attempt_ns: 0,
                },
            ));
        }
        let index = entries.iter().enumerate().map(|(index, (key, _))| (*key, index)).collect();
        let mut costs = Self { entries, index,
            packed_round_durations: std::array::from_fn(|_| vec![None; token_capacity + 1]),
            ttl_ns: super::PolicyConfig::default().estimate_ttl_ns, ..Default::default() };
        costs.refresh_packed_durations();
        Ok(costs)
    }

    pub fn advance_to(&mut self, now_ns: u64, ttl_ns: u64) {
        self.now_ns = now_ns;
        self.ttl_ns = ttl_ns;
    }

    pub fn duration(&self, kind: GraphKind, batch: usize, tokens: usize) -> Option<u64> {
        self.shape_duration(kind, batch, tokens, 0)
    }

    fn refresh_packed_durations(&mut self) {
        for row in &mut self.packed_round_durations { row.fill(None); }
        for (key, estimate) in &self.entries {
            if key.kind != GraphKind::Speculative || key.packed == 0 { continue; }
            let maximum = (key.drafts + 1).min(key.tokens).min(key.packed);
            let nanos = estimate.nanos;
            for duration in self.packed_round_durations[key.batch.trailing_zeros() as usize].iter_mut().take(maximum + 1).skip(1) {
                *duration = min_measured_duration(*duration, nanos);
            }
        }
    }

    pub fn packed_round_duration(&self, batch: usize, width: usize) -> Option<u64> {
        if !batch.is_power_of_two() || batch > super::REQUEST_LANES { return None; }
        self.packed_round_durations.get(batch.trailing_zeros() as usize)?.get(width).copied().flatten()
    }

    pub fn packed_round_shapes(&self) -> impl Iterator<Item = PackedRoundShape> + '_ {
        self.entries.iter().filter(|(key, _)| key.kind == GraphKind::Speculative && key.packed != 0)
            .map(|(key, estimate)| PackedRoundShape {
                batch: key.batch, drafts: key.drafts, width: key.tokens, tokens: key.packed,
                nanos: estimate.nanos,
            })
    }

    pub fn shape_duration(
        &self,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        prefill: usize,
    ) -> Option<u64> {
        self.exact_duration(kind, batch, tokens, prefill, 0)
    }

    fn exact_duration(&self, kind: GraphKind, batch: usize, tokens: usize, prefill: usize, packed: usize) -> Option<u64> {
        self.entry(Key {
            drafts: if kind == GraphKind::Speculative { tokens - 1 } else { 0 },
            kind,
            batch,
            tokens,
            prefill,
            packed: if kind == GraphKind::Speculative { batch * tokens } else { packed },
        })
        .map(|estimate| estimate.nanos)
    }

    /// Nanoseconds that a batch-one prefill graph gains per capacity token and per
    /// token of request context, measured across every prefill graph; zero until one
    /// graph has executed at two context positions.
    fn context_slope(&self) -> f64 {
        let (products, squares) = self.context_slope;
        if squares > 0.0 { (products / squares).max(0.0) } else { 0.0 }
    }

    /// Retained duration of a batch-one prefill graph, projected from the context
    /// position of its estimate to `position`. An estimate taken at a shallow context
    /// otherwise understates a graph's cost deep in a long prompt, and the median of
    /// a graph in continuous use lags its growth. A projection never falls below half
    /// the retained estimate.
    pub fn prefill_duration(&self, kind: GraphKind, capacity: usize, position: u64) -> Option<u64> {
        let estimate = self.entry(Key { drafts: 0, kind, batch: 1, tokens: capacity, prefill: 0, packed: 0 })?;
        if estimate.nanos == 0 {
            return Some(0);
        }
        let retained = estimate.nanos as f64;
        let shift = self.context_slope() * capacity as f64 * (position as f64 - estimate.position as f64);
        Some((retained + shift).max(retained / 2.0).round() as u64)
    }

    pub fn duration_key(&self, key: Key) -> Option<u64> {
        self.entry(key).map(|estimate| estimate.nanos)
    }

    pub fn is_fresh(&self, key: Key) -> bool {
        self.entry(key).is_some_and(|estimate| estimate.fresh(self.now_ns, self.ttl_ns))
    }

    pub fn fresh_duration(&self, kind: GraphKind, tokens: usize) -> Option<u64> {
        let key = Key { drafts: 0, kind, batch: 1, tokens, prefill: 0, packed: 0 };
        self.entry(key).filter(|estimate| estimate.fresh(self.now_ns, self.ttl_ns))
            .map(|estimate| estimate.nanos)
    }

    pub fn stale_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.entries.iter().filter(|(_, estimate)| estimate.nanos != 0
            && !estimate.fresh(self.now_ns, self.ttl_ns)).map(|(key, _)| *key)
    }

    pub fn refresh_order(&self, key: Key) -> u64 {
        self.entry(key).map_or(0, |estimate| estimate.observed_at_ns.max(estimate.last_attempt_ns))
    }

    pub fn record_attempt(&mut self, key: Key) {
        if let Some(&index) = self.index.get(&key) {
            let estimate = &mut self.entries[index].1;
            estimate.last_attempt_ns = self.now_ns;
        }
    }

    pub fn observation(&self, key: Key) -> Option<(u64, Option<u64>, bool)> {
        self.entry(key).map(|estimate| (estimate.nanos,
            (estimate.nanos != 0).then_some(estimate.observed_at_ns),
            estimate.fresh(self.now_ns, self.ttl_ns)))
    }

    fn entry(&self, key: Key) -> Option<&Estimate> {
        #[cfg(test)]
        ENTRY_LOOKUPS.set(ENTRY_LOOKUPS.get() + 1);
        self.index.get(&key).map(|&index| &self.entries[index].1)
    }

    pub fn get(&self, step: &ScheduledStep) -> Option<u64> {
        self.duration_key(Key::from_step(step))
    }

    pub fn observe(&mut self, step: &ScheduledStep, timing: ExecutionTiming) {
        let Some(raw_sample) = timing.gpu_ns.into_iter().flatten().max().filter(|n| *n > 0) else {
            return;
        };
        let sample = raw_sample.saturating_sub(timing.readiness.map_or(0, |intervals| intervals.wait_ns()));
        if sample == 0 {
            return;
        }
        let key = Key {
            drafts: step.draft_tokens,
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        // A batch-one prefill lane starts at its request's context position.
        let position = (step.kind.is_prefill() && step.batch_capacity == 1)
            .then(|| step.lanes.first().map(|lane| lane.start_position))
            .flatten();
        if let Some(&index) = self.index.get(&key) {
            let estimate = &mut self.entries[index].1;
            let fresh = estimate.fresh(self.now_ns, self.ttl_ns);
            if let Some(position) = position.filter(|_| estimate.nanos != 0) {
                // The retained median is the reference: one slow or fast replay moves
                // it little, and it carries the context position of its own sample.
                let reach = key.tokens as f64 * (position as f64 - estimate.position as f64);
                let growth = sample as f64 - estimate.nanos as f64;
                self.context_slope = (
                    CONTEXT_SLOPE_RETENTION * self.context_slope.0 + growth * reach,
                    CONTEXT_SLOPE_RETENTION * self.context_slope.1 + reach * reach,
                );
            }
            let position = position.unwrap_or(0);
            (estimate.nanos, estimate.position) = estimate.replays.observe(sample, position, fresh);
            estimate.observed_at_ns = self.now_ns;
            self.refresh_packed_durations();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionCosts, ReplayTimes, ScheduledStep};
    use crate::execution::{ExecutionTiming, GraphDescriptor, GraphKind, SamplingMode};
    use crate::scheduler::{RequestHandle, ScheduledLane};

    #[test]
    fn isolated_slow_replay_does_not_displace_the_steady_cost() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, 0, true).0, 8_000_000);
        }
        assert_eq!(replays.observe(16_000_000, 0, true).0, 8_000_000);
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, 0, true).0, 8_000_000);
        }
    }

    #[test]
    fn sustained_cost_changes_replace_the_window_and_staleness_resets_it() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            replays.observe(8, 0, true);
        }
        for _ in 0..4 {
            assert_eq!(replays.observe(16, 0, true).0, 8);
        }
        assert_eq!(replays.observe(16, 0, true).0, 16);
        assert_eq!(replays.observe(32, 0, false).0, 32);
        assert_eq!(replays.observe(16, 0, true).0, 24);
    }

    #[test]
    fn even_sample_median_does_not_overflow() {
        let mut replays = ReplayTimes::default();
        assert_eq!(replays.observe(u64::MAX, 0, true).0, u64::MAX);
        assert_eq!(replays.observe(u64::MAX - 1, 0, true).0, u64::MAX);
    }

    #[test]
    fn the_median_duration_keeps_the_context_position_of_its_sample() {
        let mut replays = ReplayTimes::default();
        assert_eq!(replays.observe(100, 0, true), (100, 0));
        assert_eq!(replays.observe(120, 4096, true), (110, 2048));
        assert_eq!(replays.observe(140, 8192, true), (120, 4096));
        // An outlier does not move the median or its position.
        assert_eq!(replays.observe(900, 12_288, true), (130, 6144));
        assert_eq!(replays.observe(160, 16_384, true), (140, 8192));
    }

    #[test]
    fn prefill_durations_are_projected_along_the_measured_context_slope() {
        let graph = |capacity: usize| GraphDescriptor {
            name: format!("prefill{capacity}"),
            kind: GraphKind::Prefill,
            sampling_mode: SamplingMode::General,
            confidence_width: 0,
            draft_tokens: 0,
            handle: capacity,
            batch_capacity: 1,
            token_capacity: capacity,
            packed_token_capacity: 0,
            prefill_capacity: 0,
            estimated_duration_ns: 0,
            input_bindings: vec![],
            tuned_bindings: vec![],
        };
        let request = RequestHandle { slot: 0, generation: 1 };
        let step = |capacity: usize, position: u64, request: RequestHandle| ScheduledStep {
            draft_tokens: 0,
            kind: GraphKind::Prefill,
            batch_capacity: 1,
            token_capacity: capacity,
            packed_token_capacity: 0,
            mixed: None,
            lanes: vec![ScheduledLane { request, start_position: position, tokens: capacity }],
            requires_feedback: false,
        };
        let timing = |nanos: u64| {
            let mut gpu_ns = [None; 16];
            gpu_ns[0] = Some(nanos);
            ExecutionTiming { gpu_ns, ..Default::default() }
        };
        let mut costs = ActionCosts::prepared(&[graph(2048), graph(4096)], 8).unwrap();
        costs.advance_to(1, 30_000_000_000);
        // The smaller graph is measured once at an empty context.
        costs.observe(&step(2048, 0, request), timing(400_000_000));
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 2048, 1 << 20), Some(400_000_000));
        // The larger graph gains 0.25 ns per capacity token and context token.
        let duration = |position: u64| 800_000_000 + 4096 * position / 4;
        for chunk in 0..6 {
            costs.observe(&step(4096, chunk * 4096, request), timing(duration(chunk * 4096)));
        }
        let position = 6 * 4096;
        let close = |projected: Option<u64>, expected: u64| projected.unwrap().abs_diff(expected) <= 1000;
        assert!(close(costs.prefill_duration(GraphKind::Prefill, 4096, position), duration(position)));
        assert!(close(costs.prefill_duration(GraphKind::Prefill, 2048, position), 400_000_000 + 2048 * position / 4));
        // The retained estimates themselves are unchanged.
        assert_eq!(costs.duration(GraphKind::Prefill, 1, 2048), Some(400_000_000));
        assert_eq!(costs.duration(GraphKind::Prefill, 1, 4096), Some((duration(2 * 4096) + duration(3 * 4096) + 1) / 2));
        // A request at an empty context is projected back, never below half the estimate.
        assert!(close(costs.prefill_duration(GraphKind::Prefill, 4096, 0), duration(0)));
        let another = RequestHandle { slot: 1, generation: 1 };
        for chunk in 0..9 {
            costs.observe(&step(4096, (1 << 22) + chunk * 4096, another), timing(duration((1 << 22) + chunk * 4096)));
        }
        let retained = costs.duration(GraphKind::Prefill, 1, 4096).unwrap();
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 4096, 0), Some(retained.div_ceil(2)));
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 1024, 0), None);
    }

    #[test]
    fn unmeasured_packed_sibling_does_not_zero_measured_round_duration() {
        let make = |name: &str, drafts, width, packed| GraphDescriptor {
            name: name.into(),
            kind: GraphKind::Speculative,
            sampling_mode: SamplingMode::General,
            confidence_width: drafts,
            draft_tokens: drafts,
            handle: 1,
            batch_capacity: 16,
            token_capacity: width,
            packed_token_capacity: packed,
            prefill_capacity: 0,
            estimated_duration_ns: 0,
            input_bindings: vec![],
            tuned_bindings: vec![],
        };
        let step_for = |drafts, width, packed| ScheduledStep {
            draft_tokens: drafts,
            kind: GraphKind::Speculative,
            batch_capacity: 16,
            token_capacity: width,
            packed_token_capacity: packed,
            mixed: None,
            lanes: vec![],
            requires_feedback: false,
        };
        let graphs = vec![
            make("b16w4p64", 3, 4, 64),
            make("b16w6p39", 5, 6, 39),
        ];
        let mut costs = ActionCosts::prepared(&graphs, 8).unwrap();
        assert_eq!(costs.packed_round_duration(16, 4), Some(0));
        // Measured first entry must not be zeroed by trailing unmeasured sibling.
        costs.set_measured(&step_for(3, 4, 64), 24_000_000);
        assert_eq!(costs.packed_round_duration(16, 4), Some(24_000_000));
        assert_eq!(costs.packed_round_duration(16, 6), Some(0));
        // Measuring a faster second sibling replaces the positive minimum.
        costs.set_measured(&step_for(5, 6, 39), 18_000_000);
        assert_eq!(costs.packed_round_duration(16, 4), Some(18_000_000));
        assert_eq!(costs.packed_round_duration(16, 6), Some(18_000_000));

        // Reverse order: unmeasured first entry must not block a measured sibling.
        let mut reversed = ActionCosts::prepared(&graphs, 8).unwrap();
        reversed.set_measured(&step_for(5, 6, 39), 18_000_000);
        assert_eq!(reversed.packed_round_duration(16, 4), Some(18_000_000));
        assert_eq!(reversed.packed_round_duration(16, 6), Some(18_000_000));
    }
}
