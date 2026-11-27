/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Request-local speculative acceptance and copy proposal metadata.

use crate::{Result, execution::BATCH_CAPACITIES};

/// Copy provenance uses the nonnegative bits of its signed 64-bit mask.
pub const COPY_DRAFT_CAPACITY: usize = i64::BITS as usize - 1;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProposalSource {
    #[default]
    Mtp,
    Copy,
}

impl ProposalSource {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "mtp" => Ok(Self::Mtp),
            "copy" => Ok(Self::Copy),
            _ => Err("speculative_proposal_source must be mtp or copy".into()),
        }
    }
}

/// Host-validated continuation and masks for the corresponding committed prefix.
#[derive(Clone, Debug)]
pub struct CopyProposal {
    pub consumed: u64,
    pub tokens: Vec<i64>,
    pub count: usize,
    pub grammar_masks: Vec<i32>,
}

pub fn graph_catalog(widths: &[usize], fixed: Option<usize>, capacity: usize) -> [Vec<usize>; 7] {
    BATCH_CAPACITIES.map(|batch| if batch <= capacity {
        fixed.map_or_else(|| widths.to_vec(), |width| vec![width])
    } else { Vec::new() })
}

pub fn validate_widths(widths: &[usize]) -> Result<()> {
    if !widths.is_empty()
        && (widths[0] != 1
            || widths.windows(2).any(|w| w[0] >= w[1])
            || widths.iter().any(|w| *w == 0))
    {
        return Err(
            "speculative widths must be sorted unique positive integers including 1".into(),
        );
    }
    Ok(())
}

/// Observations a learned population prior is worth in a request's survival estimate.
/// A request must contradict the population about this many times before its own
/// evidence outweighs it.
pub(super) const POPULATION_PRIOR_STRENGTH: f64 = 16.0;
/// Observations the fixed geometric prior is worth; no population backs it.
const GEOMETRIC_PRIOR_STRENGTH: f64 = 2.0;
/// Generated tokens over which request evidence loses half its weight.
const EVIDENCE_HALF_LIFE: f64 = 128.0;
const EVIDENCE_CAPACITY: f64 = 128.0;

/// Prior prefix survival at one draft depth, with its weight in observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DepthPrior {
    pub mean: f64,
    pub strength: f64,
}

/// The prior of proposals without a learned population: survival halves per draft depth.
pub(super) fn geometric_prior(depth: usize) -> DepthPrior {
    DepthPrior { mean: 0.5f64.powi(depth as i32 + 1), strength: GEOMETRIC_PRIOR_STRENGTH }
}

pub(super) fn geometric_priors(depths: usize) -> Vec<DepthPrior> {
    (0..depths).map(geometric_prior).collect()
}

/// The prior at `depth`, geometric beyond the depths that `priors` covers.
pub(super) fn prior_at(priors: &[DepthPrior], depth: usize) -> DepthPrior {
    priors.get(depth).copied().unwrap_or_else(|| geometric_prior(depth))
}

/// Weighted least-squares projection onto nonincreasing sequences, by pooling adjacent
/// violators. Prefix survival cannot rise with depth; pooling shares a violation between
/// the depths that produce it instead of letting the shallower estimate cap the deeper ones.
pub(super) fn project_nonincreasing(values: &mut [f64], weights: &[f64]) {
    debug_assert_eq!(values.len(), weights.len());
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(values.len());
    for (&value, &weight) in values.iter().zip(weights) {
        let mut block = (value, weight.max(f64::MIN_POSITIVE), 1);
        while let Some(&(mean, pooled, length)) = blocks.last() {
            if mean >= block.0 {
                break;
            }
            let total = pooled + block.1;
            block = ((mean * pooled + block.0 * block.1) / total, total, length + block.2);
            blocks.pop();
        }
        blocks.push(block);
    }
    let mut index = 0;
    for (mean, _, length) in blocks {
        values[index..index + length].fill(mean);
        index += length;
    }
}

/// One request's observations of a prefix depth. Their weight ages as the request
/// generates, so an estimate returns to its prior when the depth stops being observed.
#[derive(Clone, Copy, Debug)]
pub(super) struct PrefixEvidence {
    weight: f64,
    sum: f64,
    observed_at: u32,
}

impl PrefixEvidence {
    pub fn new(generated: u32) -> Self {
        Self { weight: 0.0, sum: 0.0, observed_at: generated }
    }

    fn decay(&self, generated: u32) -> f64 {
        (-f64::from(generated.saturating_sub(self.observed_at)) / EVIDENCE_HALF_LIFE).exp2()
    }

    pub fn weight_at(&self, generated: u32) -> f64 {
        self.weight * self.decay(generated)
    }

    /// Evidence and prior together, in observations.
    pub fn information(&self, generated: u32, prior: DepthPrior) -> f64 {
        self.weight_at(generated) + prior.strength
    }

    /// Mean of the aged evidence and the prior, each weighted by its observations.
    pub fn estimate(&self, generated: u32, prior: DepthPrior) -> f64 {
        let decay = self.decay(generated);
        let information = self.weight * decay + prior.strength;
        if information > 0.0 {
            (self.sum * decay + prior.strength * prior.mean) / information
        } else {
            prior.mean
        }
    }

    pub fn observe(&mut self, value: f64, generated: u32) {
        debug_assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        let aged = self.weight_at(generated);
        let mean = if aged > 0.0 { self.sum * self.decay(generated) / aged } else { value };
        self.weight = (aged + 1.0).min(EVIDENCE_CAPACITY);
        self.sum = (mean + (value - mean) / self.weight) * self.weight;
        self.observed_at = generated;
    }
}

/// Radius of the 95% one-sided Hoeffding bound for `information` observations.
pub(super) fn optimism(information: f64) -> f64 {
    (20.0f64.ln() / (2.0 * information.max(1e-9))).sqrt()
}

/// Verified prefix survival of one request's proposals, by proposal source.
#[derive(Clone)]
pub struct AcceptanceProfile {
    prefixes: [Vec<PrefixEvidence>; 2],
    rounds: u64,
}

impl Default for AcceptanceProfile {
    fn default() -> Self {
        Self::new(8)
    }
}

impl AcceptanceProfile {
    pub fn new(token_capacity: usize) -> Self {
        Self { prefixes: std::array::from_fn(|_| vec![PrefixEvidence::new(0); token_capacity.saturating_sub(1)]),
            rounds: 0 }
    }

    /// Survival of the prefix through `depth` from this request's evidence and the
    /// geometric prior. Depth zero is the first draft after the pending target token.
    /// Selection uses the population prior where one exists.
    pub fn prefix_probability(&self, source: ProposalSource, depth: usize) -> Option<f64> {
        let depths = self.prefixes[source as usize].len();
        (depth < depths).then(|| self.curve(source, 0, &geometric_priors(depths), false)[depth])
    }

    /// Nonincreasing prefix survival by depth. The optimistic curve adds each depth's
    /// confidence radius and cannot exceed a shallower depth's bound.
    pub(super) fn curve(&self, source: ProposalSource, generated: u32, priors: &[DepthPrior], optimistic: bool) -> Vec<f64> {
        let evidence = &self.prefixes[source as usize];
        let mut values: Vec<f64> = evidence.iter().enumerate()
            .map(|(depth, evidence)| evidence.estimate(generated, prior_at(priors, depth))).collect();
        let information: Vec<f64> = evidence.iter().enumerate()
            .map(|(depth, evidence)| evidence.information(generated, prior_at(priors, depth))).collect();
        project_nonincreasing(&mut values, &information);
        if optimistic {
            let mut bound = 1.0f64;
            for (value, information) in values.iter_mut().zip(&information) {
                bound = bound.min(*value + optimism(*information));
                *value = bound;
            }
        }
        values
    }

    /// Expected tokens of a round of `width` positions against the geometric prior.
    pub fn expected_tokens(&self, source: ProposalSource, width: usize) -> f64 {
        let depths = self.prefixes[source as usize].len();
        f64::from(width != 0) + self.curve(source, 0, &geometric_priors(depths), false)
            [..width.saturating_sub(1).min(depths)].iter().sum::<f64>()
    }

    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    pub(super) fn information(&self, source: ProposalSource, depth: usize, generated: u32, prior: DepthPrior) -> f64 {
        self.prefixes[source as usize][depth].information(generated, prior)
    }

    pub(super) fn observe(
        &mut self,
        source: ProposalSource,
        generated: u32,
        offered: usize,
        accepted: usize,
        rejected: bool,
        survival: Option<&[f32]>,
    ) {
        debug_assert!(accepted <= offered && offered <= self.prefixes[source as usize].len());
        if offered == 0 {
            return;
        }
        for (depth, value) in observed_prefixes(offered, accepted, rejected, survival) {
            self.prefixes[source as usize][depth].observe(value, generated);
        }
        self.rounds = self.rounds.saturating_add(1);
    }
}

/// Offered prefix depths whose survival a round determined, with the observed value.
/// Rejection falsifies every longer offered prefix; termination censors its suffix.
pub(super) fn observed_prefixes(offered: usize, accepted: usize, rejected: bool,
    survival: Option<&[f32]>) -> impl Iterator<Item = (usize, f64)> + '_ {
    let observed = if rejected { offered } else { accepted };
    (0..observed).map(move |depth| (depth, survival.map_or(f64::from(depth < accepted), |p| f64::from(p[depth]))))
}

#[cfg(test)]
mod tests {
    use super::*;

    const POPULATION: DepthPrior = DepthPrior { mean: 0.25, strength: POPULATION_PRIOR_STRENGTH };

    #[test]
    fn aged_evidence_returns_the_estimate_to_its_prior() {
        let mut evidence = PrefixEvidence::new(0);
        assert_eq!(evidence.estimate(0, POPULATION), 0.25);
        for _ in 0..256 { evidence.observe(0.0, 0); }
        let observed = evidence.estimate(0, POPULATION);
        assert!(observed < 0.03);
        assert!(evidence.estimate(128, POPULATION) > observed);
        assert!((evidence.estimate(u32::MAX, POPULATION) - 0.25).abs() < 1e-12);
        evidence.observe(1.0, 4096);
        assert!((evidence.estimate(4096, POPULATION) - (1.0 + 16.0 * 0.25) / 17.0).abs() < 1e-9);
    }

    #[test]
    fn one_observation_does_not_outweigh_the_population() {
        let mut evidence = PrefixEvidence::new(0);
        evidence.observe(0.0, 0);
        assert!((evidence.estimate(0, POPULATION) - 16.0 * 0.25 / 17.0).abs() < 1e-12);
        for _ in 0..9 { evidence.observe(0.0, 0); }
        assert!((evidence.estimate(0, POPULATION) - 16.0 * 0.25 / 26.0).abs() < 1e-12);
    }

    #[test]
    fn unobserved_depths_follow_the_prior_instead_of_freezing() {
        let mut profile = AcceptanceProfile::default();
        let low = vec![DepthPrior { mean: 0.0, strength: POPULATION_PRIOR_STRENGTH }; 7];
        let high = vec![DepthPrior { mean: 0.75, strength: POPULATION_PRIOR_STRENGTH }; 7];
        for _ in 0..8 { profile.observe(ProposalSource::Mtp, 0, 7, 0, true, None); }
        // Shallow rounds leave depths one and beyond unobserved while the request advances.
        for generated in 1..=1024 {
            profile.observe(ProposalSource::Mtp, generated, 1, 1, true, None);
        }
        let means = profile.curve(ProposalSource::Mtp, 1024, &high, false);
        assert!(means[0] > 0.95);
        assert!(means[1..].iter().all(|mean| (mean - 0.75).abs() < 0.01), "{means:?}");
        assert!(profile.curve(ProposalSource::Mtp, 1024, &low, false)[1..].iter().all(|mean| *mean < 0.01));
        let upper = profile.curve(ProposalSource::Mtp, 1024, &high, true);
        assert!(upper.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(upper.iter().zip(&means).all(|(upper, mean)| upper >= mean));
    }

    #[test]
    fn projection_pools_violations_and_preserves_ordered_curves() {
        let mut ordered = [0.9, 0.5, 0.5, 0.1];
        project_nonincreasing(&mut ordered, &[1.0; 4]);
        assert_eq!(ordered, [0.9, 0.5, 0.5, 0.1]);

        // The weighted mean of the violating pair replaces both; a running minimum
        // would instead have cut the second depth to 0.2.
        let mut violated = [0.2, 0.8, 0.1];
        project_nonincreasing(&mut violated, &[1.0, 3.0, 1.0]);
        assert!((violated[0] - 0.65).abs() < 1e-12 && (violated[1] - 0.65).abs() < 1e-12);
        assert_eq!(violated[2], 0.1);

        let mut cascade = [0.1, 0.2, 0.3, 0.4];
        project_nonincreasing(&mut cascade, &[1.0; 4]);
        assert!(cascade.iter().all(|value| (value - 0.25).abs() < 1e-12));

        let mut values = [0.31, 0.77, 0.12, 0.64, 0.64, 0.05, 0.4];
        let weights = [2.0, 1.0, 5.0, 0.5, 3.0, 1.0, 4.0];
        let total: f64 = values.iter().zip(&weights).map(|(value, weight)| value * weight).sum();
        project_nonincreasing(&mut values, &weights);
        assert!(values.windows(2).all(|pair| pair[0] >= pair[1]));
        let projected: f64 = values.iter().zip(&weights).map(|(value, weight)| value * weight).sum();
        assert!((projected - total).abs() < 1e-12);
    }

    #[test]
    fn a_noisy_shallow_depth_does_not_cap_well_observed_deeper_depths() {
        let mut profile = AcceptanceProfile::default();
        let priors = vec![DepthPrior { mean: 0.5, strength: 1.0 }; 7];
        // Depth one holds many high observations; depth zero holds one low observation.
        for _ in 0..32 { profile.prefixes[ProposalSource::Mtp as usize][1].observe(0.8, 0); }
        profile.prefixes[ProposalSource::Mtp as usize][0].observe(0.0, 0);
        let curve = profile.curve(ProposalSource::Mtp, 0, &priors, false);
        assert!(curve[1] > 0.7, "{curve:?}");
        assert!(curve.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn observed_prefixes_cover_offered_depths_only() {
        let depths = |offered, accepted, rejected, survival: Option<&[f32]>|
            observed_prefixes(offered, accepted, rejected, survival).collect::<Vec<_>>();
        assert_eq!(depths(3, 1, true, None), [(0, 1.0), (1, 0.0), (2, 0.0)]);
        assert_eq!(depths(3, 1, false, None), [(0, 1.0)]);
        assert_eq!(depths(0, 0, true, None), []);
        assert_eq!(depths(2, 0, true, Some(&[0.5, 0.25])), [(0, 0.5), (1, 0.25)]);
    }

    #[test]
    fn probability_feedback_replaces_the_binary_sample_and_respects_censoring() {
        let mut profile = AcceptanceProfile::default();
        let survival = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01, -1., -1., -1., -1., -1., -1., -1., -1.];
        profile.observe(ProposalSource::Mtp, 0, 7, 0, true, Some(&survival));
        for depth in 0..7 {
            let expected = (2.0 * 0.5f64.powi(depth as i32 + 1) + f64::from(survival[depth])) / 3.0;
            assert!((profile.prefix_probability(ProposalSource::Mtp, depth).unwrap() - expected).abs() < 1e-12);
        }
        let priors = geometric_priors(7);
        let before = profile.curve(ProposalSource::Mtp, 0, &priors, false);
        profile.observe(ProposalSource::Mtp, 0, 7, 1, false, Some(&survival));
        assert_eq!(profile.curve(ProposalSource::Mtp, 0, &priors, false)[1..], before[1..]);
    }
}

#[cfg(test)]
#[test]
fn arbitrary_width_catalog_preserves_declared_values() {
    validate_widths(&[1, 8, 13, 16, 32]).unwrap();
    assert!(validate_widths(&[1, 0]).is_err());
    assert_eq!(graph_catalog(&[1, 8, 13, 16, 32], None, 1)[0], vec![1, 8, 13, 16, 32]);
    assert_eq!(graph_catalog(&[1, 8, 32], Some(32), 1)[0], vec![32]);
    assert!(graph_catalog(&[1, 8, 32], None, 1)[1].is_empty());
}

#[cfg(test)]
#[test]
fn acceptance_profiles_follow_core_capacity() {
    for capacity in [1, 8, 13, 16, 32, 257] {
        let mut profile = AcceptanceProfile::new(capacity);
        assert_eq!(profile.curve(ProposalSource::Mtp, 0, &geometric_priors(capacity - 1), false).len(), capacity - 1);
        profile.observe(ProposalSource::Mtp, 0, capacity - 1, capacity - 1, false, None);
        assert_eq!(profile.rounds(), u64::from(capacity > 1));
        assert!(profile.prefix_probability(ProposalSource::Mtp, capacity - 1).is_none());
    }
}
