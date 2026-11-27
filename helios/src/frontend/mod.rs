/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native tokenization and rendering over verified Hugging Face assets.

mod deepseek;
mod boundaries;
mod tml;
mod templates;
#[cfg(feature = "images")]
pub mod images;
mod json;
mod jinja;

mod vocabulary;

use std::fs;
use std::path::PathBuf;

use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokenizers::{PostProcessor, Tokenizer};

use crate::{Error, Failure, FrontendError, Result};

pub(crate) fn validate_chat_template(source: &str) -> Result<()> {
    templates::compile(source).map(|_| ()).map_err(|error| {
        Error::frontend(FrontendError::Template(Failure::caused_by(error.to_string(), error)))
    })
}

pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
    pub range: Option<(u64, usize)>,
}

impl Asset {
    pub(crate) fn read(&self) -> Result<Vec<u8>> {
        if !self.path.is_absolute() {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset path must be absolute: {}",
                    self.path.display()
                ),
            ))));
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!("invalid expected SHA256 for {}", self.path.display()),
            ))));
        }
        let bytes = (|| -> std::io::Result<Vec<u8>> {
            if let Some((offset, size)) = self.range {
                use crate::os::FileReadAt;
                let file = fs::File::open(&self.path)?;
                if offset
                    .checked_add(size as u64)
                    .is_none_or(|end| end > file.metadata().map_or(0, |m| m.len()))
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "frontend asset exceeds core",
                    ));
                }
                let mut bytes = vec![0; size];
                file.read_exact_at(&mut bytes, offset)?;
                Ok(bytes)
            } else {
                fs::read(&self.path)
            }
        })()
        .map_err(|error| {
            Error::frontend(FrontendError::Asset(Failure::caused_by(
                format!("cannot read {}: {error}", self.path.display()),
                error,
            )))
        })?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset SHA256 mismatch for {}: expected {}, received {actual}",
                    self.path.display(),
                    self.sha256
                ),
            ))));
        }
        Ok(bytes)
    }
}

pub struct FrontendAssets {
    pub tokenizer: Asset,
    pub tokens: TokenizerSettings,
    pub chat_template: Option<Asset>,
}

#[cfg(any(test, feature = "test-tools"))]
#[path = "../../tests/support/frontend.rs"]
pub mod test_support;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSettings {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Option<TokenList>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenList {
    pub values: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendSettings {
    #[serde(default)]
    pub prompt_encoder: PromptEncoder,
    #[serde(default)]
    pub image_placeholder_token_id: Option<u32>,
    pub tokenizer: TokenizerSettings,
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub template_encoding: TemplateEncoding,
    #[cfg(feature = "images")]
    pub image: Option<images::ImageSettings>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub enum PromptEncoder {
    #[default]
    Jinja = 0,
    DeepseekV4 = 1,
    DeepseekV41 = 2,
    /// TMLv0 message frames: role and content-kind control tokens around ordinary text.
    TmlV0 = 3,
}

impl PromptEncoder {
    /// Whether the encoder renders `tool_choice` and JSON response formats in its
    /// own syntax. Other encoders receive the shared instruction messages instead.
    pub fn renders_tool_choice(self) -> bool {
        matches!(self, Self::DeepseekV4 | Self::DeepseekV41)
    }

    /// Whether the encoder renders failed tool results in its own syntax. Other
    /// encoders receive them prefixed with a text part saying the tool failed.
    pub fn renders_tool_errors(self) -> bool {
        matches!(self, Self::TmlV0)
    }
}

impl From<PromptEncoder> for i32 {
    fn from(value: PromptEncoder) -> Self {
        value as i32
    }
}
impl TryFrom<i32> for PromptEncoder {
    type Error = &'static str;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Jinja),
            1 => Ok(Self::DeepseekV4),
            2 => Ok(Self::DeepseekV41),
            3 => Ok(Self::TmlV0),
            _ => Err("unsupported prompt encoder"),
        }
    }
}

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub enum TemplateEncoding {
    #[default]
    Text = 0,
    TokenSegments = 1,
}

impl From<TemplateEncoding> for i32 {
    fn from(value: TemplateEncoding) -> Self {
        value as i32
    }
}

impl TryFrom<i32> for TemplateEncoding {
    type Error = &'static str;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Text),
            1 => Ok(Self::TokenSegments),
            _ => Err("unsupported chat template encoding"),
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenSegment {
    text: String,
    allow_special: bool,
    #[serde(default)]
    image_index: Option<usize>,
}

fn order_segment_images(
    segments: &[TokenSegment],
    sources: &mut Vec<String>,
    hints: &mut [usize],
) -> Result<()> {
    let order: Vec<_> = segments.iter().filter_map(|segment| segment.image_index).collect();
    if order.is_empty() {
        return Ok(());
    }
    if order.len() != sources.len() {
        return Err(deepseek::invalid("image segment indices must include each image source exactly once"));
    }
    let mut positions = vec![None; sources.len()];
    for (position, &index) in order.iter().enumerate() {
        let Some(slot) = positions.get_mut(index) else {
            return Err(deepseek::invalid("image segment index exceeds the image source count"));
        };
        if slot.replace(position).is_some() {
            return Err(deepseek::invalid("image segment indices must not repeat an image source"));
        }
    }
    for hint in hints {
        *hint = positions[*hint].unwrap();
    }
    let mut original: Vec<_> = std::mem::take(sources).into_iter().map(Some).collect();
    *sources = order.into_iter().map(|index| original[index].take().unwrap()).collect();
    Ok(())
}

fn token_segments(text: &str, hints: &mut Vec<usize>) -> serde_json::Result<Vec<TokenSegment>> {
    if hints.is_empty() {
        return serde_json::from_str(text);
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawSegment<'a> {
        #[serde(borrow)]
        text: &'a serde_json::value::RawValue,
        allow_special: bool,
        #[serde(default)]
        image_index: Option<usize>,
    }
    let raw_segments: Vec<RawSegment<'_>> = serde_json::from_str(text)?;
    let mut segments = Vec::with_capacity(raw_segments.len());
    let mut positions = Vec::new();
    let mut byte_start = 0;
    for segment in raw_segments {
        let raw = segment.text.get();
        let start = raw.as_ptr() as usize - text.as_ptr() as usize;
        let decoded: String = serde_json::from_str(raw)?;
        for &hint in hints.iter() {
            if hint > start && hint < start + raw.len() {
                // The hint ends inside this JSON string, before its closing quote.
                let mut prefix = raw[..hint - start].to_owned();
                prefix.push('"');
                let prefix: String = serde_json::from_str(&prefix)?;
                positions.push(byte_start + prefix.len());
            }
        }
        byte_start += decoded.len();
        segments.push(TokenSegment {
            text: decoded,
            allow_special: segment.allow_special,
            image_index: segment.image_index,
        });
    }
    *hints = positions;
    Ok(segments)
}

pub struct RenderedPrompt {
    pub image_sources: Vec<String>,
    /// Indices in image_sources whose expanded spans end at requested checkpoints.
    pub image_checkpoint_hints: Vec<usize>,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub message_boundaries: Vec<usize>,
    /// Sorted, disjoint byte ranges of `text` that came from the request. A token whose bytes
    /// overlap one is request text; an output marker is skipped only when one range holds all of
    /// it. Empty for prompts that are not rendered from a Jinja text template.
    pub ordinary: Vec<std::ops::Range<usize>>,
}

/// Whether `ordinary`, sorted and disjoint, overlaps the byte range `start..end`.
pub(crate) fn overlaps_ordinary(ordinary: &[std::ops::Range<usize>], start: usize, end: usize) -> bool {
    let first = ordinary.partition_point(|range| range.end <= start);
    ordinary.get(first).is_some_and(|range| range.start < end)
}

/// Whether one range of `ordinary` holds all of the byte range `start..end`. Ranges are sorted,
/// disjoint and merged where they touch, so a match that the request spelled completely lies in
/// one range, and a match with any template-written byte does not.
#[cfg(feature = "server")]
pub(crate) fn within_ordinary(ordinary: &[std::ops::Range<usize>], start: usize, end: usize) -> bool {
    let first = ordinary.partition_point(|range| range.end < end);
    ordinary.get(first).is_some_and(|range| range.start <= start)
}

/// Whether `token` occurs in a string of `value`, nested arrays and objects included.
fn json_names(value: &Json, token: &str) -> bool {
    match value {
        Json::String(text) => text.contains(token),
        Json::Array(items) => items.iter().any(|item| json_names(item, token)),
        Json::Object(map) => map.values().any(|item| json_names(item, token)),
        _ => false,
    }
}

const PARALLEL_ENCODE_MIN_BYTES: usize = 32 * 1024;
const PARALLEL_ENCODE_TARGET_CHUNK_BYTES: usize = 16 * 1024;
const PARALLEL_ENCODE_MIN_CHUNK_BYTES: usize = 4 * 1024;
const PARALLEL_ENCODE_SEARCH_WINDOW_BYTES: usize = 8 * 1024;
const PARALLEL_ENCODE_MAX_THREADS: usize = 16;
const PARALLEL_ENCODE_MIN_STRADDLE_BYTES: usize = 256;
const PARALLEL_ENCODE_MAX_VERIFY_WHITESPACE_BYTES: usize = 4 * 1024;

fn default_parallel_encode_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(PARALLEL_ENCODE_MAX_THREADS)
}

/// Shared non-blocking thread budget bounding concurrent parallel tokenization
/// workers across requests sharing a [`Frontend`]. Each active encode call
/// accounts for its own calling thread (1) and may claim helper threads only
/// while total active tokenization threads stay within `max_threads`.
struct EncodeThreadBudget {
    max_threads: usize,
    active_threads: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    active_helpers: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    peak_helpers: std::sync::atomic::AtomicUsize,
}

struct EncodeThreadPermit<'a> {
    budget: &'a EncodeThreadBudget,
    held: usize,
}

impl EncodeThreadPermit<'_> {
    fn threads(&self) -> usize {
        self.held
    }

    fn shrink_to(&mut self, threads: usize) {
        debug_assert!(threads >= 1 && threads <= self.held);
        let release = self.held - threads;
        if release > 0 {
            self.budget
                .active_threads
                .fetch_sub(release, std::sync::atomic::Ordering::AcqRel);
            self.held = threads;
        }
    }
}

impl Drop for EncodeThreadPermit<'_> {
    fn drop(&mut self) {
        if self.held > 0 {
            self.budget
                .active_threads
                .fetch_sub(self.held, std::sync::atomic::Ordering::AcqRel);
        }
    }
}

#[cfg(test)]
struct ActiveHelperGuard<'a>(&'a EncodeThreadBudget);

#[cfg(test)]
impl Drop for ActiveHelperGuard<'_> {
    fn drop(&mut self) {
        self.0
            .active_helpers
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl EncodeThreadBudget {
    fn new(max_threads: usize) -> Self {
        Self {
            max_threads: max_threads.clamp(1, PARALLEL_ENCODE_MAX_THREADS),
            active_threads: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            active_helpers: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            peak_helpers: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn max_threads(&self) -> usize {
        self.max_threads
    }

    fn acquire(&self, desired_threads: usize) -> EncodeThreadPermit<'_> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let desired = desired_threads.clamp(1, self.max_threads);
        let mut current = self.active_threads.load(Acquire);
        loop {
            let helpers = self
                .max_threads
                .saturating_sub(current.saturating_add(1))
                .min(desired - 1);
            let claim = 1 + helpers;
            match self.active_threads.compare_exchange_weak(
                current,
                current + claim,
                AcqRel,
                Acquire,
            ) {
                Ok(_) => {
                    return EncodeThreadPermit {
                        budget: self,
                        held: claim,
                    };
                }
                Err(actual) => current = actual,
            }
        }
    }

    #[cfg(test)]
    fn active_threads(&self) -> usize {
        self.active_threads.load(std::sync::atomic::Ordering::Acquire)
    }

    #[cfg(test)]
    fn peak_helpers(&self) -> usize {
        self.peak_helpers.load(std::sync::atomic::Ordering::Acquire)
    }

    #[cfg(test)]
    fn enter_helper(&self) -> ActiveHelperGuard<'_> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let now = self.active_helpers.fetch_add(1, AcqRel) + 1;
        let mut peak = self.peak_helpers.load(Acquire);
        while now > peak {
            match self
                .peak_helpers
                .compare_exchange_weak(peak, now, AcqRel, Acquire)
            {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
        ActiveHelperGuard(self)
    }
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

fn normalizer_is_noop(normalizer: Option<&tokenizers::NormalizerWrapper>) -> bool {
    match normalizer {
        None => true,
        Some(tokenizers::NormalizerWrapper::Sequence(seq)) => seq.as_ref().is_empty(),
        _ => false,
    }
}

fn pre_tokenizer_is_chunk_safe(pre: Option<&tokenizers::PreTokenizerWrapper>) -> bool {
    use tokenizers::PreTokenizerWrapper as P;
    match pre {
        None => false,
        Some(P::ByteLevel(bl)) => !bl.add_prefix_space,
        Some(P::Split(_))
        | Some(P::Whitespace(_))
        | Some(P::WhitespaceSplit(_))
        | Some(P::Delimiter(_))
        | Some(P::Punctuation(_))
        | Some(P::Digits(_))
        | Some(P::UnicodeScripts(_))
        | Some(P::BertPreTokenizer(_)) => true,
        Some(P::Metaspace(m)) => {
            m.prepend_scheme == tokenizers::pre_tokenizers::metaspace::PrependScheme::Never
        }
        Some(P::Sequence(seq)) => {
            !seq.as_ref().is_empty()
                && seq
                    .as_ref()
                    .iter()
                    .all(|child| pre_tokenizer_is_chunk_safe(Some(child)))
        }
        Some(P::FixedLength(_)) => false,
    }
}

fn tokenizer_is_parallel_eligible(tokenizer: &Tokenizer, add_special_tokens: bool) -> bool {
    if !normalizer_is_noop(tokenizer.get_normalizer())
        || !pre_tokenizer_is_chunk_safe(tokenizer.get_pre_tokenizer())
    {
        return false;
    }
    if add_special_tokens
        && tokenizer
            .get_post_processor()
            .is_some_and(|p| p.added_tokens(false) > 0)
    {
        return false;
    }
    true
}

/// Verifies that splitting `text` at `cut` produces identical token IDs and byte
/// offsets to sequential encoding over a straddling window.
///
/// The window extends past any contiguous Unicode whitespace run touching `cut`
/// on either side (up to `PARALLEL_ENCODE_MAX_VERIFY_WHITESPACE_BYTES`) plus
/// `straddle_half` bytes beyond the whitespace run, so added tokens with
/// `lstrip`/`rstrip` or whitespace-spanning pre-tokenizers cannot interact
/// across `cut` outside the verified slice. Cuts touching a whitespace run
/// longer than that bound are rejected.
fn verify_split_boundary(
    tokenizer: &Tokenizer,
    text: &str,
    cut: usize,
    prev_cut: usize,
    straddle_half: usize,
) -> bool {
    if cut <= prev_cut || cut >= text.len() || !text.is_char_boundary(cut) {
        return false;
    }
    let mut left_ws_start = cut;
    for ch in text[prev_cut..cut].chars().rev() {
        if !ch.is_whitespace() {
            break;
        }
        left_ws_start -= ch.len_utf8();
        if cut - left_ws_start > PARALLEL_ENCODE_MAX_VERIFY_WHITESPACE_BYTES {
            return false;
        }
    }
    if prev_cut > 0 && left_ws_start.saturating_sub(straddle_half) < prev_cut {
        return false;
    }
    let mut right_ws_end = cut;
    for ch in text[cut..].chars() {
        if !ch.is_whitespace() {
            break;
        }
        right_ws_end += ch.len_utf8();
        if right_ws_end - cut > PARALLEL_ENCODE_MAX_VERIFY_WHITESPACE_BYTES {
            return false;
        }
    }
    let win_start = floor_char_boundary(text, left_ws_start.saturating_sub(straddle_half));
    let win_end = ceil_char_boundary(
        text,
        right_ws_end.saturating_add(straddle_half).min(text.len()),
    );
    if win_start >= cut || win_end <= cut {
        return false;
    }
    let (Ok(full), Ok(left), Ok(right)) = (
        tokenizer.encode(&text[win_start..win_end], false),
        tokenizer.encode(&text[win_start..cut], false),
        tokenizer.encode(&text[cut..win_end], false),
    ) else {
        return false;
    };
    let (l_ids, r_ids, f_ids) = (left.get_ids(), right.get_ids(), full.get_ids());
    if l_ids.len() < 2 || r_ids.len() < 2 || l_ids.len() + r_ids.len() != f_ids.len() {
        return false;
    }
    if !l_ids.iter().chain(r_ids.iter()).eq(f_ids.iter()) {
        return false;
    }
    let rel_cut = cut - win_start;
    left.get_offsets()
        .iter()
        .copied()
        .chain(
            right
                .get_offsets()
                .iter()
                .map(|&(start, end)| (rel_cut + start, rel_cut + end)),
        )
        .eq(full.get_offsets().iter().copied())
}

/// Selects verified UTF-8 split ranges for up to `max_threads` concurrent workers.
///
/// Targets are spaced at least `PARALLEL_ENCODE_TARGET_CHUNK_BYTES` (16 KiB)
/// apart. Each candidate cut is searched within `PARALLEL_ENCODE_SEARCH_WINDOW_BYTES`
/// (8 KiB) after its target and kept at least `PARALLEL_ENCODE_MIN_CHUNK_BYTES`
/// (4 KiB) from the previous cut and from the end of `text`, so interior chunks
/// can be as small as ~8 KiB and every chunk is at least 4 KiB.
fn find_parallel_chunks_with_threads(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
    max_threads: usize,
) -> Option<Vec<std::ops::Range<usize>>> {
    if text.len() < PARALLEL_ENCODE_MIN_BYTES
        || !tokenizer_is_parallel_eligible(tokenizer, add_special_tokens)
    {
        return None;
    }
    let threads = max_threads
        .min(PARALLEL_ENCODE_MAX_THREADS)
        .min(text.len() / PARALLEL_ENCODE_TARGET_CHUNK_BYTES);
    if threads < 2 {
        return None;
    }
    let max_added_len = tokenizer
        .get_added_vocabulary()
        .get_vocab()
        .keys()
        .map(String::len)
        .max()
        .unwrap_or(0);
    // Reserve space beyond the longest added token so single-word boundary checks
    // (one character before/after the match) also fit inside the verified window.
    let straddle_half = max_added_len
        .saturating_add(16)
        .max(PARALLEL_ENCODE_MIN_STRADDLE_BYTES);
    let bytes = text.as_bytes();
    let mut cuts = Vec::with_capacity(threads - 1);
    let mut prev_cut = 0usize;
    for i in 1..threads {
        let target = (text.len() * i) / threads;
        let lo = target.max(prev_cut + PARALLEL_ENCODE_MIN_CHUNK_BYTES);
        let hi = (target + PARALLEL_ENCODE_SEARCH_WINDOW_BYTES)
            .min(text.len().saturating_sub(PARALLEL_ENCODE_MIN_CHUNK_BYTES));
        if lo >= hi {
            continue;
        }
        // Prefer splitting right after a newline run; fall back to before an ASCII space
        // preceded by non-whitespace for single-line payloads (e.g. minified JSON).
        let mut candidate = None;
        for pos in lo..hi {
            if bytes[pos - 1] == b'\n'
                && bytes[pos] != b'\n'
                && bytes[pos] != b'\r'
                && text.is_char_boundary(pos)
            {
                candidate = Some(pos);
                break;
            }
        }
        if candidate.is_none() {
            for pos in lo..hi {
                if bytes[pos] == b' '
                    && text.is_char_boundary(pos)
                    && !text[..pos].chars().next_back().is_some_and(char::is_whitespace)
                {
                    candidate = Some(pos);
                    break;
                }
            }
        }
        if let Some(cut) = candidate
            && verify_split_boundary(tokenizer, text, cut, prev_cut, straddle_half)
        {
            cuts.push(cut);
            prev_cut = cut;
        }
    }
    if cuts.is_empty() {
        return None;
    }
    let mut ranges = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for cut in cuts {
        ranges.push(start..cut);
        start = cut;
    }
    ranges.push(start..text.len());
    Some(ranges)
}

fn map_tokenizer_error(error: tokenizers::Error) -> Error {
    Error::frontend(FrontendError::Tokenizer(Failure::boxed(
        error.to_string(),
        error,
    )))
}

fn encode_with_budget(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
    budget: &EncodeThreadBudget,
) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
    let desired = text.len() / PARALLEL_ENCODE_TARGET_CHUNK_BYTES;
    if text.len() < PARALLEL_ENCODE_MIN_BYTES
        || desired < 2
        || budget.max_threads() < 2
        || !tokenizer_is_parallel_eligible(tokenizer, add_special_tokens)
    {
        let encoding = tokenizer
            .encode(text, add_special_tokens)
            .map_err(map_tokenizer_error)?;
        return Ok((encoding.get_ids().to_vec(), encoding.get_offsets().to_vec()));
    }
    let mut permit = budget.acquire(desired);
    let Some(ranges) =
        find_parallel_chunks_with_threads(tokenizer, text, add_special_tokens, permit.threads())
    else {
        permit.shrink_to(1);
        let encoding = tokenizer
            .encode(text, add_special_tokens)
            .map_err(map_tokenizer_error)?;
        return Ok((encoding.get_ids().to_vec(), encoding.get_offsets().to_vec()));
    };
    permit.shrink_to(ranges.len());
    let encode_chunk = |range: std::ops::Range<usize>| -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let start = range.start;
        let encoding = tokenizer
            .encode(&text[range], false)
            .map_err(map_tokenizer_error)?;
        let ids = encoding.get_ids().to_vec();
        let offsets = encoding
            .get_offsets()
            .iter()
            .map(|&(a, b)| (start + a, start + b))
            .collect();
        Ok((ids, offsets))
    };
    let parts = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(ranges.len() - 1);
        for range in ranges[1..].iter().cloned() {
            handles.push(scope.spawn(move || {
                #[cfg(test)]
                let _helper = budget.enter_helper();
                encode_chunk(range)
            }));
        }
        let mut results = Vec::with_capacity(ranges.len());
        results.push(encode_chunk(ranges[0].clone()));
        for handle in handles {
            results.push(handle.join().unwrap());
        }
        results
    });
    let mut total = 0usize;
    let mut decoded = Vec::with_capacity(parts.len());
    for part in parts {
        let (ids, offsets) = part?;
        total += ids.len();
        decoded.push((ids, offsets));
    }
    let mut ids = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(total);
    for (part_ids, part_offsets) in decoded {
        ids.extend_from_slice(&part_ids);
        offsets.extend_from_slice(&part_offsets);
    }
    Ok((ids, offsets))
}

pub struct Frontend {
    prompt_encoder: PromptEncoder,
    image_placeholder_token_id: Option<u32>,
    tokenizer: Tokenizer,
    /// The tokenizer without its added vocabulary, for text that must not spell controls.
    ordinary_tokenizer: Tokenizer,
    /// The added tokens that request text may not spell: the special tokens, the ones the chat
    /// template or the image settings name, and the ones the prepared syntax names (see
    /// `with_controls`). Other added tokens, such as `/nothink`, are lexical: request text that
    /// spells one tokenizes to that token.
    controls: std::collections::HashSet<u32>,
    /// The control that request text positions images with, when the image layout allows it.
    spelled_image_placeholder: Option<u32>,
    template_encoding: TemplateEncoding,
    template: Option<jinja::CompiledTemplate>,
    template_source: String,
    bos_token_id: Option<u32>,
    effort_levels: std::sync::OnceLock<Vec<String>>,
    vocabulary_bytes: std::sync::OnceLock<Result<vocabulary::VocabularyBytes>>,
    inline_system_messages: std::sync::OnceLock<bool>,
    defaults: Map<String, Json>,
    encode_threads: EncodeThreadBudget,
}

impl Frontend {
    pub fn prompt_encoder(&self) -> PromptEncoder {
        self.prompt_encoder
    }
    pub fn with_template_encoding(mut self, encoding: TemplateEncoding) -> Self {
        self.template_encoding = encoding;
        self
    }
    #[cfg(any(test, feature = "server"))]
    pub(crate) fn max_encode_threads(&self) -> usize {
        self.encode_threads.max_threads()
    }
    #[cfg(any(test, feature = "server"))]
    pub(crate) fn with_max_encode_threads(mut self, max_threads: usize) -> Self {
        self.encode_threads = EncodeThreadBudget::new(max_threads);
        self
    }
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Resolves advisory effort against the vocabulary accepted by the prepared template.
    pub fn resolve_effort<'a>(&self, candidates: &'a [&str]) -> Result<&'a str> {
        if self.prompt_encoder == PromptEncoder::TmlV0 {
            return candidates
                .iter()
                .copied()
                .find(|level| tml::effort_level(level))
                .ok_or_else(|| deepseek::invalid("unsupported reasoning effort"));
        }
        if self.prompt_encoder != PromptEncoder::Jinja {
            return candidates
                .iter()
                .copied()
                .find(|level| matches!(*level, "low" | "high" | "xhigh" | "max"))
                .ok_or_else(|| deepseek::invalid("unsupported reasoning effort"));
        }
        let accepted = self.effort_levels.get_or_init(|| {
            let mut context = self.defaults.clone();
            context.insert(
                "messages".into(),
                serde_json::json!([{"role":"user","content":"Hello"}]),
            );
            context.insert("enable_thinking".into(), true.into());
            context.insert("thinking_mode".into(), "thinking".into());
            context.insert("add_generation_prompt".into(), true.into());
            context.insert("continue_final_message".into(), false.into());
            ["low", "medium", "high", "xhigh", "max"]
                .into_iter()
                .filter(|effort| {
                    context.insert("reasoning_effort".into(), (*effort).into());
                    self.template
                        .as_ref()
                        .unwrap()
                        .render(&context, &[])
                        .is_ok()
                })
                .map(str::to_owned)
                .collect()
        });
        candidates
            .iter()
            .copied()
            .find(|candidate| accepted.iter().any(|level| level == candidate))
            .ok_or_else(|| {
                Error::admission(
                    crate::AdmissionError::Feature,
                    "prepared chat template does not accept the requested reasoning effort",
                )
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn template_default(&self, name: &str) -> Option<&Json> {
        self.defaults.get(name)
    }

    pub fn normalize_inline_system_messages(&self, messages: &mut [Json]) -> Result<()> {
        if self.prompt_encoder != PromptEncoder::Jinja {
            return Ok(());
        }
        if !messages.iter().skip(1).any(|m| m["role"] == "system") {
            return Ok(());
        }
        let supported = self.inline_system_messages.get_or_init(|| {
            let marker = "FLARE_INLINE_SYSTEM_PROBE";
            let mut context = self.defaults.clone();
            context.insert(
                "messages".into(),
                serde_json::json!([
                    {"role":"system","content":"Instructions"},
                    {"role":"user","content":"Hello"},
                    {"role":"system","content":marker},
                    {"role":"user","content":"Continue"}
                ]),
            );
            context.insert("add_generation_prompt".into(), false.into());
            context.insert("continue_final_message".into(), false.into());
            self.template
                .as_ref()
                .unwrap()
                .render(&context, &[])
                .is_ok_and(|rendered| rendered.text.contains(marker))
        });
        if *supported {
            return Ok(());
        }
        for message in messages.iter_mut().skip(1) {
            if message["role"] != "system" {
                continue;
            }
            let invalid = || {
                Error::frontend(FrontendError::Input(Failure::new(
                    "inline system messages require text content",
                )))
            };
            let text = match &message["content"] {
                Json::String(text) => text.clone(),
                Json::Array(parts) => parts
                    .iter()
                    .map(|part| {
                        if part["type"] != "text" {
                            return Err(invalid());
                        }
                        part["text"].as_str().ok_or_else(invalid)
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join("\n\n"),
                _ => return Err(invalid()),
            };
            message["role"] = "user".into();
            message["content"] = format!("<system>{text}</system>").into();
        }
        Ok(())
    }

    pub fn load(assets: FrontendAssets, template_defaults: Map<String, Json>) -> Result<Self> {
        Self::load_encoder(assets, template_defaults, PromptEncoder::Jinja, None)
    }

    pub fn load_with_settings(
        assets: FrontendAssets,
        template_defaults: Map<String, Json>,
        settings: &FrontendSettings,
    ) -> Result<Self> {
        if settings.prompt_encoder != PromptEncoder::Jinja && assets.chat_template.is_some() {
            return Err(deepseek::invalid(
                "official prompt encoding does not use a chat template asset",
            ));
        }
        if settings.prompt_encoder != PromptEncoder::Jinja
            && matches!(settings.template_encoding, TemplateEncoding::TokenSegments)
        {
            return Err(deepseek::invalid(
                "official prompt encoding requires text template encoding",
            ));
        }
        Self::load_encoder(
            assets,
            template_defaults,
            settings.prompt_encoder,
            settings.image_placeholder_token_id,
        )
        .map(|frontend| {
            frontend
                .with_template_encoding(settings.template_encoding)
                .with_image_placeholder_text(settings)
        })
    }

    /// Records the token that request text may spell to position images.
    fn with_image_placeholder_text(mut self, settings: &FrontendSettings) -> Self {
        #[cfg(feature = "images")]
        let text = settings
            .image
            .as_ref()
            .and_then(images::ImageSettings::spelled_placeholder);
        #[cfg(not(feature = "images"))]
        let text = {
            let _ = settings;
            None::<&str>
        };
        self.spelled_image_placeholder = text.and_then(|text| self.tokenizer.token_to_id(text));
        self.controls.extend(self.spelled_image_placeholder);
        self
    }

    /// Adds the tokens that the prepared syntax names as markers (reasoning and tool-call
    /// delimiters, image tokens) to the controls. Such a marker is a control even when the
    /// tokenizer does not mark it special and the chat template does not name it. Tokens outside
    /// the added vocabulary, and whitespace, are ignored.
    pub fn with_controls(mut self, tokens: impl IntoIterator<Item = u32>) -> Self {
        let added = self.tokenizer.get_added_vocabulary().get_added_tokens_decoder();
        self.controls.extend(
            tokens
                .into_iter()
                .filter(|id| added.get(id).is_some_and(|token| !token.content.trim().is_empty())),
        );
        self
    }

    fn load_encoder(
        assets: FrontendAssets,
        template_defaults: Map<String, Json>,
        prompt_encoder: PromptEncoder,
        image_placeholder_token_id: Option<u32>,
    ) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_bytes(assets.tokenizer.read()?).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                format!("invalid tokenizer: {error}"),
                error,
            )))
        })?;
        tokenizer.with_truncation(None).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        tokenizer.with_padding(None);
        boundaries::preserve_offsets(&mut tokenizer);
        if prompt_encoder != PromptEncoder::Jinja
            && image_placeholder_token_id.is_some_and(|id| tokenizer.id_to_token(id).is_none())
        {
            return Err(deepseek::invalid(
                "image placeholder token ID is absent from the checkpoint tokenizer",
            ));
        }
        let template_bytes = match assets.chat_template {
            Some(asset) if prompt_encoder == PromptEncoder::Jinja => asset.read()?,
            _ if prompt_encoder != PromptEncoder::Jinja => Vec::new(),
            _ => {
                return Err(deepseek::invalid(
                    "Jinja prompt encoding requires a chat template asset",
                ));
            }
        };
        let template_source = String::from_utf8(template_bytes).map_err(|error| {
            Error::frontend(FrontendError::Template(Failure::caused_by(
                format!("chat template is not UTF-8: {error}"),
                error,
            )))
        })?;
        let template = if prompt_encoder == PromptEncoder::Jinja {
            Some(templates::compile(&template_source).map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(error.to_string(), error)))
            })?)
        } else { None };
        let bos_token_id = assets
            .tokens
            .bos_token
            .as_deref()
            .and_then(|token| tokenizer.token_to_id(token));
        let mut defaults = Map::new();
        for (name, value) in [
            ("bos_token", assets.tokens.bos_token),
            ("eos_token", assets.tokens.eos_token),
            ("unk_token", assets.tokens.unk_token),
            ("sep_token", assets.tokens.sep_token),
            ("pad_token", assets.tokens.pad_token),
            ("cls_token", assets.tokens.cls_token),
            ("mask_token", assets.tokens.mask_token),
        ] {
            if let Some(value) = value {
                defaults.insert(name.into(), Json::String(value));
            }
        }
        if let Some(tokens) = assets.tokens.additional_special_tokens {
            defaults.insert(
                "additional_special_tokens".into(),
                Json::Array(tokens.values.into_iter().map(Json::String).collect()),
            );
        }
        defaults.insert(
            "add_generation_prompt".into(),
            Json::Bool(prompt_encoder != PromptEncoder::Jinja),
        );
        defaults.insert("tools".into(), Json::Null);
        defaults.insert("documents".into(), Json::Null);
        defaults.extend(template_defaults);
        let mut ordinary_tokenizer = tokenizer.clone();
        ordinary_tokenizer.with_added_vocabulary(tokenizers::AddedVocabulary::new());
        // A token the template names in its text or writes through a default (a special-token
        // string of the tokenizer config, a recipe default, nested ones included) is a control.
        // Whitespace alone is never a marker, however often a template writes it.
        let controls = tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, token)| {
                token.special
                    || (!token.content.trim().is_empty()
                        && (template_source.contains(&token.content)
                            || defaults.values().any(|value| json_names(value, &token.content))))
            })
            .map(|(&id, _)| id)
            .chain(image_placeholder_token_id)
            .collect();
        let frontend = Self {
            prompt_encoder,
            image_placeholder_token_id,
            bos_token_id,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            tokenizer,
            ordinary_tokenizer,
            controls,
            spelled_image_placeholder: None,
            template_encoding: TemplateEncoding::Text,
            template,
            template_source,
            defaults,
            encode_threads: EncodeThreadBudget::new(default_parallel_encode_threads()),
        };
        match prompt_encoder {
            PromptEncoder::Jinja => {}
            PromptEncoder::TmlV0 => tml::validate_defaults(&frontend)?,
            PromptEncoder::DeepseekV4 | PromptEncoder::DeepseekV41 => deepseek::validate_defaults(&frontend)?,
        }
        Ok(frontend)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        Ok(encode_with_budget(&self.tokenizer, text, add_special_tokens, &self.encode_threads)?.0)
    }

    /// Encodes `text` without the added vocabulary, so a spelled control stays ordinary text.
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        Ok(encode_with_budget(&self.ordinary_tokenizer, text, false, &self.encode_threads)?.0)
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })
    }

    pub fn render_chat(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        match self.prompt_encoder {
            PromptEncoder::Jinja => {}
            PromptEncoder::TmlV0 => return tml::render(self, messages, options),
            PromptEncoder::DeepseekV4 | PromptEncoder::DeepseekV41 => {
                return deepseek::render(self, messages, options);
            }
        }
        if messages.is_empty() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "Cannot apply chat template to an empty conversation. Provide at least one message.",
            ))));
        }
        self.render_jinja(messages, options)
    }

    fn render_jinja(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        let mut context = self.defaults.clone();
        context.extend(options.clone());
        let continuation = match context.remove("continue_final_message") {
            None | Some(Json::Null | Json::Bool(false)) => None,
            Some(Json::Bool(true)) => Some("content".to_owned()),
            Some(Json::String(field)) if field.is_empty() => None,
            Some(Json::String(field)) => Some(field),
            _ => {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "continue_final_message must be a boolean or field name",
                ))));
            }
        };
        if continuation.is_some() && context.get("add_generation_prompt") == Some(&Json::Bool(true))
        {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead.",
            ))));
        }
        if let Some(tools) = context.get("tools").filter(|value| !value.is_null()) {
            if !tools
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Tools should either be a JSON schema, or a callable function with type hints and a docstring suitable for auto-conversion to a schema.",
                ))));
            }
        }
        if let Some(documents) = context.get("documents").filter(|value| !value.is_null()) {
            if !documents
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Documents should be a list of dicts with 'title' and 'text' keys!",
                ))));
            }
        }
        let mut messages = messages.to_vec();
        let mut image_sources = Vec::new();
        let mut image_checkpoint_hints = Vec::new();
        let mut checkpoint_hints = Vec::new();
        for (index, message) in messages.iter_mut().enumerate() {
            if !message.is_object() {
                return Err(deepseek::invalid("messages must contain objects"));
            }
            if let Some(Json::Array(hints)) =
                message.as_object_mut().unwrap().remove("_checkpoint_hints")
            {
                checkpoint_hints.extend(
                    hints
                        .iter()
                        .filter_map(|hint| hint.as_u64().map(|count| (index, count as usize))),
                );
            }
            if let Some(parts) = message.get_mut("content").and_then(Json::as_array_mut) {
                for (part_index, part) in parts.iter_mut().enumerate() {
                    let kind = part
                        .get("type")
                        .and_then(Json::as_str)
                        .ok_or_else(|| deepseek::invalid("message content part requires type"))?;
                    if kind == "text" && !part.get("text").is_some_and(Json::is_string) {
                        return Err(deepseek::invalid("text content must be a string"));
                    }
                    if kind == "image_url" {
                        let url = part["image_url"]["url"]
                            .as_str()
                            .ok_or_else(|| deepseek::invalid("image_url.url must be a string"))?;
                        if checkpoint_hints.contains(&(index, part_index + 1)) {
                            image_checkpoint_hints.push(image_sources.len());
                        }
                        let image_index = image_sources.len();
                        image_sources.push(url.to_owned());
                        *part = serde_json::json!({"type":"image", "image_index":image_index});
                    }
                }
            }
        }
        let final_text = continuation
            .map(|field| self.mark_continuation(&mut messages, &field))
            .transpose()?;
        context.insert("messages".into(), Json::Array(messages));
        let default_names: Vec<_> = self
            .defaults
            .keys()
            .filter(|name| !options.contains_key(*name) && name.as_str() != "messages")
            .map(String::as_str)
            .collect();
        let rendered = self
            .template
            .as_ref()
            .unwrap()
            .render_with_defaults(&context, &checkpoint_hints, &default_names)
            .map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    error.message.clone(),
                    error,
                )))
            })?;
        let ordinary = rendered.ordinary;
        let marks = rendered.marks;
        let rendered = rendered.text;
        let mut hint_bytes = marks
            .iter()
            .filter_map(|mark| match mark {
                jinja::Mark::Hint(byte) => Some(*byte),
                _ => None,
            })
            .collect();
        let segmented = matches!(self.template_encoding, TemplateEncoding::TokenSegments);
        let mut segments: Vec<TokenSegment> = if segmented {
            token_segments(&rendered, &mut hint_bytes).map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    "chat template must render an array of text and allow_special segments",
                    error,
                )))
            })?
        } else {
            vec![TokenSegment {
                text: rendered,
                allow_special: true,
                image_index: None,
            }]
        };
        order_segment_images(&segments, &mut image_sources, &mut image_checkpoint_hints)?;
        let mut segment_ends = Vec::with_capacity(segments.len() + 1);
        segment_ends.push(0usize);
        for segment in &segments {
            segment_ends.push(segment_ends.last().unwrap() + segment.text.len());
        }
        let mut byte_boundaries = Vec::new();
        for mark in marks {
            match mark {
                jinja::Mark::Hint(_) => {}
                jinja::Mark::Byte(byte) if !segmented && byte <= *segment_ends.last().unwrap() => {
                    byte_boundaries.push(byte);
                }
                jinja::Mark::Position(count) if segmented => {
                    byte_boundaries.push(*segment_ends.get(count).ok_or_else(|| {
                        deepseek::invalid("mark position exceeds the rendered segment count")
                    })?);
                }
                _ => {
                    return Err(deepseek::invalid(
                        "text templates require byte marks; segmented templates require segment-count marks",
                    ));
                }
            }
        }
        let mut text: String = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect();
        if let Some(final_text) = final_text {
            let marker = "CONTINUE_FINAL_MESSAGE_TAG";
            let index = text.rfind(marker).filter(|_| text.contains(final_text.trim())).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but the final message does not appear in the chat after applying the chat template!"))))?;
            let preserved_space = text[index + marker.len()..].starts_with(' ');
            text.truncate(index);
            if !preserved_space {
                text.truncate(text.trim_end().len());
            }
        }
        let mut remaining = text.len();
        let mut token_ids = Vec::new();
        let mut offsets = Vec::new();
        let mut byte_start = 0;
        for segment in &mut segments {
            segment.text.truncate(remaining.min(segment.text.len()));
            remaining -= segment.text.len();
            let tokenizer = if segment.allow_special {
                &self.tokenizer
            } else {
                &self.ordinary_tokenizer
            };
            let (seg_ids, seg_offsets) = encode_with_budget(
                tokenizer,
                &segment.text,
                false,
                &self.encode_threads,
            )?;
            let first = token_ids.len();
            if segmented {
                token_ids.extend_from_slice(&seg_ids);
                offsets.extend_from_slice(&seg_offsets);
            } else {
                self.push_segment(
                    &segment.text,
                    &ordinary,
                    &seg_ids,
                    &seg_offsets,
                    image_sources.len(),
                    &mut token_ids,
                    &mut offsets,
                )?;
            }
            for span in &mut offsets[first..] {
                *span = (byte_start + span.0, byte_start + span.1);
            }
            byte_start += segment.text.len();
            if remaining == 0 {
                break;
            }
        }
        let message_boundaries =
            boundaries::token_ends(&byte_boundaries, &hint_bytes, &offsets, text.len())?;
        Ok(RenderedPrompt {
            message_boundaries,
            text,
            token_ids,
            image_sources,
            image_checkpoint_hints,
            // Segmented text is not the template output the ranges index.
            ordinary: if segmented { Vec::new() } else { ordinary },
        })
    }

    /// Appends the tokens of a prompt segment to `ids_out` and `offsets_out`, with byte spans
    /// relative to `text`, so that request text spelling a control stays ordinary. `ids` and
    /// `offsets` are the segment encoded with the added vocabulary. Added tokens other than
    /// spelled controls (template controls and lexical tokens such as `/nothink`) delimit
    /// regions; only a region that holds a spelled control is encoded again, without the added
    /// vocabulary.
    ///
    /// That second encoding allocates host memory after the host cache is sized. The memory draws
    /// on the headroom `host_pool_budget` (selene/src/offload.rs) leaves unallocated:
    /// `HOST_SERVICE_HEADROOM_BYTES` (2 GiB) plus one twentieth of total memory.
    fn push_segment(
        &self,
        text: &str,
        ordinary: &[std::ops::Range<usize>],
        ids: &[u32],
        offsets: &[(usize, usize)],
        images: usize,
        ids_out: &mut Vec<u32>,
        offsets_out: &mut Vec<(usize, usize)>,
    ) -> Result<()> {
        let added = self
            .tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder();
        // An added token with lstrip or rstrip also covers adjacent whitespace, which may be
        // request text. Only the token's own bytes decide whether the request spelled it.
        let spelled = |at: usize| {
            let (start, end) = offsets[at];
            let own = added.get(&ids[at]).and_then(|token| {
                let found = text.get(start..end)?.find(token.content.as_str())?;
                Some(start + found..start + found + token.content.len())
            });
            let own = own.unwrap_or(start..end);
            overlaps_ordinary(ordinary, own.start, own.end)
        };
        // The request positions images by spelling their placeholder when the spelled and
        // template-written placeholders together account for every attached image.
        let positional = self.spelled_image_placeholder.filter(|id| {
            let requested = (0..ids.len()).filter(|&at| ids[at] == *id && spelled(at)).count();
            let written = ids.iter().filter(|token| *token == id).count() - requested;
            requested > 0 && written + requested == images
        });
        // A token ends a region when it is a template control or a lexical added token, or the
        // image placeholder that positions an attached image.
        let closes = |at: usize| {
            let id = ids[at];
            (added.contains_key(&id) && !(self.controls.contains(&id) && spelled(at)))
                || positional == Some(id)
        };
        let (mut from, mut from_byte, mut dirty) = (0, 0, false);
        for at in 0..=ids.len() {
            if at < ids.len() && !closes(at) {
                dirty |= self.controls.contains(&ids[at]) && added.contains_key(&ids[at]);
                continue;
            }
            if dirty {
                let end = offsets.get(at).map_or(text.len(), |span| span.0);
                self.push_ordinary(text, from_byte..end, ids_out, offsets_out)?;
            } else {
                ids_out.extend_from_slice(&ids[from..at]);
                offsets_out.extend_from_slice(&offsets[from..at]);
            }
            if at < ids.len() {
                ids_out.push(ids[at]);
                offsets_out.push(offsets[at]);
                from_byte = offsets[at].1;
            }
            from = at + 1;
            dirty = false;
        }
        Ok(())
    }

    /// Appends the encoding of `text[range]` without the added vocabulary, with byte spans
    /// relative to `text`.
    fn push_ordinary(
        &self,
        text: &str,
        range: std::ops::Range<usize>,
        ids: &mut Vec<u32>,
        offsets: &mut Vec<(usize, usize)>,
    ) -> Result<()> {
        let (part_ids, part_offsets) = encode_with_budget(
            &self.ordinary_tokenizer,
            &text[range.clone()],
            false,
            &self.encode_threads,
        )?;
        ids.extend_from_slice(&part_ids);
        offsets.extend(
            part_offsets
                .into_iter()
                .map(|(a, b)| (range.start + a, range.start + b)),
        );
        Ok(())
    }

    fn mark_continuation(&self, messages: &mut [Json], field: &str) -> Result<String> {
        let message = messages.last_mut().ok_or_else(|| {
            Error::frontend(FrontendError::Input(Failure::new(
                "list index out of range",
            )))
        })?;
        let value = message.get_mut(field).filter(|value| !value.is_null()).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new(format!("continue_final_message is set but the final message has no \"{field}\" to continue!")))))?;
        if !self.template_source.contains(field) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                format!(
                    "continue_final_message is set to \"{field}\" but this is not an accepted field in the chat_template"
                ),
            ))));
        }
        let text = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .and_then(|value| value.as_str())
                .map(str::to_owned)
        } else {
            value.as_str().map(str::to_owned)
        };
        let original = text.ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but we could not find any text to continue in the final message!"))))?;
        let target = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .unwrap()
        } else {
            value
        };
        *target = Json::String(format!("{original}CONTINUE_FINAL_MESSAGE_TAG "));
        Ok(original)
    }

    #[cfg(feature = "server")]
    pub(crate) fn prepare_token_bytes(&self) -> Result<()> {
        self.vocabulary_bytes.get_or_init(|| vocabulary::VocabularyBytes::new(&self.tokenizer))
            .as_ref().map(|_| ()).map_err(Clone::clone)
    }

    pub fn token_bytes(&self, token: u32) -> Result<&[u8]> {
        self.vocabulary_bytes.get_or_init(|| vocabulary::VocabularyBytes::new(&self.tokenizer))
            .as_ref().map_err(Clone::clone)?.get(token)
    }

    pub fn incremental_decoder(&self, skip_special_tokens: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: &self.tokenizer,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            finished: false,
        }
    }
}

pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    finished: bool,
}

impl IncrementalDecoder<'_> {
    pub fn push(&mut self, token_ids: &[u32]) -> Result<Option<String>> {
        if self.finished {
            return Err(Error::frontend(FrontendError::Tokenizer(Failure::new(
                "incremental decoder is finished",
            ))));
        }
        if token_ids.is_empty() {
            return Ok(None);
        }
        tokenizers::tokenizer::step_decode_stream(
            self.tokenizer,
            token_ids.to_vec(),
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })
    }

    #[cfg(feature = "server")]
    pub(crate) fn pending_text(&self) -> Result<String> {
        let text = self.tokenizer.decode(&self.ids, self.skip_special_tokens).map_err(|e| e.to_string())?;
        Ok(text.strip_prefix(&self.prefix).ok_or("decoder changed its emitted prefix")?.to_owned())
    }

    pub fn finish(&mut self) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        let remainder = self
            .tokenizer
            .decode(&self.ids, self.skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })?;
        let suffix = remainder
            .strip_prefix(&self.prefix)
            .ok_or_else(|| {
                Error::frontend(FrontendError::Tokenizer(Failure::new(
                    "incremental decoder final text changed its emitted prefix",
                )))
            })?
            .to_owned();
        self.finished = true;
        Ok(suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_is_shareable_between_native_request_workers() {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Frontend>();
    }

    fn role_frontend(guard: &str) -> Frontend {
        let source = format!(
            "{{% for message in messages %}}{guard}{{{{ message.role }}}}:{{{{ message.content }}}}\n{{% endfor %}}"
        );
        let template = Some(jinja::CompiledTemplate::compile(&source).unwrap());
        Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            ordinary_tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            controls: Default::default(),
            spelled_image_placeholder: None,
            template_encoding: TemplateEncoding::Text,
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template,
            template_source: source,
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            defaults: Map::new(),
            encode_threads: EncodeThreadBudget::new(1),
        }
    }

    #[test]
    fn inline_system_roles_follow_template_capability() {
        let source = serde_json::json!([
            {"role":"system","content":"Initial instructions"},
            {"role":"user","content":"Hello"},
            {"role":"system","content":[
                {"type":"text","text":"Environment"},
                {"type":"text","text":"Use the repository"}
            ]},
            {"role":"assistant","content":"Ready"},
            {"role":"user","content":"Continue"}
        ]);
        let original = source.as_array().unwrap();
        for (guard, supported) in [
            ("", true),
            (
                "{% if message.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}",
                false,
            ),
            (
                "{% if message.role == 'system' and not loop.first %}{% continue %}{% endif %}",
                false,
            ),
        ] {
            let frontend = role_frontend(guard);
            let mut messages = original.clone();
            frontend
                .normalize_inline_system_messages(&mut messages)
                .unwrap();
            assert_eq!(&messages[..2], &original[..2]);
            assert_eq!(&messages[3..], &original[3..]);
            if supported {
                assert_eq!(messages, *original);
            } else {
                assert_eq!(
                    messages[2],
                    serde_json::json!({
                        "role":"user","content":"<system>Environment\n\nUse the repository</system>"
                    })
                );
                frontend.render_chat(&messages, &Map::new()).unwrap();
            }
            let once = messages.clone();
            frontend
                .normalize_inline_system_messages(&mut messages)
                .unwrap();
            assert_eq!(messages, once);
        }
    }

    #[test]
    fn inline_system_conversion_preserves_text_and_rejects_nontext() {
        let frontend = role_frontend(
            "{% if message.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}",
        );
        let mut messages = vec![
            serde_json::json!({"role":"user","content":"Hello"}),
            serde_json::json!({"role":"system","content":"\nUse <tag> & café.\n"}),
        ];
        frontend
            .normalize_inline_system_messages(&mut messages)
            .unwrap();
        assert_eq!(
            messages[1]["content"],
            "<system>\nUse <tag> & café.\n</system>"
        );
        messages[1] = serde_json::json!({"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]});
        assert!(
            frontend
                .normalize_inline_system_messages(&mut messages)
                .is_err()
        );
    }

    #[test]
    fn advisory_effort_uses_the_prepared_templates_vocabulary() {
        for (levels, expected) in [
            ("['low','medium','xhigh']", ["low", "medium", "xhigh"]),
            ("['low','high','max']", ["low", "high", "max"]),
        ] {
            let source = format!(
                "{{% if reasoning_effort not in {levels} %}}{{{{ raise_exception('unsupported effort') }}}}{{% endif %}}"
            );
            let template = Some(jinja::CompiledTemplate::compile(&source).unwrap());
            let frontend = Frontend {
                prompt_encoder: PromptEncoder::Jinja,
                image_placeholder_token_id: None,
                ordinary_tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
                controls: Default::default(),
                spelled_image_placeholder: None,
                template_encoding: TemplateEncoding::Text,
                tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
                template,
                template_source: source,
                bos_token_id: None,
                vocabulary_bytes: Default::default(),
                effort_levels: Default::default(),
                inline_system_messages: Default::default(),
                defaults: Map::new(),
                encode_threads: EncodeThreadBudget::new(1),
            };
            for (candidates, expected) in [
                (&["low"][..], expected[0]),
                (&["high", "medium"][..], expected[1]),
                (&["max", "xhigh", "high", "medium"][..], expected[2]),
            ] {
                assert_eq!(frontend.resolve_effort(candidates).unwrap(), expected);
            }
            assert!(frontend.resolve_effort(&["unsupported"]).is_err());
        }
    }

    #[test]
    fn missing_asset_retains_io_failure() {
        use std::error::Error as _;

        let asset = Asset {
            range: None,
            path: std::env::temp_dir()
                .join(format!("flare-missing-asset-{}", std::process::id()))
                .join("tokenizer.json"),
            sha256: "0".repeat(64),
        };
        let error = asset.read().unwrap_err();
        assert!(matches!(error, Error::Frontend(FrontendError::Asset(_))));
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap();
        assert_eq!(cause.kind(), std::io::ErrorKind::NotFound);
        let cloned = error.clone();
        assert_eq!(cloned.to_string(), error.to_string());
        assert!(std::ptr::eq(
            error.source().unwrap(),
            cloned.source().unwrap()
        ));
    }

    #[test]
    fn render_failure_retains_template_cause_and_detail() {
        use std::error::Error as _;

        let source = "{{ raise_exception('fixture rejection') }}";
        let template = Some(jinja::CompiledTemplate::compile(source).unwrap());
        let frontend = Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            ordinary_tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            controls: Default::default(),
            spelled_image_placeholder: None,
            template_encoding: TemplateEncoding::Text,
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template,
            template_source: source.to_owned(),
            defaults: Map::new(),
            encode_threads: EncodeThreadBudget::new(1),
        };
        let error = frontend
            .render_chat(
                &[serde_json::json!({"role":"user","content":"x"})],
                &Map::new(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, Error::Frontend(FrontendError::Template(_))));
        assert_eq!(error.to_string(), "fixture rejection");
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<jinja::Error>()
            .unwrap();
        assert_eq!(cause.message, "fixture rejection");
        let input = frontend.render_chat(&[], &Map::new()).err().unwrap();
        assert!(matches!(input, Error::Frontend(FrontendError::Input(_))));
        assert!(input.source().is_none());
    }

    #[test]
    fn verified_asset_rejects_content_changed_after_fingerprinting() {
        let directory = std::env::temp_dir().join(format!("flare-asset-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("tokenizer_config.json");
        let original = b"{\"eos_token\":\"<eos>\"}";
        fs::write(&path, original).unwrap();
        let asset = Asset {
            range: None,
            path: path.clone(),
            sha256: format!("{:x}", Sha256::digest(original)),
        };
        assert_eq!(asset.read().unwrap(), original);
        fs::write(&path, b"{\"eos_token\":\"changed\"}").unwrap();
        assert!(matches!(
            asset.read().unwrap_err(),
            Error::Frontend(FrontendError::Asset(_))
        ));
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn image_segment_indices_reorder_sources_and_hints_and_reject_ambiguity() {
        let mut hints = Vec::new();
        let segments = token_segments(r#"[{"text":"second","allow_special":true,"image_index":1},{"text":"first","allow_special":true,"image_index":0}]"#, &mut hints).unwrap();
        let mut sources = vec!["first".into(), "second".into()];
        let mut checkpoints = vec![0];
        order_segment_images(&segments, &mut sources, &mut checkpoints).unwrap();
        assert_eq!(sources, ["second", "first"]);
        assert_eq!(checkpoints, [1]);
        for indices in [vec![0,0], vec![0,2], vec![1]] {
            let segments = indices.into_iter().map(|index| TokenSegment {
                text: "image".into(), allow_special:true, image_index:Some(index),
            }).collect::<Vec<_>>();
            assert!(order_segment_images(&segments, &mut sources, &mut checkpoints).is_err());
        }
        let segments = token_segments(r#"[{"text":"legacy","allow_special":true}]"#, &mut hints).unwrap();
        order_segment_images(&segments, &mut sources, &mut checkpoints).unwrap();
        assert_eq!(sources, ["second", "first"]);
        assert_eq!(checkpoints, [1]);
    }

    #[test]
    fn segment_hints_survive_json_escapes_key_order_and_nested_serialization() {
        use jinja::{CompiledTemplate, Mark};
        use serde_json::json;
        let context = json!({"messages":[{"content":[{"text":"é\n\"\\😀"}]}]});
        for ascii in ["true", "false"] {
            for sort in ["true", "false"] {
                let source = format!(
                    concat!(
                        "{{% set history = messages[0].content[0].text %}}",
                        "{{% set segments = [{{'text': 'prefix' + history + 'tail', 'allow_special': false}},",
                        "{{'text': history | tojson(ensure_ascii={0}), 'allow_special': true}}] %}}",
                        "{{{{ segments | tojson(ensure_ascii={0}, sort_keys={1}, indent=2) }}}}",
                    ),
                    ascii, sort
                );
                let template = CompiledTemplate::compile(&source).unwrap();
                let plain = template.render(context.as_object().unwrap(), &[]).unwrap();
                let marked = template
                    .render(context.as_object().unwrap(), &[(0, 1)])
                    .unwrap();
                assert_eq!(plain.text, marked.text);
                let mut hints = marked
                    .marks
                    .into_iter()
                    .map(|mark| match mark {
                        Mark::Hint(byte) => byte,
                        _ => panic!("unexpected mark"),
                    })
                    .collect();
                let segments = token_segments(&marked.text, &mut hints).unwrap();
                assert_eq!(segments[0].text, "prefixé\n\"\\😀tail");
                assert!(!segments[0].allow_special);
                assert!(segments[1].allow_special);
                assert_eq!(
                    hints,
                    [
                        "prefixé\n\"\\😀".len(),
                        segments[0].text.len() + segments[1].text.len() - 1
                    ]
                );
            }
        }
        let template = CompiledTemplate::compile(
            "{{ {'text': messages[0].content[0].text, 'text': 'x'} | tojson }}",
        )
        .unwrap();
        let rendered = template
            .render(context.as_object().unwrap(), &[(0, 1)])
            .unwrap();
        assert_eq!(rendered.text, "{\"text\": \"x\"}");
        assert!(rendered.marks.is_empty());
    }

    fn byte_level_bpe_tokenizer(extra_merges: &[(&str, &str)]) -> Tokenizer {
        use tokenizers::models::bpe::{BPE, Vocab};
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::Split;
        use tokenizers::SplitDelimiterBehavior;

        let mut vocab = Vocab::default();
        let mut next_id = 0u32;
        for ch in ByteLevel::alphabet() {
            let key = ch.to_string();
            if !vocab.contains_key(&key) {
                vocab.insert(key, next_id);
                next_id += 1;
            }
        }
        let mut merges = vec![
            ("Ġ".to_string(), "t".to_string()),
            ("Ġt".to_string(), "h".to_string()),
            ("Ġth".to_string(), "e".to_string()),
            ("Ċ".to_string(), "Ċ".to_string()),
            ("1".to_string(), "2".to_string()),
            ("12".to_string(), "3".to_string()),
        ];
        for &(left, right) in extra_merges {
            merges.push((left.to_string(), right.to_string()));
        }
        for (left, right) in &merges {
            let merged = format!("{left}{right}");
            if !vocab.contains_key(&merged) {
                vocab.insert(merged, next_id);
                next_id += 1;
            }
        }
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(bpe);
        let pre = Sequence::new(vec![
            Split::new(
                r"\s*[\r\n]+|\s+(?!\S)|\s+",
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap()
            .into(),
            ByteLevel::new(false, false, false).into(),
        ]);
        tokenizer.with_pre_tokenizer(Some(pre));
        tokenizer
    }

    fn char_bpe_tokenizer() -> Tokenizer {
        use tokenizers::models::bpe::{BPE, Vocab};

        let mut vocab = Vocab::default();
        let mut next_id = 0u32;
        for byte in 0x20u8..=0x7eu8 {
            vocab.insert((byte as char).to_string(), next_id);
            next_id += 1;
        }
        for ch in ['\n', '\r', '\t', '▁', 'é'] {
            vocab.insert(ch.to_string(), next_id);
            next_id += 1;
        }
        let merges = vec![
            ("t".to_string(), "h".to_string()),
            ("th".to_string(), "e".to_string()),
            ("▁".to_string(), "t".to_string()),
            ("▁t".to_string(), "h".to_string()),
            ("1".to_string(), "2".to_string()),
            ("12".to_string(), "3".to_string()),
        ];
        for (left, right) in &merges {
            let merged = format!("{left}{right}");
            if !vocab.contains_key(&merged) {
                vocab.insert(merged, next_id);
                next_id += 1;
            }
        }
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();
        Tokenizer::new(bpe)
    }

    fn assert_parallel_matches_sequential(
        tokenizer: &Tokenizer,
        text: &str,
        add_special_tokens: bool,
        budget: &EncodeThreadBudget,
    ) {
        let seq = tokenizer.encode(text, add_special_tokens).unwrap();
        let (par_ids, par_offsets) =
            encode_with_budget(tokenizer, text, add_special_tokens, budget).unwrap();
        assert_eq!(par_ids, seq.get_ids());
        assert_eq!(par_offsets, seq.get_offsets());
    }

    #[test]
    fn parallel_encode_matches_sequential_ids_and_offsets_exactly() {
        use tokenizers::normalizers::NFC;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::processors::template::TemplateProcessing;
        use tokenizers::AddedToken;

        let tokenizer = byte_level_bpe_tokenizer(&[]);
        let budget = EncodeThreadBudget::new(4);

        // 1. Multi-paragraph text exceeding PARALLEL_ENCODE_MIN_BYTES (32 KiB) with Unicode & CRLF.
        let para = "The quick brown fox jumps over 123 lazy dogs! café 😀\n\nSecond line with\r\nCRLF and punctuation...\n";
        let multi_para = para.repeat(900); // ~85 KiB
        assert!(multi_para.len() > PARALLEL_ENCODE_MIN_BYTES * 2);
        let chunks = find_parallel_chunks_with_threads(&tokenizer, &multi_para, false, 4).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks
            .iter()
            .all(|range| range.len() >= PARALLEL_ENCODE_MIN_CHUNK_BYTES));
        assert!(find_parallel_chunks_with_threads(&tokenizer, &multi_para, false, 1).is_none());
        assert_parallel_matches_sequential(&tokenizer, &multi_para, false, &budget);
        assert_parallel_matches_sequential(
            &tokenizer,
            &multi_para,
            false,
            &EncodeThreadBudget::new(1),
        );

        // 2. Single-line minified text (no newlines — exercises ASCII space fallback).
        let single_line = "the quick brown fox 123 json payload item ".repeat(2200);
        assert!(single_line.len() > PARALLEL_ENCODE_MIN_BYTES * 2);
        assert!(find_parallel_chunks_with_threads(&tokenizer, &single_line, false, 4).is_some());
        assert_parallel_matches_sequential(&tokenizer, &single_line, false, &budget);

        // 3. Added vocabulary token near split boundaries must remain bit-identical.
        let mut with_added = tokenizer.clone();
        let _ = with_added.add_special_tokens([AddedToken::from("<|special|>", true)]);
        let with_ctrl = format!("{}\n<|special|>\n{}", para.repeat(450), para.repeat(450));
        assert_parallel_matches_sequential(&with_added, &with_ctrl, false, &budget);

        // 4. add_prefix_space=true must disable chunk splitting and stay exact.
        let mut prefixed = tokenizer.clone();
        prefixed.with_pre_tokenizer(Some(ByteLevel::new(true, false, false)));
        assert!(find_parallel_chunks_with_threads(&prefixed, &multi_para, false, 4).is_none());
        assert_parallel_matches_sequential(&prefixed, &multi_para, false, &budget);

        // 5. Post-processor that adds special tokens disables splitting only when
        //    add_special_tokens is true.
        let mut with_bos = tokenizer.clone();
        let _ = with_bos.add_special_tokens([AddedToken::from("<|bos|>", true)]);
        let bos_id = with_bos.token_to_id("<|bos|>").unwrap();
        let processor = TemplateProcessing::builder()
            .try_single("<|bos|> $A")
            .unwrap()
            .special_tokens(vec![("<|bos|>", bos_id)])
            .build()
            .unwrap();
        with_bos.with_post_processor(Some(processor));
        assert!(find_parallel_chunks_with_threads(&with_bos, &multi_para, true, 4).is_none());
        assert!(find_parallel_chunks_with_threads(&with_bos, &multi_para, false, 4).is_some());
        assert_parallel_matches_sequential(&with_bos, &multi_para, true, &budget);
        assert_parallel_matches_sequential(&with_bos, &multi_para, false, &budget);

        // 6. Non-noop normalizer disables chunk splitting and stays exact.
        let mut normalized = tokenizer.clone();
        normalized.with_normalizer(Some(NFC)).unwrap();
        assert!(find_parallel_chunks_with_threads(&normalized, &multi_para, false, 4).is_none());
        assert_parallel_matches_sequential(&normalized, &multi_para, false, &budget);
    }

    #[test]
    fn parallel_encode_handles_stripping_tokens_across_long_whitespace() {
        use tokenizers::AddedToken;

        let mut tokenizer = byte_level_bpe_tokenizer(&[]);
        let _ = tokenizer.add_special_tokens([
            AddedToken::from("<|rstrip|>", true).rstrip(true),
            AddedToken::from("<|lstrip|>", true).lstrip(true),
        ]);
        let budget = EncodeThreadBudget::new(4);
        let straddle_half = PARALLEL_ENCODE_MIN_STRADDLE_BYTES;
        let filler = "The quick brown fox jumps over 123 lazy dogs!\n".repeat(450); // ~20 KiB

        // 512 spaces (> 256-byte straddle window) between <|rstrip|> and a newline cut
        // followed by trailing spaces: <|rstrip|> absorbs whitespace across the cut.
        let spaces_512 = " ".repeat(512);
        let rstrip_prefix = format!("{filler}<|rstrip|>{spaces_512}\n");
        let rstrip_cut = rstrip_prefix.len();
        let rstrip_text = format!("{rstrip_prefix}{spaces_512}{filler}");
        assert!(!verify_split_boundary(
            &tokenizer,
            &rstrip_text,
            rstrip_cut,
            0,
            straddle_half,
        ));
        assert_parallel_matches_sequential(&tokenizer, &rstrip_text, false, &budget);

        // 512 spaces (> 256-byte straddle window) between a newline cut and <|lstrip|>:
        // <|lstrip|> absorbs whitespace across the cut.
        let lstrip_prefix = format!("{filler}{spaces_512}\n");
        let lstrip_cut = lstrip_prefix.len();
        let lstrip_text = format!("{lstrip_prefix}{spaces_512}<|lstrip|>{filler}");
        assert!(!verify_split_boundary(
            &tokenizer,
            &lstrip_text,
            lstrip_cut,
            0,
            straddle_half,
        ));
        assert_parallel_matches_sequential(&tokenizer, &lstrip_text, false, &budget);

        // Both <|rstrip|> and <|lstrip|> flanking a 1024-byte whitespace run around the cut:
        // token IDs match across split vs full, but offsets differ.
        let both_prefix = format!("{filler}<|rstrip|>{spaces_512}\n");
        let both_cut = both_prefix.len();
        let both_text = format!("{both_prefix}{spaces_512}<|lstrip|>{filler}");
        assert!(!verify_split_boundary(
            &tokenizer,
            &both_text,
            both_cut,
            0,
            straddle_half,
        ));
        assert_parallel_matches_sequential(&tokenizer, &both_text, false, &budget);

        // Whitespace runs longer than PARALLEL_ENCODE_MAX_VERIFY_WHITESPACE_BYTES (4 KiB)
        // touching the cut are rejected unconditionally without unbounded window growth.
        let spaces_5000 = " ".repeat(PARALLEL_ENCODE_MAX_VERIFY_WHITESPACE_BYTES + 904);
        let huge_prefix = format!("{filler}<|rstrip|>{spaces_5000}\n");
        let huge_cut = huge_prefix.len();
        let huge_text = format!("{huge_prefix}{spaces_5000}{filler}");
        assert!(!verify_split_boundary(
            &tokenizer,
            &huge_text,
            huge_cut,
            0,
            straddle_half,
        ));
        assert_parallel_matches_sequential(&tokenizer, &huge_text, false, &budget);
    }

    #[test]
    fn parallel_encode_falls_back_when_verify_split_boundary_rejects_cuts() {
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;

        // ByteLevel with use_regex=false does not split on newlines, so merging
        // "Ċ" + "a" spans every "\n" -> "a" candidate boundary and forces
        // verify_split_boundary to reject every cut.
        let mut tokenizer = byte_level_bpe_tokenizer(&[("Ċ", "a")]);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));

        let text = "aaaaaa\n".repeat(6000); // ~42 KiB
        assert!(text.len() > PARALLEL_ENCODE_MIN_BYTES);
        let cut = text[20_000..].find('\n').unwrap() + 20_000 + 1;
        assert!(!verify_split_boundary(
            &tokenizer,
            &text,
            cut,
            0,
            PARALLEL_ENCODE_MIN_STRADDLE_BYTES,
        ));
        assert!(find_parallel_chunks_with_threads(&tokenizer, &text, false, 4).is_none());
        assert_parallel_matches_sequential(&tokenizer, &text, false, &EncodeThreadBudget::new(4));
    }

    #[test]
    fn parallel_encode_supports_whitelisted_pre_tokenizers_and_rejects_disallowed() {
        use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::delimiter::CharDelimiterSplit;
        use tokenizers::pre_tokenizers::digits::Digits;
        use tokenizers::pre_tokenizers::fixed_length::FixedLength;
        use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
        use tokenizers::pre_tokenizers::punctuation::Punctuation;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::Split;
        use tokenizers::pre_tokenizers::unicode_scripts::UnicodeScripts;
        use tokenizers::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
        use tokenizers::{PreTokenizerWrapper, SplitDelimiterBehavior};

        let budget = EncodeThreadBudget::new(4);
        let ascii_text = "the quick brown fox 123, lazy dogs! cafe\nsecond line 123.\n".repeat(1200);
        assert!(ascii_text.len() > PARALLEL_ENCODE_MIN_BYTES * 2);

        let allowed: Vec<(&str, PreTokenizerWrapper)> = vec![
            ("ByteLevel(use_regex=true)", ByteLevel::new(false, false, true).into()),
            (
                "Split",
                Split::new(r"\s+", SplitDelimiterBehavior::Isolated, false)
                    .unwrap()
                    .into(),
            ),
            ("Whitespace", Whitespace {}.into()),
            ("WhitespaceSplit", WhitespaceSplit {}.into()),
            ("Delimiter", CharDelimiterSplit::new('\n').into()),
            ("Punctuation", Punctuation::new(SplitDelimiterBehavior::Isolated).into()),
            ("Digits", Digits::new(true).into()),
            ("UnicodeScripts", UnicodeScripts {}.into()),
            ("BertPreTokenizer", BertPreTokenizer {}.into()),
            (
                "Metaspace(Never)",
                Metaspace::new('▁', PrependScheme::Never, true).into(),
            ),
            (
                "Sequence([WhitespaceSplit, Digits])",
                Sequence::new(vec![WhitespaceSplit {}.into(), Digits::new(true).into()]).into(),
            ),
        ];

        for (label, pre) in allowed {
            let mut tokenizer = if label.starts_with("ByteLevel") {
                byte_level_bpe_tokenizer(&[])
            } else {
                char_bpe_tokenizer()
            };
            tokenizer.with_pre_tokenizer(Some(pre));
            assert!(
                pre_tokenizer_is_chunk_safe(tokenizer.get_pre_tokenizer()),
                "{label} should be chunk-safe"
            );
            let chunks =
                find_parallel_chunks_with_threads(&tokenizer, &ascii_text, false, 4);
            assert!(
                chunks.as_ref().is_some_and(|c| c.len() >= 2),
                "{label} should split into >= 2 verified chunks"
            );
            assert_parallel_matches_sequential(&tokenizer, &ascii_text, false, &budget);
        }

        let disallowed: Vec<(&str, Option<PreTokenizerWrapper>)> = vec![
            ("None", None),
            ("EmptySequence", Some(Sequence::new(vec![]).into())),
            ("FixedLength", Some(FixedLength::new(16).into())),
            (
                "Metaspace(Always)",
                Some(Metaspace::new('▁', PrependScheme::Always, true).into()),
            ),
            (
                "Metaspace(First)",
                Some(Metaspace::new('▁', PrependScheme::First, true).into()),
            ),
            (
                "Sequence([Whitespace, FixedLength])",
                Some(Sequence::new(vec![Whitespace {}.into(), FixedLength::new(16).into()]).into()),
            ),
        ];

        for (label, pre) in disallowed {
            let mut tokenizer = char_bpe_tokenizer();
            tokenizer.with_pre_tokenizer(pre);
            assert!(
                !pre_tokenizer_is_chunk_safe(tokenizer.get_pre_tokenizer()),
                "{label} must not be chunk-safe"
            );
            assert!(
                find_parallel_chunks_with_threads(&tokenizer, &ascii_text, false, 4).is_none(),
                "{label} must fall back to sequential"
            );
            assert_parallel_matches_sequential(&tokenizer, &ascii_text, false, &budget);
        }
    }

    #[test]
    fn parallel_encode_thread_budget_bounds_concurrent_workers() {
        // 1. Deterministic permit accounting.
        let budget = EncodeThreadBudget::new(4);
        assert_eq!(budget.max_threads(), 4);
        assert_eq!(budget.active_threads(), 0);

        let mut p1 = budget.acquire(8);
        assert_eq!(p1.threads(), 4);
        assert_eq!(budget.active_threads(), 4);

        // Budget saturated: concurrent caller gets 1 (its own thread, 0 helpers).
        let p2 = budget.acquire(8);
        assert_eq!(p2.threads(), 1);
        assert_eq!(budget.active_threads(), 5);

        // Shrinking releases unused slots immediately.
        p1.shrink_to(2);
        assert_eq!(p1.threads(), 2);
        assert_eq!(budget.active_threads(), 3);

        // Dropping p1 leaves active_threads = 1, so next caller can claim 3 (1 + 2 helpers).
        drop(p1);
        assert_eq!(budget.active_threads(), 1);
        let p3 = budget.acquire(8);
        assert_eq!(p3.threads(), 3);
        assert_eq!(budget.active_threads(), 4);

        drop(p2);
        drop(p3);
        assert_eq!(budget.active_threads(), 0);

        // 2. Concurrent long-prompt encoding never exceeds max_threads - 1 helper threads.
        let tokenizer = byte_level_bpe_tokenizer(&[]);
        let text = "The quick brown fox jumps over 123 lazy dogs! café 😀\n".repeat(3000); // ~165 KiB
        let expected = tokenizer.encode(text.as_str(), false).unwrap();

        for cap in [1usize, 4usize] {
            let shared_budget = EncodeThreadBudget::new(cap);
            let barrier = std::sync::Barrier::new(8);
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for _ in 0..8 {
                    handles.push(scope.spawn(|| {
                        barrier.wait();
                        for _ in 0..3 {
                            let (ids, offsets) = encode_with_budget(
                                &tokenizer,
                                &text,
                                false,
                                &shared_budget,
                            )
                            .unwrap();
                            assert_eq!(ids, expected.get_ids());
                            assert_eq!(offsets, expected.get_offsets());
                        }
                    }));
                }
                for handle in handles {
                    handle.join().unwrap();
                }
            });
            assert_eq!(shared_budget.active_threads(), 0);
            assert!(
                shared_budget.peak_helpers() <= cap - 1,
                "cap={cap}: observed {} concurrent helpers, expected <= {}",
                shared_budget.peak_helpers(),
                cap - 1
            );
        }

        // 3. Frontend::with_max_encode_threads clamps to 1..=PARALLEL_ENCODE_MAX_THREADS.
        let frontend = role_frontend("").with_max_encode_threads(0);
        assert_eq!(frontend.max_encode_threads(), 1);
        let frontend = frontend.with_max_encode_threads(4);
        assert_eq!(frontend.max_encode_threads(), 4);
        let frontend = frontend.with_max_encode_threads(64);
        assert_eq!(frontend.max_encode_threads(), PARALLEL_ENCODE_MAX_THREADS);
    }
}
