/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Terminal rendering and rolling-window rate math for `Execution.persistent_workers` telemetry.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, BorderType, Chart, Dataset, GraphType, Paragraph, Row, Table},
};

use super::{BACKGROUND, CYAN, GOLD, GREEN, HISTORY, MUTED, PURPLE, TEXT, WINDOW, rate};
use crate::execution::{PersistentWorkerSnapshot, WorkerState};

const SM103_EXPERT_MOVER: &str = "sm103_expert_mover";

/// Display-formatted telemetry decoded from the 8-counter `sm103_expert_mover` layout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoverTelemetry {
    pub swaps_completed: u64,
    pub bytes_copied: u64,
    pub rcu_waits: u64,
    pub skipped_already_resident: u64,
    pub last_packed: u64,
    pub last_layer: Option<u32>,
    pub last_expert: Option<u32>,
    pub predict_hits: u64,
    pub predict_misses: u64,
    pub predict_enqueues: u64,
    /// Cumulative HBM hit percentage (`hits / (hits + misses) * 100.0`).
    pub cumulative_hit_pct: Option<f64>,
    /// Recent sliding-window HBM hit percentage (`dhits / (dhits + dmisses) * 100.0`).
    pub recent_hit_pct: Option<f64>,
    /// Swaps completed per second over the recent sliding window.
    pub swaps_per_sec: f64,
    /// Bytes copied per second over the recent sliding window.
    pub bytes_per_sec: f64,
    /// RCU waits per second over the recent sliding window.
    pub rcu_waits_per_sec: f64,
    /// Predict enqueues per second over the recent sliding window.
    pub enqueues_per_sec: f64,
}

/// Per-second rates and optional decoded mover view derived from successive worker snapshots.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkerRates {
    /// Per-second delta rate for each raw counter index.
    pub counter_rates: Vec<f64>,
    /// Decoded mover telemetry when `name == "sm103_expert_mover"` and `counters.len() >= 8`.
    pub mover: Option<MoverTelemetry>,
}

#[derive(Clone, Copy, Debug)]
struct WorkerPoint {
    time: f64,
    recent_hit_pct: Option<f64>,
    cumulative_hit_pct: Option<f64>,
    swaps_per_sec: f64,
    gb_per_sec: f64,
    primary_rate: f64,
}

/// Stateful rolling history and terminal renderer for a persistent worker's counters.
#[derive(Debug)]
pub struct WorkerView {
    start: Instant,
    previous: VecDeque<(Instant, PersistentWorkerSnapshot)>,
    points: VecDeque<WorkerPoint>,
    snapshot: Option<PersistentWorkerSnapshot>,
    rates: WorkerRates,
}

impl Default for WorkerView {
    fn default() -> Self {
        Self::new(Instant::now())
    }
}

impl WorkerView {
    /// Creates a worker telemetry view whose history timestamps count seconds from `start`.
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            previous: VecDeque::new(),
            points: VecDeque::with_capacity(256),
            snapshot: None,
            rates: WorkerRates::default(),
        }
    }

    /// Returns `true` when the latest sample contained an active or configured worker.
    pub fn is_present(&self) -> bool {
        self.snapshot.is_some()
    }

    /// Tab label in the serving dashboard (`Parallax` for the expert mover, `Worker` otherwise).
    pub fn tab_label(&self) -> &'static str {
        if self.rates.mover.is_some() {
            " Parallax "
        } else {
            " Worker "
        }
    }

    /// Most recently sampled worker snapshot, if any.
    pub fn snapshot(&self) -> Option<&PersistentWorkerSnapshot> {
        self.snapshot.as_ref()
    }

    /// Rates computed at the most recent sample.
    pub fn rates(&self) -> &WorkerRates {
        &self.rates
    }

    /// Records the latest optional worker snapshot at `now` and updates rolling-window rates.
    pub fn sample(&mut self, now: Instant, snapshot: Option<PersistentWorkerSnapshot>) {
        let Some(snapshot) = snapshot else {
            self.snapshot = None;
            self.rates = WorkerRates::default();
            self.previous.clear();
            self.points.clear();
            return;
        };

        let reset_history = self.previous.back().is_some_and(|(_, prev)| {
            prev.name != snapshot.name
                || prev.counters.len() != snapshot.counters.len()
                || counters_regressed(prev, &snapshot)
        });
        if reset_history {
            self.previous.clear();
            self.points.clear();
        }

        while self.previous.len() > 1 && now.saturating_duration_since(self.previous[1].0) >= WINDOW
        {
            self.previous.pop_front();
        }

        let rates = match self.previous.front() {
            Some((then, before)) => compute_rates(
                before,
                &snapshot,
                now.saturating_duration_since(*then),
                self.rates.mover.and_then(|m| m.recent_hit_pct),
            ),
            None => compute_rates(&snapshot, &snapshot, Duration::ZERO, None),
        };

        let time = now.saturating_duration_since(self.start).as_secs_f64();
        let point = if let Some(mover) = rates.mover {
            WorkerPoint {
                time,
                recent_hit_pct: mover.recent_hit_pct,
                cumulative_hit_pct: mover.cumulative_hit_pct,
                swaps_per_sec: mover.swaps_per_sec,
                gb_per_sec: mover.bytes_per_sec / 1e9,
                primary_rate: mover.swaps_per_sec,
            }
        } else {
            WorkerPoint {
                time,
                recent_hit_pct: None,
                cumulative_hit_pct: None,
                swaps_per_sec: 0.0,
                gb_per_sec: 0.0,
                primary_rate: rates.counter_rates.first().copied().unwrap_or(0.0),
            }
        };
        self.points.push_back(point);
        while self
            .points
            .front()
            .is_some_and(|p| time - p.time > HISTORY)
        {
            self.points.pop_front();
        }
        self.previous.push_back((now, snapshot.clone()));
        self.snapshot = Some(snapshot);
        self.rates = rates;
    }

    /// Compact single-line text summary for tiny terminals.
    pub fn compact_line(&self) -> Option<Line<'static>> {
        let snap = self.snapshot.as_ref()?;
        let state_label = snap.state.as_str();
        let summary = if let Some(mover) = self.rates.mover {
            let hit = mover
                .recent_hit_pct
                .or(mover.cumulative_hit_pct)
                .map_or_else(|| "—".into(), |pct| format!("{pct:.1}%"));
            format!(
                " Parallax     [{state_label}] {} SMs · hit {hit} · {} swaps/s · {:.2} GB/s",
                snap.worker_sm_count,
                rate(mover.swaps_per_sec),
                mover.bytes_per_sec / 1e9,
            )
        } else {
            let first = snap
                .counters
                .first()
                .map(|v| format!("c0={}", format_commas(*v)))
                .unwrap_or_else(|| "no counters".into());
            format!(
                " Worker       {} [{state_label}] {} SMs · {first}",
                snap.name, snap.worker_sm_count
            )
        };
        Some(Line::from(Span::styled(
            summary,
            Style::default().fg(CYAN),
        )))
    }

    /// Plain-text multi-line summary for `--plain` or non-TTY CLI output.
    pub fn format_plain_summary(&self) -> String {
        let Some(snap) = self.snapshot.as_ref() else {
            return "No persistent worker configured.".into();
        };
        let mut out = if self.rates.mover.is_some() {
            format!(
                "Parallax ({})  state={}  sms={}\n",
                snap.name,
                snap.state.as_str(),
                snap.worker_sm_count
            )
        } else {
            format!(
                "Worker: {}  state={}  sms={}\n",
                snap.name,
                snap.state.as_str(),
                snap.worker_sm_count
            )
        };
        if let Some(mover) = self.rates.mover {
            let live_hit = mover
                .recent_hit_pct
                .map_or_else(|| "—".into(), |p| format!("{p:.2}%"));
            let cum_hit = mover
                .cumulative_hit_pct
                .map_or_else(|| "—".into(), |p| format!("{p:.2}%"));
            let last = format_last_swap(mover.last_layer, mover.last_expert);
            out.push_str(&format!(
                "HBM Hit Rate: {live_hit} recent / {cum_hit} cumulative  (hits={} misses={})\n\
                 Swaps: {} ({}/s)   Copied: {} ({:.2} GB/s)   Last: {last}\n\
                 Enqueues: {} ({}/s)   RCU Waits: {} ({}/s)   Skipped Resident: {}\n",
                format_commas(mover.predict_hits),
                format_commas(mover.predict_misses),
                format_commas(mover.swaps_completed),
                rate(mover.swaps_per_sec),
                format_bytes_decimal(mover.bytes_copied),
                mover.bytes_per_sec / 1e9,
                format_commas(mover.predict_enqueues),
                rate(mover.enqueues_per_sec),
                format_commas(mover.rcu_waits),
                rate(mover.rcu_waits_per_sec),
                format_commas(mover.skipped_already_resident),
            ));
        } else if snap.counters.is_empty() {
            out.push_str("Counters: none\n");
        } else {
            for (idx, value) in snap.counters.iter().enumerate() {
                let r = self.rates.counter_rates.get(idx).copied().unwrap_or(0.0);
                out.push_str(&format!("  c{idx}: {} (+{}/s)\n", format_commas(*value), rate(r)));
            }
        }
        out
    }

    /// Compact 4-line bordered card embedded at the bottom of the launch Overview pane.
    pub fn render_overview_panel(&self, frame: &mut Frame, area: Rect) {
        let Some(snap) = self.snapshot.as_ref() else {
            return;
        };
        if area.width < 20 || area.height < 3 {
            return;
        }
        let (state_text, state_color) = state_badge(snap.state);
        let heading = if self.rates.mover.is_some() {
            " PARALLAX · ".to_owned()
        } else {
            format!(" WORKER · {} · ", snap.name)
        };
        let title = Line::from(vec![
            Span::styled(
                heading,
                Style::default()
                    .fg(if self.rates.mover.is_some() { CYAN } else { TEXT })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                state_text,
                Style::default().fg(state_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" · {} SMs ", snap.worker_sm_count),
                Style::default().fg(MUTED),
            ),
        ]);
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(MUTED))
            .title(title);

        let lines = if let Some(mover) = self.rates.mover {
            let live_hit = mover
                .recent_hit_pct
                .map_or_else(|| "—".into(), |p| format!("{p:.1}%"));
            let cum_hit = mover
                .cumulative_hit_pct
                .map_or_else(|| "—".into(), |p| format!("{p:.1}%"));
            let hit_color = hit_rate_color(mover.recent_hit_pct.or(mover.cumulative_hit_pct));
            let last = format_last_swap(mover.last_layer, mover.last_expert);
            vec![
                Line::from(vec![
                    Span::styled(" HBM HIT ", Style::default().fg(MUTED)),
                    Span::styled(
                        format!("{live_hit} live"),
                        Style::default().fg(hit_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!(" · {cum_hit} total"), Style::default().fg(CYAN)),
                    Span::styled("   SWAPS ", Style::default().fg(MUTED)),
                    Span::styled(
                        format!("{}/s", rate(mover.swaps_per_sec)),
                        Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" ({})", format_commas(mover.swaps_completed)),
                        Style::default().fg(TEXT),
                    ),
                    Span::styled("   MOVER ", Style::default().fg(MUTED)),
                    Span::styled(
                        format!("{:.2} GB/s", mover.bytes_per_sec / 1e9),
                        Style::default().fg(PURPLE).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" ({})", format_bytes_decimal(mover.bytes_copied)),
                        Style::default().fg(TEXT),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(" ENQUEUES ", Style::default().fg(MUTED)),
                    Span::styled(
                        format!(
                            "{}/s ({})",
                            rate(mover.enqueues_per_sec),
                            format_commas(mover.predict_enqueues)
                        ),
                        Style::default().fg(TEXT),
                    ),
                    Span::styled("   RCU WAITS ", Style::default().fg(MUTED)),
                    Span::styled(format_commas(mover.rcu_waits), Style::default().fg(TEXT)),
                    Span::styled("   SKIPPED ", Style::default().fg(MUTED)),
                    Span::styled(
                        format_commas(mover.skipped_already_resident),
                        Style::default().fg(TEXT),
                    ),
                    Span::styled("   LAST ", Style::default().fg(MUTED)),
                    Span::styled(last, Style::default().fg(GREEN).add_modifier(Modifier::BOLD)),
                ]),
            ]
        } else if snap.counters.is_empty() {
            vec![
                Line::from(Span::styled(
                    " Active persistent worker (no mapped-host counters declared).",
                    Style::default().fg(MUTED),
                )),
                Line::default(),
            ]
        } else {
            let mut spans = vec![Span::styled(" COUNTERS ", Style::default().fg(MUTED))];
            for (idx, val) in snap.counters.iter().take(4).enumerate() {
                let r = self.rates.counter_rates.get(idx).copied().unwrap_or(0.0);
                spans.push(Span::styled(
                    format!("c{idx}={} (+{}/s)  ", format_commas(*val), rate(r)),
                    Style::default().fg(TEXT),
                ));
            }
            let mut line2_spans = vec![Span::raw("          ")];
            for (idx, val) in snap.counters.iter().enumerate().skip(4).take(4) {
                let r = self.rates.counter_rates.get(idx).copied().unwrap_or(0.0);
                line2_spans.push(Span::styled(
                    format!("c{idx}={} (+{}/s)  ", format_commas(*val), rate(r)),
                    Style::default().fg(MUTED),
                ));
            }
            vec![Line::from(spans), Line::from(line2_spans)]
        };

        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    /// Full-screen worker telemetry dashboard (used in both Helios's Worker tab and `flare worker`).
    pub fn render_full(&self, frame: &mut Frame, area: Rect) {
        frame.render_widget(
            Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
            area,
        );
        if area.width == 0 || area.height == 0 {
            return;
        }
        let Some(snap) = self.snapshot.as_ref() else {
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(Span::styled(
                        " No persistent worker telemetry available.",
                        Style::default().fg(MUTED),
                    )),
                    Line::from(Span::styled(
                        " Waiting for server response...",
                        Style::default().fg(MUTED),
                    )),
                ]),
                area,
            );
            return;
        };

        if area.width < 60 || area.height < 14 {
            let plain = self.format_plain_summary();
            let mut lines: Vec<Line<'static>> = plain
                .lines()
                .map(|line| Line::from(format!(" {line}")))
                .collect();
            lines.push(
                Line::from(" Enlarge terminal (>= 60x14) for live charts.")
                    .style(Style::default().fg(MUTED)),
            );
            frame.render_widget(Paragraph::new(lines), area);
            return;
        }

        if let Some(mover) = self.rates.mover {
            self.render_mover_full(frame, area, snap, mover);
        } else {
            self.render_generic_full(frame, area, snap);
        }
    }

    fn render_mover_full(
        &self,
        frame: &mut Frame,
        area: Rect,
        snap: &PersistentWorkerSnapshot,
        mover: MoverTelemetry,
    ) {
        let outer = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(9),
            Constraint::Length(6),
        ])
        .split(area);

        render_mover_header(frame, outer[0], snap, mover);

        let charts = Layout::horizontal([Constraint::Percentage(52), Constraint::Percentage(48)])
            .split(outer[1]);
        self.render_hit_rate_chart(frame, charts[0], mover);

        if charts[1].height >= 12 {
            let right = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(charts[1]);
            self.render_swaps_chart(frame, right[0], mover);
            self.render_bandwidth_chart(frame, right[1], mover);
        } else {
            self.render_swaps_chart(frame, charts[1], mover);
        }

        render_mover_stats_card(frame, outer[2], mover);
    }

    fn render_generic_full(
        &self,
        frame: &mut Frame,
        area: Rect,
        snap: &PersistentWorkerSnapshot,
    ) {
        let outer = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length((snap.counters.len().clamp(3, 10) as u16) + 3),
        ])
        .split(area);

        let (state_text, state_color) = state_badge(snap.state);
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(vec![
                    Span::styled(
                        " WORKER ",
                        Style::default()
                            .fg(BACKGROUND)
                            .bg(CYAN)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(
                        format!(" {state_text} "),
                        Style::default()
                            .fg(BACKGROUND)
                            .bg(state_color)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  {}  ·  {} SMs  ·  {} counters", snap.name, snap.worker_sm_count, snap.counters.len()),
                        Style::default().fg(TEXT),
                    ),
                ]),
                Line::from(Span::styled(
                    " Generic persistent worker telemetry (mapped-host u64 counters).",
                    Style::default().fg(MUTED),
                )),
            ]),
            outer[0],
        );

        let now_t = self.points.back().map_or(0.0, |p| p.time);
        let data: Vec<(f64, f64)> = self
            .points
            .iter()
            .map(|p| (p.time - now_t, p.primary_rate))
            .collect();
        let peak = data.iter().map(|&(_, y)| y).fold(0.0f64, f64::max);
        let cur = self.rates.counter_rates.first().copied().unwrap_or(0.0);
        let title = if snap.counters.is_empty() {
            " COUNTER RATE  — ".to_owned()
        } else {
            format!(" COUNTER c0 RATE  {}/s ", rate(cur))
        };
        let datasets = vec![
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(CYAN))
                .data(&data),
        ];
        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(MUTED))
                        .title(title)
                        .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
                )
                .style(Style::default().bg(BACKGROUND))
                .x_axis(
                    Axis::default()
                        .bounds([-HISTORY, 0.0])
                        .labels(["−120s", "now"])
                        .style(Style::default().fg(MUTED)),
                )
                .y_axis(
                    Axis::default()
                        .bounds([0.0, peak.max(1.0) * 1.15])
                        .labels(["0".into(), rate(peak.max(1.0))])
                        .style(Style::default().fg(MUTED)),
                ),
            outer[1],
        );

        let rows: Vec<Row> = snap
            .counters
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let r = self.rates.counter_rates.get(index).copied().unwrap_or(0.0);
                Row::new(vec![
                    format!("c{index}"),
                    format_commas(*value),
                    format!("+{}/s", rate(r)),
                ])
                .style(Style::default().fg(TEXT))
            })
            .collect();
        let table = Table::new(
            rows,
            [
                Constraint::Length(10),
                Constraint::Length(22),
                Constraint::Min(14),
            ],
        )
        .header(
            Row::new(["Counter", "Value", "Rate"])
                .style(Style::default().fg(CYAN).add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(MUTED))
                .title(" MAPPED-HOST COUNTERS ")
                .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
        );
        frame.render_widget(table, outer[2]);
    }

    fn render_hit_rate_chart(&self, frame: &mut Frame, area: Rect, mover: MoverTelemetry) {
        let now_t = self.points.back().map_or(0.0, |p| p.time);
        let live_data: Vec<(f64, f64)> = self
            .points
            .iter()
            .filter_map(|p| p.recent_hit_pct.map(|y| (p.time - now_t, y)))
            .collect();
        let cum_data: Vec<(f64, f64)> = self
            .points
            .iter()
            .filter_map(|p| p.cumulative_hit_pct.map(|y| (p.time - now_t, y)))
            .collect();

        let min_val = live_data
            .iter()
            .chain(cum_data.iter())
            .map(|&(_, y)| y)
            .fold(100.0f64, f64::min);
        let y_min = if live_data.is_empty() && cum_data.is_empty() {
            0.0
        } else {
            (((min_val - 3.0).floor() / 5.0).floor() * 5.0).clamp(0.0, 90.0)
        };

        let live_now = mover
            .recent_hit_pct
            .map_or_else(|| "—".into(), |p| format!("{p:.1}%"));
        let cum_now = mover
            .cumulative_hit_pct
            .map_or_else(|| "—".into(), |p| format!("{p:.1}%"));
        let title = Line::from(vec![
            Span::styled(
                " HBM HIT RATE  ",
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{live_now} live"),
                Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" · ", Style::default().fg(MUTED)),
            Span::styled(format!("{cum_now} total "), Style::default().fg(CYAN)),
        ]);

        let datasets = vec![
            Dataset::default()
                .name("total")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(CYAN))
                .data(&cum_data),
            Dataset::default()
                .name("live")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(GREEN))
                .data(&live_data),
        ];

        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(MUTED))
                        .title(title),
                )
                .style(Style::default().bg(BACKGROUND))
                .x_axis(
                    Axis::default()
                        .bounds([-HISTORY, 0.0])
                        .labels(["−120s", "now"])
                        .style(Style::default().fg(MUTED)),
                )
                .y_axis(
                    Axis::default()
                        .bounds([y_min, 100.0])
                        .labels([format!("{y_min:.0}%"), "100%".into()])
                        .style(Style::default().fg(MUTED)),
                ),
            area,
        );
    }

    fn render_swaps_chart(&self, frame: &mut Frame, area: Rect, mover: MoverTelemetry) {
        let now_t = self.points.back().map_or(0.0, |p| p.time);
        let data: Vec<(f64, f64)> = self
            .points
            .iter()
            .map(|p| (p.time - now_t, p.swaps_per_sec))
            .collect();
        let peak = data.iter().map(|&(_, y)| y).fold(0.0f64, f64::max);
        let title = format!(" SWAPS  {}/s ", rate(mover.swaps_per_sec));
        let datasets = vec![
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(GOLD))
                .data(&data),
        ];
        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(MUTED))
                        .title(title)
                        .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
                )
                .style(Style::default().bg(BACKGROUND))
                .x_axis(
                    Axis::default()
                        .bounds([-HISTORY, 0.0])
                        .labels(["−120s", "now"])
                        .style(Style::default().fg(MUTED)),
                )
                .y_axis(
                    Axis::default()
                        .bounds([0.0, peak.max(10.0) * 1.15])
                        .labels(["0".into(), rate(peak.max(10.0))])
                        .style(Style::default().fg(MUTED)),
                ),
            area,
        );
    }

    fn render_bandwidth_chart(&self, frame: &mut Frame, area: Rect, mover: MoverTelemetry) {
        let now_t = self.points.back().map_or(0.0, |p| p.time);
        let data: Vec<(f64, f64)> = self
            .points
            .iter()
            .map(|p| (p.time - now_t, p.gb_per_sec))
            .collect();
        let peak = data.iter().map(|&(_, y)| y).fold(0.0f64, f64::max);
        let cur = mover.bytes_per_sec / 1e9;
        let title = format!(" MOVER BANDWIDTH  {cur:.2} GB/s ");
        let datasets = vec![
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(PURPLE))
                .data(&data),
        ];
        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(MUTED))
                        .title(title)
                        .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
                )
                .style(Style::default().bg(BACKGROUND))
                .x_axis(
                    Axis::default()
                        .bounds([-HISTORY, 0.0])
                        .labels(["−120s", "now"])
                        .style(Style::default().fg(MUTED)),
                )
                .y_axis(
                    Axis::default()
                        .bounds([0.0, peak.max(1.0) * 1.15])
                        .labels(["0".into(), format!("{:.1}", peak.max(1.0))])
                        .style(Style::default().fg(MUTED)),
                ),
            area,
        );
    }
}

fn render_mover_header(
    frame: &mut Frame,
    area: Rect,
    snap: &PersistentWorkerSnapshot,
    mover: MoverTelemetry,
) {
    let (state_text, state_color) = state_badge(snap.state);
    let active_hit = mover.recent_hit_pct.or(mover.cumulative_hit_pct);
    let hit_color = hit_rate_color(active_hit);
    let hit_str = active_hit.map_or_else(|| "—".into(), |p| format!("{p:.2}%"));
    let cum_str = mover
        .cumulative_hit_pct
        .map_or_else(|| "—".into(), |p| format!("{p:.2}%"));
    let last = format_last_swap(mover.last_layer, mover.last_expert);

    let line1 = Line::from(vec![
        Span::styled(
            " PARALLAX ",
            Style::default()
                .fg(BACKGROUND)
                .bg(CYAN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            format!(" {state_text} "),
            Style::default()
                .fg(BACKGROUND)
                .bg(state_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {}  ·  {} SMs", snap.name, snap.worker_sm_count),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  HBM HIT ", Style::default().fg(MUTED)),
        Span::styled(
            hit_str,
            Style::default().fg(hit_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" (total {cum_str})"), Style::default().fg(CYAN)),
        Span::styled("  ·  LAST ", Style::default().fg(MUTED)),
        Span::styled(last, Style::default().fg(GREEN).add_modifier(Modifier::BOLD)),
    ]);

    // Line 2: Visual hit-rate progress bar + throughput summary.
    let bar_width = (area.width as usize)
        .saturating_sub(54)
        .clamp(12, 36);
    let filled = active_hit
        .map(|pct| ((pct.clamp(0.0, 100.0) / 100.0) * (bar_width as f64)).round() as usize)
        .unwrap_or(0)
        .min(bar_width);
    let empty = bar_width.saturating_sub(filled);
    let total_routes = mover.predict_hits.saturating_add(mover.predict_misses);
    let line2 = Line::from(vec![
        Span::raw(" "),
        Span::styled("█".repeat(filled), Style::default().fg(hit_color)),
        Span::styled("░".repeat(empty), Style::default().fg(MUTED)),
        Span::styled(
            format!(
                "  {} / {} hits  ·  {}/s swaps  ·  {:.2} GB/s ({})",
                format_commas(mover.predict_hits),
                format_commas(total_routes),
                rate(mover.swaps_per_sec),
                mover.bytes_per_sec / 1e9,
                format_bytes_decimal(mover.bytes_copied),
            ),
            Style::default().fg(MUTED),
        ),
    ]);

    frame.render_widget(Paragraph::new(vec![line1, line2]), area);
}

fn render_mover_stats_card(frame: &mut Frame, area: Rect, mover: MoverTelemetry) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(MUTED))
        .title(" PARALLAX COUNTERS ")
        .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(inner);

    let kv = |k: &'static str, v: String, color: Color| {
        Line::from(vec![
            Span::styled(format!(" {k:<15}"), Style::default().fg(MUTED)),
            Span::styled(v, Style::default().fg(color)),
        ])
    };
    let live_hit = mover
        .recent_hit_pct
        .map_or_else(|| "—".into(), |p| format!("{p:.2}%"));
    let cum_hit = mover
        .cumulative_hit_pct
        .map_or_else(|| "—".into(), |p| format!("{p:.2}%"));
    let left_lines = vec![
        kv(
            "HBM Hit Rate",
            format!("{live_hit} live  /  {cum_hit} total"),
            hit_rate_color(mover.recent_hit_pct.or(mover.cumulative_hit_pct)),
        ),
        kv(
            "Predict Hits",
            format!(
                "{}  (miss {})",
                format_commas(mover.predict_hits),
                format_commas(mover.predict_misses)
            ),
            TEXT,
        ),
        kv(
            "Enqueues",
            format!(
                "{}  ({}/s)",
                format_commas(mover.predict_enqueues),
                rate(mover.enqueues_per_sec)
            ),
            CYAN,
        ),
        kv(
            "Skipped Res.",
            format_commas(mover.skipped_already_resident),
            MUTED,
        ),
    ];

    let last = format_last_swap(mover.last_layer, mover.last_expert);
    let right_lines = vec![
        kv(
            "Swaps Done",
            format!(
                "{}  ({}/s)",
                format_commas(mover.swaps_completed),
                rate(mover.swaps_per_sec)
            ),
            GOLD,
        ),
        kv(
            "Data Copied",
            format!(
                "{}  ({:.2} GB/s)",
                format_bytes_decimal(mover.bytes_copied),
                mover.bytes_per_sec / 1e9
            ),
            PURPLE,
        ),
        kv(
            "RCU Waits",
            format!(
                "{}  ({}/s)",
                format_commas(mover.rcu_waits),
                rate(mover.rcu_waits_per_sec)
            ),
            TEXT,
        ),
        kv("Last Swapped", last, GREEN),
    ];

    frame.render_widget(Paragraph::new(left_lines), cols[0]);
    frame.render_widget(Paragraph::new(right_lines), cols[1]);
}

/// Returns `true` when `snapshot` matches the 8-counter `sm103_expert_mover` display layout.
pub fn is_sm103_expert_mover(snapshot: &PersistentWorkerSnapshot) -> bool {
    snapshot.name == SM103_EXPERT_MOVER && snapshot.counters.len() >= 8
}

fn counters_regressed(
    before: &PersistentWorkerSnapshot,
    after: &PersistentWorkerSnapshot,
) -> bool {
    if is_sm103_expert_mover(before) && is_sm103_expert_mover(after) {
        // Indices 0, 1, 2, 3, 5, 6, 7 are monotonic; index 4 is `last_packed`.
        [0usize, 1, 2, 3, 5, 6, 7]
            .into_iter()
            .any(|idx| after.counters[idx] < before.counters[idx])
    } else {
        before
            .counters
            .first()
            .zip(after.counters.first())
            .is_some_and(|(b, a)| a < b)
    }
}

/// Computes per-second rates and optional decoded mover metrics between two snapshots.
pub fn compute_rates(
    before: &PersistentWorkerSnapshot,
    after: &PersistentWorkerSnapshot,
    elapsed: Duration,
    previous_recent_hit_pct: Option<f64>,
) -> WorkerRates {
    let dt = elapsed.as_secs_f64();
    let per_second = |cur: u64, prev: u64| {
        if dt > 0.0 {
            cur.saturating_sub(prev) as f64 / dt
        } else {
            0.0
        }
    };
    let counter_rates = after
        .counters
        .iter()
        .enumerate()
        .map(|(index, &cur)| {
            let prev = before.counters.get(index).copied().unwrap_or(cur);
            per_second(cur, prev)
        })
        .collect();

    let mover = is_sm103_expert_mover(after).then(|| {
        let swaps_completed = after.counters[0];
        let bytes_copied = after.counters[1];
        let rcu_waits = after.counters[2];
        let skipped_already_resident = after.counters[3];
        let last_packed = after.counters[4];
        let predict_hits = after.counters[5];
        let predict_misses = after.counters[6];
        let predict_enqueues = after.counters[7];

        let (last_layer, last_expert) = if swaps_completed > 0 || last_packed != 0 {
            (
                Some((last_packed >> 32) as u32),
                Some((last_packed & 0xFFFF_FFFF) as u32),
            )
        } else {
            (None, None)
        };

        let total_routes = predict_hits.saturating_add(predict_misses);
        let cumulative_hit_pct = (total_routes > 0)
            .then(|| (predict_hits as f64) * 100.0 / (total_routes as f64));

        let before_swaps = before.counters.first().copied().unwrap_or(swaps_completed);
        let before_bytes = before.counters.get(1).copied().unwrap_or(bytes_copied);
        let before_rcu = before.counters.get(2).copied().unwrap_or(rcu_waits);
        let before_hits = before.counters.get(5).copied().unwrap_or(predict_hits);
        let before_misses = before.counters.get(6).copied().unwrap_or(predict_misses);
        let before_enqueues = before.counters.get(7).copied().unwrap_or(predict_enqueues);

        let dh = predict_hits.saturating_sub(before_hits);
        let dm = predict_misses.saturating_sub(before_misses);
        let recent_hit_pct = if dh.saturating_add(dm) > 0 {
            Some((dh as f64) * 100.0 / (dh.saturating_add(dm) as f64))
        } else {
            previous_recent_hit_pct.or(cumulative_hit_pct)
        };

        MoverTelemetry {
            swaps_completed,
            bytes_copied,
            rcu_waits,
            skipped_already_resident,
            last_packed,
            last_layer,
            last_expert,
            predict_hits,
            predict_misses,
            predict_enqueues,
            cumulative_hit_pct,
            recent_hit_pct,
            swaps_per_sec: per_second(swaps_completed, before_swaps),
            bytes_per_sec: per_second(bytes_copied, before_bytes),
            rcu_waits_per_sec: per_second(rcu_waits, before_rcu),
            enqueues_per_sec: per_second(predict_enqueues, before_enqueues),
        }
    });

    WorkerRates {
        counter_rates,
        mover,
    }
}

fn state_badge(state: WorkerState) -> (&'static str, Color) {
    match state {
        WorkerState::Idle => ("IDLE", CYAN),
        WorkerState::Running => ("RUNNING", GREEN),
        WorkerState::Draining => ("DRAINING", GOLD),
        WorkerState::Stopped => ("STOPPED", MUTED),
        WorkerState::FailedDrain => ("FAILED DRAIN", GOLD),
    }
}

fn hit_rate_color(pct: Option<f64>) -> Color {
    match pct {
        Some(p) if p >= 95.0 => GREEN,
        Some(p) if p >= 85.0 => CYAN,
        Some(_) => GOLD,
        None => MUTED,
    }
}

fn format_last_swap(layer: Option<u32>, expert: Option<u32>) -> String {
    match (layer, expert) {
        (Some(l), Some(e)) => format!("L{l:02}:E{e}"),
        _ => "—".into(),
    }
}

fn format_bytes_decimal(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes} B")
    }
}

fn format_commas(value: u64) -> String {
    let s = value.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn mover_snapshot(
        state: WorkerState,
        swaps: u64,
        bytes: u64,
        rcu: u64,
        skipped: u64,
        layer: u32,
        expert: u32,
        hits: u64,
        misses: u64,
        enqueues: u64,
    ) -> PersistentWorkerSnapshot {
        PersistentWorkerSnapshot {
            name: "sm103_expert_mover".into(),
            worker_sm_count: 8,
            state,
            counters: vec![
                swaps,
                bytes,
                rcu,
                skipped,
                ((layer as u64) << 32) | (expert as u64),
                hits,
                misses,
                enqueues,
            ],
        }
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn mover_rates_compute_window_and_cumulative_hit_rates_and_last_swap() {
        let start = Instant::now();
        let mut view = WorkerView::new(start);
        let first = mover_snapshot(
            WorkerState::Running,
            100,
            2_000_000_000,
            5,
            20,
            3,
            17,
            900,
            100,
            120,
        );
        view.sample(start, Some(first));
        let m0 = view.rates().mover.unwrap();
        assert!((m0.cumulative_hit_pct.unwrap() - 90.0).abs() < 1e-6);
        assert_eq!(m0.swaps_per_sec, 0.0);
        assert_eq!((m0.last_layer, m0.last_expert), (Some(3), Some(17)));

        // Advance 2 seconds: +200 swaps (100/s), +4 GB (2 GB/s), +980 hits / +20 misses (98% live).
        let second = mover_snapshot(
            WorkerState::Running,
            300,
            6_000_000_000,
            15,
            35,
            14,
            203,
            1880,
            120,
            360,
        );
        view.sample(start + Duration::from_secs(2), Some(second));
        let m1 = view.rates().mover.unwrap();
        assert!((m1.recent_hit_pct.unwrap() - 98.0).abs() < 1e-6);
        assert!((m1.cumulative_hit_pct.unwrap() - 94.0).abs() < 1e-6);
        assert!((m1.swaps_per_sec - 100.0).abs() < 1e-6);
        assert!((m1.bytes_per_sec - 2_000_000_000.0).abs() < 1e-3);
        assert!((m1.rcu_waits_per_sec - 5.0).abs() < 1e-6);
        assert!((m1.enqueues_per_sec - 120.0).abs() < 1e-6);
        assert_eq!((m1.last_layer, m1.last_expert), (Some(14), Some(203)));

        // Counter regression (e.g. server restart) resets rolling baseline without panicking.
        let restarted = mover_snapshot(WorkerState::Running, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        view.sample(start + Duration::from_secs(3), Some(restarted));
        let m2 = view.rates().mover.unwrap();
        assert_eq!(m2.swaps_per_sec, 0.0);
        assert_eq!(m2.cumulative_hit_pct, None);
        assert_eq!((m2.last_layer, m2.last_expert), (None, None));
    }

    #[test]
    fn mover_and_generic_views_render_cleanly_in_full_and_overview_panels() {
        let start = Instant::now();
        let mut mover_view = WorkerView::new(start);
        mover_view.sample(
            start,
            Some(mover_snapshot(
                WorkerState::Running,
                10,
                500_000_000,
                1,
                2,
                0,
                0,
                95,
                5,
                12,
            )),
        );
        mover_view.sample(
            start + Duration::from_secs(1),
            Some(mover_snapshot(
                WorkerState::Running,
                60,
                2_500_000_000,
                3,
                8,
                14,
                203,
                195,
                5,
                72,
            )),
        );

        let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
        terminal
            .draw(|frame| mover_view.render_full(frame, frame.area()))
            .unwrap();
        let full_text = buffer_text(&terminal);
        assert!(full_text.contains("PARALLAX"));
        assert!(full_text.contains("sm103_expert_mover"));
        assert!(full_text.contains("RUNNING"));
        assert!(full_text.contains("8 SMs"));
        assert!(full_text.contains("HBM HIT RATE"));
        assert!(full_text.contains("L14:E203"));
        assert!(full_text.contains("2.00 GB/s"));

        let mut overview_term = Terminal::new(TestBackend::new(120, 4)).unwrap();
        overview_term
            .draw(|frame| mover_view.render_overview_panel(frame, frame.area()))
            .unwrap();
        let overview_text = buffer_text(&overview_term);
        assert!(overview_text.contains("PARALLAX"));
        assert!(overview_text.contains("L14:E203"));
        assert_eq!(mover_view.tab_label(), " Parallax ");

        // Generic arbitrary worker renders c0..cN without assuming mover semantics.
        let mut generic_view = WorkerView::new(start);
        generic_view.sample(
            start,
            Some(PersistentWorkerSnapshot {
                name: "custom_dma_worker".into(),
                worker_sm_count: 4,
                state: WorkerState::Draining,
                counters: vec![10, 20, 30],
            }),
        );
        generic_view.sample(
            start + Duration::from_secs(1),
            Some(PersistentWorkerSnapshot {
                name: "custom_dma_worker".into(),
                worker_sm_count: 4,
                state: WorkerState::Draining,
                counters: vec![25, 50, 30],
            }),
        );
        let mut gen_term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        gen_term
            .draw(|frame| generic_view.render_full(frame, frame.area()))
            .unwrap();
        let gen_text = buffer_text(&gen_term);
        assert!(gen_text.contains("custom_dma_worker"));
        assert!(gen_text.contains("DRAINING"));
        assert!(gen_text.contains("4 SMs"));
        assert!(gen_text.contains("c0"));
        assert!(gen_text.contains("c2"));
        assert_eq!(generic_view.tab_label(), " Worker ");
    }
}
