/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! `flare parallax`: live terminal monitor for Parallax and persistent-worker telemetry.

use std::{
    io::{self, IsTerminal},
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

use flare_helios::server::view::{
    BACKGROUND, CYAN, GOLD, GREEN, MUTED, TEXT, WorkerTelemetryResponse, WorkerView, crossterm,
    ratatui,
};

use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

use crate::cli::ParallaxArgs;

struct Screen;

impl Drop for Screen {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stderr(), Show, LeaveAlternateScreen);
    }
}

#[derive(Clone, Debug, PartialEq)]
enum PollStatus {
    Connecting,
    Live { uptime_secs: f64 },
    NoWorker,
    Error(String),
}

#[derive(Clone, Debug, PartialEq)]
enum PollOutcome {
    Snapshot(WorkerTelemetryResponse),
    NoWorker,
    Error(String),
}

pub(crate) fn endpoint_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1/worker") {
        trimmed.to_owned()
    } else {
        format!("{trimmed}/v1/worker")
    }
}

fn http_client(timeout_secs: u64) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(timeout_secs)))
        .build()
        .into()
}

fn poll_endpoint(client: &ureq::Agent, endpoint: &str) -> PollOutcome {
    match client.get(endpoint).call() {
        Ok(mut response) => match response.body_mut().read_to_string() {
            Ok(body) => match serde_json::from_str::<WorkerTelemetryResponse>(&body) {
                Ok(parsed) => PollOutcome::Snapshot(parsed),
                Err(error) => PollOutcome::Error(format!("invalid JSON: {error}")),
            },
            Err(error) => PollOutcome::Error(error.to_string()),
        },
        Err(ureq::Error::StatusCode(404)) => PollOutcome::NoWorker,
        Err(error) => PollOutcome::Error(error.to_string()),
    }
}

fn draw(
    frame: &mut Frame<'_>,
    base_url: &str,
    elapsed: Duration,
    status: &PollStatus,
    view: &WorkerView,
) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
        area,
    );
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area);
    let age = elapsed.as_secs();
    let (badge, badge_color) = match status {
        PollStatus::Connecting => ("● CONNECTING".into(), CYAN),
        PollStatus::Live { .. } => ("● LIVE".into(), GREEN),
        PollStatus::NoWorker => ("● NO WORKER".into(), GOLD),
        PollStatus::Error(err) => (format!("● RECONNECTING ({err})"), GOLD),
    };
    let uptime_str = match status {
        PollStatus::Live { uptime_secs } => {
            let up = *uptime_secs as u64;
            format!("  up {:02}:{:02}:{:02}", up / 3600, up / 60 % 60, up % 60)
        }
        _ => String::new(),
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " FLARE ",
                Style::default()
                    .fg(BACKGROUND)
                    .bg(CYAN)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" / PARALLAX  ", Style::default().fg(CYAN)),
            Span::styled(badge, Style::default().fg(badge_color)),
            Span::styled(
                format!(
                    "    {:02}:{:02}:{:02}{uptime_str}    {base_url}    [q] Quit",
                    age / 3600,
                    age / 60 % 60,
                    age % 60,
                ),
                Style::default().fg(MUTED),
            ),
        ])),
        rows[0],
    );

    if matches!(status, PollStatus::NoWorker) && !view.is_present() {
        frame.render_widget(
            Paragraph::new(vec![
                Line::default(),
                Line::from(Span::styled(
                    " Target server is reachable, but the loaded core has no persistent worker.",
                    Style::default().fg(GOLD),
                )),
                Line::from(Span::styled(
                    format!(" Endpoint: {}", endpoint_url(base_url)),
                    Style::default().fg(MUTED),
                )),
            ]),
            rows[1],
        );
        return;
    }

    view.render_full(frame, rows[1]);
}

pub fn run(args: ParallaxArgs) -> Result<(), Box<dyn std::error::Error>> {
    let base_url = args.url.trim().trim_end_matches('/').to_owned();
    let endpoint = endpoint_url(&base_url);
    let client = http_client(3);

    if args.json {
        return match poll_endpoint(&client, &endpoint) {
            PollOutcome::Snapshot(response) => {
                println!("{}", serde_json::to_string_pretty(&response)?);
                Ok(())
            }
            PollOutcome::NoWorker => {
                Err("no persistent worker is configured on the target server".into())
            }
            PollOutcome::Error(error) => Err(error.into()),
        };
    }

    if args.plain || !io::stderr().is_terminal() {
        return match poll_endpoint(&client, &endpoint) {
            PollOutcome::Snapshot(response) => {
                let now = Instant::now();
                let mut view = WorkerView::new(now);
                view.sample(now, Some(response.snapshot()));
                print!("{}", view.format_plain_summary());
                Ok(())
            }
            PollOutcome::NoWorker => {
                Err("no persistent worker is configured on the target server".into())
            }
            PollOutcome::Error(error) => Err(error.into()),
        };
    }

    #[cfg(unix)]
    // SAFETY: The handler only stores to the process-wide atomic stop flag.
    unsafe {
        libc::signal(libc::SIGINT, crate::stop as *const () as usize);
        libc::signal(libc::SIGTERM, crate::stop as *const () as usize);
    }

    enable_raw_mode()?;
    let _screen = Screen;
    execute!(io::stderr(), EnterAlternateScreen, Hide)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stderr()))?;
    terminal.clear()?;

    let poll_interval = Duration::from_millis(args.interval_ms);
    let start = Instant::now();
    let mut view = WorkerView::new(start);
    let mut status = PollStatus::Connecting;
    let mut last_poll: Option<Instant> = None;

    loop {
        if crate::STOP.load(Ordering::Relaxed) {
            break;
        }
        let now = Instant::now();
        if last_poll.is_none_or(|prev| now.duration_since(prev) >= poll_interval) {
            match poll_endpoint(&client, &endpoint) {
                PollOutcome::Snapshot(response) => {
                    status = PollStatus::Live {
                        uptime_secs: response.uptime_secs,
                    };
                    view.sample(now, Some(response.snapshot()));
                }
                PollOutcome::NoWorker => {
                    status = PollStatus::NoWorker;
                    view.sample(now, None);
                }
                PollOutcome::Error(error) => {
                    status = PollStatus::Error(error);
                }
            }
            last_poll = Some(now);
        }

        let elapsed = now.saturating_duration_since(start);
        terminal.draw(|frame| draw(frame, &base_url, elapsed, &status, &view))?;

        if event::poll(Duration::from_millis(60))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q' | 'Q') | KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_helios::server::view::{PersistentWorkerSnapshot, WorkerState};
    use ratatui::backend::TestBackend;

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
    fn endpoint_url_normalizes_trailing_slashes_and_existing_path() {
        assert_eq!(
            endpoint_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/v1/worker"
        );
        assert_eq!(
            endpoint_url("http://127.0.0.1:8000/"),
            "http://127.0.0.1:8000/v1/worker"
        );
        assert_eq!(
            endpoint_url("http://127.0.0.1:8000/v1/worker/"),
            "http://127.0.0.1:8000/v1/worker"
        );
    }

    #[test]
    fn monitor_renders_live_mover_delta_rates_and_no_worker_state() {
        let start = Instant::now();
        let mut view = WorkerView::new(start);

        // No-worker state displays clear guidance.
        let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
        terminal
            .draw(|frame| {
                draw(
                    frame,
                    "http://127.0.0.1:8000",
                    Duration::from_secs(5),
                    &PollStatus::NoWorker,
                    &view,
                )
            })
            .unwrap();
        let no_worker_text = buffer_text(&terminal);
        assert!(no_worker_text.contains("PARALLAX"));
        assert!(no_worker_text.contains("NO WORKER"));
        assert!(no_worker_text.contains("no persistent worker"));

        // Live mover updates render hit rate, swap rate, bandwidth, and last swapped expert.
        let first = PersistentWorkerSnapshot {
            name: "sm103_expert_mover".into(),
            worker_sm_count: 8,
            state: WorkerState::Running,
            counters: vec![20, 1_000_000_000, 0, 4, (1u64 << 32) | 5, 190, 10, 24],
        };
        let second = PersistentWorkerSnapshot {
            name: "sm103_expert_mover".into(),
            worker_sm_count: 8,
            state: WorkerState::Running,
            counters: vec![120, 4_000_000_000, 2, 11, (22u64 << 32) | 189, 1170, 30, 144],
        };
        view.sample(start, Some(first));
        view.sample(start + Duration::from_secs(2), Some(second));

        let rates = view.rates().mover.unwrap();
        assert!((rates.swaps_per_sec - 50.0).abs() < 1e-6);
        assert!((rates.bytes_per_sec - 1_500_000_000.0).abs() < 1e-3);
        assert!((rates.recent_hit_pct.unwrap() - 98.0).abs() < 1e-6);

        terminal
            .draw(|frame| {
                draw(
                    frame,
                    "http://127.0.0.1:8000",
                    Duration::from_secs(12),
                    &PollStatus::Live { uptime_secs: 65.0 },
                    &view,
                )
            })
            .unwrap();
        let live_text = buffer_text(&terminal);
        assert!(live_text.contains("PARALLAX"));
        assert!(live_text.contains("LIVE"));
        assert!(live_text.contains("sm103_expert_mover"));
        assert!(live_text.contains("8 SMs"));
        assert!(live_text.contains("L22:E189"));
        assert!(live_text.contains("1.50 GB/s"));
        assert!(live_text.contains("up 00:01:05"));
    }

    #[test]
    fn poll_endpoint_decodes_http_200_json_and_maps_http_404_to_no_worker() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let payload = WorkerTelemetryResponse {
            name: "sm103_expert_mover".into(),
            worker_sm_count: 8,
            state: WorkerState::Running,
            counters: vec![5, 1024, 0, 1, (4u64 << 32) | 12, 80, 20, 9],
            uptime_secs: 42.5,
            timestamp_ms: 1_700_000_000_000,
        };
        let body = serde_json::to_string(&payload).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body_clone = body.clone();
        let server = std::thread::spawn(move || {
            for response_bytes in [
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body_clone.len(),
                    body_clone
                ),
                "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}".to_owned(),
            ] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut req = [0u8; 512];
                let _ = stream.read(&mut req);
                stream.write_all(response_bytes.as_bytes()).unwrap();
            }
        });

        let client = http_client(2);
        let url = endpoint_url(&format!("http://{addr}"));
        assert_eq!(poll_endpoint(&client, &url), PollOutcome::Snapshot(payload));
        assert_eq!(poll_endpoint(&client, &url), PollOutcome::NoWorker);
        server.join().unwrap();
    }
}
