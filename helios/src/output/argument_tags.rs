/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! GLM think envelopes and raw-string/JSON argument tags from the chat template.

use crate::{
    Result,
    output::tagged::{RawArguments, ToolDialect, parameter_value, strict_nonstring_raw},
};
use serde_json::Value;

#[cfg(test)]
use crate::output::tagged::{OutputConfig, TaggedOutput};

/// The argument tags that `ArgumentTags::parse` reads. They are markers of the parser that
/// `ToolDialect` does not carry; `OutputPipeline::declared_markers` lists them.
pub(crate) const ARGUMENT_MARKERS: [&str; 4] =
    ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"];

/// The argument tags that `HyArgumentTags::parse` reads.
pub(crate) const HY_ARGUMENT_MARKERS: [&str; 4] = [
    "<arg_key:opensource>",
    "</arg_key:opensource>",
    "<arg_value:opensource>",
    "</arg_value:opensource>",
];

const HY_CALL_OPEN: &str = "<tool_call:opensource>";
const HY_CALL_CLOSE: &str = "</tool_call:opensource>";

fn parse_argument_body(
    body: &str,
    tools: &[Value],
    key_open: &str,
    key_close: &str,
    value_open: &str,
    value_close: &str,
    label: &str,
) -> Result<Option<(String, String)>> {
    let body = body.trim();
    let split = body.find(key_open).unwrap_or(body.len());
    let name = body[..split].trim();
    if name.is_empty() {
        return Err("tool function name is empty".into());
    }
    let mut rest = &body[split..];
    let function = tools
        .iter()
        .filter_map(|tool| tool.get("function"))
        .find(|function| function.get("name").and_then(Value::as_str) == Some(name));
    let strict = function.is_some_and(|f| f["strict"].as_bool() == Some(true));
    let properties = function
        .and_then(|function| function.get("parameters"))
        .and_then(|parameters| parameters.get("properties"))
        .and_then(Value::as_object);
    let mut arguments = RawArguments::new();
    while !rest.trim().is_empty() {
        let key = rest
            .trim_start()
            .strip_prefix(key_open)
            .ok_or_else(|| format!("{label} tool call requires an argument key"))?;
        let (key, tail) = key
            .split_once(key_close)
            .ok_or_else(|| format!("unterminated {label} argument key"))?;
        if key.is_empty() || arguments.contains_key(key) {
            return Err(format!("{label} argument names must be nonempty and unique").into());
        }
        let value = tail
            .trim_start()
            .strip_prefix(value_open)
            .ok_or_else(|| format!("{label} argument requires a value tag"))?;
        let schema = properties.and_then(|p| p.get(key));
        let Ok(container) = super::tagged::container_parameter(value, schema.filter(|_| strict))
        else {
            return Ok(None);
        };
        if let Some((parsed, remaining)) = container
            && let Some(next) = remaining.strip_prefix(value_close)
        {
            arguments.insert(key.to_owned(), parsed);
            rest = next;
            continue;
        }
        let (value, tail) = value
            .split_once(value_close)
            .ok_or_else(|| format!("unterminated {label} argument value"))?;
        if let Some(raw_json) = strict_nonstring_raw(value, schema.filter(|_| strict)) {
            arguments.insert(key.to_owned(), raw_json);
        } else {
            arguments.insert_value(key.to_owned(), &parameter_value(value, schema)?)?;
        }
        rest = tail;
    }
    Ok(Some((name.to_owned(), arguments.to_json_string()?)))
}

fn starts_argument_call(text: &str, open: &str, key_open: &str, close: &str) -> Option<bool> {
    // Calls continue with the function name and an argument key or closing tag;
    // quoted tags followed by prose remain text.
    let text = text.strip_prefix(open).unwrap_or(text).trim_start();
    if text.is_empty() {
        return None;
    }
    let end = text.find(|c: char| c == '<' || c == '>' || c.is_whitespace())?;
    if end == 0 {
        return Some(false);
    }
    let next = text[end..].trim_start();
    if [key_open, close].iter().any(|tag| next.starts_with(tag)) {
        Some(true)
    } else if [key_open, close].iter().any(|tag| tag.starts_with(next)) {
        None
    } else {
        Some(false)
    }
}

pub struct ArgumentTags;
impl ToolDialect for ArgumentTags {
    const NAME: &'static str = "GLM";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        Self::parse_candidate(body, tools)?.ok_or_else(|| "incomplete JSON argument".into())
    }
    fn parse_candidate(body: &str, tools: &[Value]) -> Result<Option<Vec<(String, String)>>> {
        Ok(parse_argument_body(
            body,
            tools,
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
            "GLM",
        )?
        .map(|call| vec![call]))
    }
    fn starts_call(text: &str) -> Option<bool> {
        starts_argument_call(text, Self::OPEN, "<arg_key>", Self::CLOSE)
    }
}

pub struct HyArgumentTags;
impl ToolDialect for HyArgumentTags {
    const NAME: &'static str = "Hy";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<tool_calls:opensource>";
    const CLOSE: &'static str = "</tool_calls:opensource>";
    const FUNCTION: Option<(&'static str, &'static str)> = Some((HY_CALL_OPEN, HY_CALL_CLOSE));

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        Self::parse_candidate(body, tools)?.ok_or_else(|| "incomplete JSON argument".into())
    }
    fn parse_candidate(body: &str, tools: &[Value]) -> Result<Option<Vec<(String, String)>>> {
        let rest = body.trim();
        if rest.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let inner = rest
            .strip_prefix(HY_CALL_OPEN)
            .ok_or("Hy tool call requires a <tool_call:opensource> tag")?
            .strip_suffix(HY_CALL_CLOSE)
            .ok_or("unterminated Hy tool call")?;
        Ok(parse_argument_body(
            inner,
            tools,
            HY_ARGUMENT_MARKERS[0],
            HY_ARGUMENT_MARKERS[1],
            HY_ARGUMENT_MARKERS[2],
            HY_ARGUMENT_MARKERS[3],
            "Hy",
        )?
        .map(|call| vec![call]))
    }
    fn starts_call(text: &str) -> Option<bool> {
        if let Some(after_outer) = text.strip_prefix(Self::OPEN) {
            let next = after_outer.trim_start();
            if next.is_empty() {
                return None;
            }
            if next.starts_with(Self::CLOSE) {
                return Some(true);
            }
            if next.starts_with(HY_CALL_OPEN) {
                return starts_argument_call(next, HY_CALL_OPEN, HY_ARGUMENT_MARKERS[0], HY_CALL_CLOSE);
            }
            if [HY_CALL_OPEN, Self::CLOSE]
                .iter()
                .any(|tag| tag.starts_with(next))
            {
                return None;
            }
            return Some(false);
        }
        starts_argument_call(text, HY_CALL_OPEN, HY_ARGUMENT_MARKERS[0], HY_CALL_CLOSE)
    }
}

#[cfg(test)]
type ArgumentTagOutput = TaggedOutput<ArgumentTags>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        output::{Input, ModelOutput, Termination},
        server::output::Accumulator,
    };
    use serde_json::json;

    fn config() -> OutputConfig {
        OutputConfig {
            structured: false,
            start: crate::output::tagged::OutputStart::Reasoning,
            tools: vec![json!({"type":"function", "function": {
            "name":"run", "parameters":{"type":"object", "properties":{
                "text":{"type":"string"}, "count":{"type":"integer"}, "enabled":{"type":"boolean"},
                "items":{"type":"array"}, "options":{"type":"object"}, "empty":{"type":"null"}
            }}}})],
        }
    }

    #[test]
    fn declared_markers_are_the_tags_the_parser_reads() {
        let [key, key_end, value, value_end] = ARGUMENT_MARKERS;
        let parsed = ArgumentTags::parse(
            &format!("run{key}text{key_end}{value}hello{value_end}"),
            &config().tools,
        )
        .unwrap();
        assert_eq!(
            parsed,
            [("run".to_owned(), r#"{"text":"hello"}"#.to_owned())]
        );
        let syntax = serde_json::from_str::<crate::model::PreparedContract>(include_str!(
            "../../tests/support/contracts/glm53.json"
        ))
        .unwrap()
        .frontend;
        let declared = crate::output::OutputPipeline::declared_markers(&syntax);
        assert!(
            ARGUMENT_MARKERS.iter().all(|tag| declared.contains(tag)),
            "{declared:?}"
        );
    }

    #[test]
    fn split_streams_preserve_reasoning_and_decode_typed_tool_arguments() {
        let text = "\nreason α\n</think>\nAnswer\n<tool_call>run<arg_key>text</arg_key><arg_value>\nhello α\n</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>items</arg_key><arg_value>[1,2]</arg_value><arg_key>options</arg_key><arg_value>{\"x\":1}</arg_value><arg_key>empty</arg_key><arg_value>null</arg_value></tool_call><tool_call>run</tool_call>\nend";
        for boundary in text
            .char_indices()
            .map(|(index, _)| index)
            .chain([text.len()])
        {
            let mut parser = ArgumentTagOutput::new(config());
            let mut output = Accumulator::new("test".into());
            output
                .apply(parser.push(Input::Text(&text[..boundary])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[boundary..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "\nreason α\n");
            assert_eq!(message["content"], "\nAnswer\n\nend");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0]["function"]["name"], "run");
            let arguments: Value =
                serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(
                arguments,
                json!({"text":"\nhello α\n", "count":2, "enabled":true, "items":[1,2], "options":{"x":1}, "empty":null})
            );
            assert_eq!(calls[1]["function"]["arguments"], "{}");
        }
    }

    #[test]
    fn malformed_calls_fail() {
        for body in [
            "",
            "run<arg_key></arg_key><arg_value>x</arg_value>",
            "run<arg_key>text</arg_key><arg_value>a</arg_value><arg_key>text</arg_key><arg_value>b</arg_value>",
            "run<arg_key>text</arg_key><arg_value>unfinished",
            "run<arg_key>x</arg_key>text",
        ] {
            assert!(
                ArgumentTags::parse(body, &config().tools).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn invalid_type_declarations_fail_unless_a_valid_branch_matches() {
        let tools = vec![json!({"type":"function", "function": {
        "name":"run", "parameters":{"type":"object", "properties":{
            "bad":{"type":42}, "union":{"anyOf":[{"type":42}]},
            "mixed":{"oneOf":[{"type":42},{"type":"integer"}]}
        }}}})];
        let call = |key: &str, value: &str| {
            ArgumentTags::parse(
                &format!("run<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"),
                &tools,
            )
        };
        for (key, value) in [("bad", "123"), ("union", "123"), ("mixed", "two")] {
            assert!(call(key, value).is_err(), "{key}={value}");
        }
        assert_eq!(call("mixed", "123").unwrap()[0].1, "{\"mixed\":123}");
    }

    #[test]
    fn mismatched_argument_types_retain_generated_values_without_coercion() {
        let tools = vec![json!({"type":"function", "function": {
        "name":"run", "parameters":{"type":"object", "properties":{
            "count":{"type":"integer"}, "enabled":{"type":"boolean"},
            "items":{"type":"array"}, "choice":{"anyOf":[{"type":"integer"},{"type":"null"}]},
            "listed":{"anyOf":[{"type":"object"},{"type":"string"}]}
        }}}})];
        let text = "</think><tool_call>run<arg_key>enabled</arg_key><arg_value>\"true\"</arg_value><arg_key>count</arg_key><arg_value>two</arg_value><arg_key>items</arg_key><arg_value>2.5</arg_value><arg_key>choice</arg_key><arg_value>false</arg_value><arg_key>listed</arg_key><arg_value>3</arg_value></tool_call>";
        for split in text.char_indices().map(|(at, _)| at) {
            let mut parser = ArgumentTagOutput::new(OutputConfig {
                tools: tools.clone(),
                ..config()
            });
            let mut output = Accumulator::new("test".into());
            for part in [&text[..split], &text[split..]] {
                output
                    .apply(parser.push(Input::Text(part)).unwrap())
                    .unwrap();
            }
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            let arguments: Value = serde_json::from_str(
                message["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                arguments,
                json!({"enabled":"true", "count":"two", "items":2.5, "choice":false, "listed":"3"}),
                "split {split}"
            );
        }
    }

    #[test]
    fn prose_tool_tag_mentions_remain_answer_text() {
        let call = "<tool_call>run<arg_key>text</arg_key><arg_value>x</arg_value></tool_call>";
        for (text, content, calls) in [
            (
                "</think>The `<tool_call>` tag wraps calls. END".to_owned(),
                "The `<tool_call>` tag wraps calls. END",
                0,
            ),
            (
                "</think>Write <tool_call> or `<tool_call>run` literally.".to_owned(),
                "Write <tool_call> or `<tool_call>run` literally.",
                0,
            ),
            (
                format!("</think>Quote `<tool_call>` here.\n{call}"),
                "Quote `<tool_call>` here.\n",
                1,
            ),
        ] {
            for split in text.char_indices().map(|(at, _)| at) {
                let mut parser = ArgumentTagOutput::new(config());
                let mut output = Accumulator::new("test".into());
                for part in [&text[..split], &text[split..]] {
                    output
                        .apply(parser.push(Input::Text(part)).unwrap())
                        .unwrap();
                }
                output
                    .apply(parser.finish(Termination::Eos).unwrap())
                    .unwrap();
                output.finish().unwrap();
                let message = output.message();
                assert_eq!(message["content"], content, "{text:?} at {split}");
                let tools = message["tool_calls"]
                    .as_array()
                    .map_or(&[][..], Vec::as_slice);
                assert_eq!(tools.len(), calls, "{text:?} at {split}");
                if let Some(tool) = tools.first() {
                    assert_eq!(tool["function"]["name"], "run");
                    assert_eq!(tool["function"]["arguments"], "{\"text\":\"x\"}");
                }
            }
        }
    }

    #[test]
    fn truncation_and_structured_output_keep_protocol_semantics() {
        let mut parser = ArgumentTagOutput::new(config());
        parser
            .push(Input::Text("</think><tool_call>run<arg_key>text</arg_key>"))
            .unwrap();
        assert!(parser.finish(Termination::OutputLimit).unwrap().is_empty());
        let mut parser = ArgumentTagOutput::new(OutputConfig {
            structured: true,
            start: crate::output::tagged::OutputStart::Content,
            ..config()
        });
        assert_eq!(
            parser
                .push(Input::Text("<think>{\"x\":1}</think>"))
                .unwrap(),
            vec![crate::output::Event::Content(
                "<think>{\"x\":1}</think>".into()
            )]
        );
    }

    #[test]
    fn hy_argument_tags_stream_multiple_calls_and_preserve_prose_mentions() {
        type HyOutput = TaggedOutput<HyArgumentTags>;
        let text = "Answer\n<tool_calls:opensource><tool_call:opensource>run<arg_key:opensource>text</arg_key:opensource><arg_value:opensource>hello α</arg_value:opensource><arg_key:opensource>count</arg_key:opensource><arg_value:opensource>7</arg_value:opensource><arg_key:opensource>options</arg_key:opensource><arg_value:opensource>{\"s\":\"</tool_call:opensource>\"}</arg_value:opensource></tool_call:opensource><tool_call:opensource>run</tool_call:opensource></tool_calls:opensource>\ntail";
        let hy_cfg = OutputConfig {
            structured: false,
            start: crate::output::tagged::OutputStart::Content,
            tools: vec![json!({"type":"function", "function": {
                "name":"run",
                "strict": true,
                "parameters":{"type":"object", "properties":{
                    "text":{"type":"string"}, "count":{"type":"integer"}, "enabled":{"type":"boolean"},
                    "items":{"type":"array"}, "options":{"type":"object"}, "empty":{"type":"null"}
                }}
            }})],
        };
        for split in text.char_indices().map(|(at, _)| at).chain([text.len()]) {
            let mut parser = HyOutput::new(hy_cfg.clone());
            let mut output = Accumulator::new("hy".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["content"], "Answer\n\ntail", "split {split}");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 2, "split {split}");
            assert_eq!(calls[0]["function"]["name"], "run");
            let args: Value = serde_json::from_str(
                calls[0]["function"]["arguments"].as_str().unwrap(),
            )
            .unwrap();
            assert_eq!(
                args,
                json!({"text": "hello α", "count": 7, "options": {"s": "</tool_call:opensource>"}})
            );
            assert_eq!(calls[1]["function"]["name"], "run");
            assert_eq!(calls[1]["function"]["arguments"], "{}");
        }
        // Prose mentions of the wrapper/call tags remain content.
        let prose = "Use `<tool_calls:opensource>` and `<tool_call:opensource>foo` as examples.";
        let mut parser = HyOutput::new(hy_cfg);
        let mut output = Accumulator::new("hy-prose".into());
        for ch in prose.chars() {
            output
                .apply(parser.push(Input::Text(&ch.to_string())).unwrap())
                .unwrap();
        }
        output
            .apply(parser.finish(Termination::Eos).unwrap())
            .unwrap();
        output.finish().unwrap();
        assert_eq!(output.message()["content"], prose);
        assert!(output.message().get("tool_calls").is_none());
    }
}
