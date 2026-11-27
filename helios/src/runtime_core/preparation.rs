/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Validation, legacy projection, inspection formatting, and generic compatibility
//! comparison for `PreparationManifest`.
//!
//! Serving ignores preparation metadata completely. Core combination and cached-weight
//! reuse compare preparation compatibility generically through parameter scopes declared
//! by Ignition, without hardcoding Ignition-specific option names or inheritance rules.

use std::borrow::Cow;
use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::format::invalid;
use super::wire;
use crate::Result;

/// Preparation manifest schema version written by current exporters.
pub const PREPARATION_MANIFEST_VERSION: u32 = 1;

const MAX_PARAMETERS: usize = 512;
const MAX_KEY_BYTES: usize = 128;
const MAX_STRING_BYTES: usize = 4096;
const MAX_LIST_VALUES: usize = 256;
const MAX_VALUE_DEPTH: usize = 4;
const MAX_REPRODUCTION_ARGS: usize = 1024;

fn check_bounded_text(value: &str, limit: usize, what: &str) -> Result<()> {
    if value.len() > limit || value.chars().any(char::is_control) {
        return Err(invalid(format!("invalid {what}")));
    }
    Ok(())
}

fn validate_key(key: &str) -> Result<()> {
    if key.is_empty()
        || key.len() > MAX_KEY_BYTES
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        return Err(invalid(format!("invalid preparation parameter key {key:?}")));
    }
    Ok(())
}

fn validate_value(value: &wire::PreparationValue, depth: usize) -> Result<()> {
    use wire::preparation_value::Value as Kind;
    match value.value.as_ref() {
        Some(Kind::BoolValue(_) | Kind::IntValue(_) | Kind::UintValue(_)) => Ok(()),
        Some(Kind::FloatValue(number)) => {
            if !number.is_finite() {
                return Err(invalid("preparation parameter float value must be finite"));
            }
            Ok(())
        }
        Some(Kind::StringValue(text)) => {
            check_bounded_text(text, MAX_STRING_BYTES, "preparation parameter string value")
        }
        Some(Kind::ListValue(list)) => {
            if depth >= MAX_VALUE_DEPTH {
                return Err(invalid("preparation parameter list nesting exceeds limit"));
            }
            if list.values.len() > MAX_LIST_VALUES {
                return Err(invalid("preparation parameter list exceeds maximum length"));
            }
            for item in &list.values {
                validate_value(item, depth + 1)?;
            }
            Ok(())
        }
        None => Err(invalid("preparation parameter value is required")),
    }
}

/// Validate structural bounds, key ordering, parameter scopes, and values.
pub fn validate_preparation_manifest(manifest: &wire::PreparationManifest) -> Result<()> {
    if manifest.version != PREPARATION_MANIFEST_VERSION {
        return Err(invalid(format!(
            "unsupported preparation manifest version {}",
            manifest.version
        )));
    }
    check_bounded_text(&manifest.recipe, MAX_STRING_BYTES, "preparation recipe")?;
    check_bounded_text(
        &manifest.source_identity,
        MAX_STRING_BYTES,
        "preparation source_identity",
    )?;
    check_bounded_text(
        &manifest.checkpoint_repository,
        MAX_STRING_BYTES,
        "preparation checkpoint_repository",
    )?;
    check_bounded_text(
        &manifest.checkpoint_revision,
        MAX_STRING_BYTES,
        "preparation checkpoint_revision",
    )?;
    if manifest.reproduction_args.len() > MAX_REPRODUCTION_ARGS {
        return Err(invalid("reproduction_args exceeds maximum entry count"));
    }
    for arg in &manifest.reproduction_args {
        if arg.is_empty() {
            return Err(invalid("reproduction_args entries must be non-empty"));
        }
        check_bounded_text(arg, MAX_STRING_BYTES, "reproduction_args entry")?;
    }
    if manifest.parameters.len() > MAX_PARAMETERS {
        return Err(invalid("preparation parameters exceed maximum count"));
    }
    let mut previous_key: Option<&str> = None;
    for parameter in &manifest.parameters {
        validate_key(&parameter.key)?;
        if previous_key.is_some_and(|prev| prev >= parameter.key.as_str()) {
            return Err(invalid(
                "preparation parameters must have unique keys in strictly ascending order",
            ));
        }
        previous_key = Some(&parameter.key);
        let scope = wire::PreparationParameterScope::try_from(parameter.scope)
            .map_err(|_| invalid("unknown preparation parameter scope"))?;
        if scope == wire::PreparationParameterScope::Unspecified {
            return Err(invalid("preparation parameter scope must be specified"));
        }
        let value = parameter
            .value
            .as_ref()
            .ok_or_else(|| invalid("preparation parameter value is required"))?;
        validate_value(value, 0)?;
    }
    Ok(())
}

fn param_bool(value: bool) -> wire::PreparationValue {
    wire::PreparationValue {
        value: Some(wire::preparation_value::Value::BoolValue(value)),
    }
}

fn param_string(value: impl Into<String>) -> wire::PreparationValue {
    wire::PreparationValue {
        value: Some(wire::preparation_value::Value::StringValue(value.into())),
    }
}

/// Project a deprecated `RebuildManifest` into its canonical semantic
/// `PreparationManifest` form for cross-version core combination and
/// cached-weight reuse.
#[allow(deprecated)]
pub fn from_legacy_rebuild_manifest(legacy: &wire::RebuildManifest) -> wire::PreparationManifest {
    use wire::PreparationParameterScope as Scope;

    let mut parameters = BTreeMap::<String, (Scope, wire::PreparationValue)>::new();
    let mut insert = |key: &str, scope: Scope, value: wire::PreparationValue| {
        parameters.insert(key.to_owned(), (scope, value));
    };

    // Target-model weight semantics.
    insert(
        "online_mxfp8",
        Scope::TargetWeight,
        param_bool(legacy.online_mxfp8),
    );
    insert(
        "online_mxfp8_scope",
        Scope::TargetWeight,
        param_string(if legacy.online_mxfp8_scope.is_empty() {
            "all"
        } else {
            &legacy.online_mxfp8_scope
        }),
    );
    if let Some(compression) = &legacy.expert_scale_compression {
        insert(
            "expert_scale_compression",
            Scope::TargetWeight,
            param_string(compression.clone()),
        );
    }

    // Shared core-combination semantics.
    insert(
        "language_model_only",
        Scope::Combine,
        param_bool(legacy.language_model_only),
    );
    insert(
        "mla_compute_mode",
        Scope::Combine,
        param_string(legacy.mla_compute_mode.clone()),
    );
    if let Some(speculation) = &legacy.speculation {
        insert(
            "speculation.vocabulary_parallel_sampling",
            Scope::Combine,
            param_bool(speculation.vocabulary_parallel_sampling),
        );
        if !speculation.proposal_source.is_empty() {
            insert(
                "speculation.proposal_source",
                Scope::Combine,
                param_string(speculation.proposal_source.clone()),
            );
        }
        if speculation.history_drafts {
            insert("speculation.history_drafts", Scope::Combine, param_bool(true));
        }

        // Draft-model weight semantics.
        insert("speculation.has_draft", Scope::DraftWeight, param_bool(true));
        if let Some(quant) = legacy.effective_draft_body_quantization() {
            insert(
                "speculation.draft_body_quantization",
                Scope::DraftWeight,
                param_string(quant),
            );
        }
        if let Some(quant) = &speculation.draft_head_quantization {
            insert(
                "speculation.draft_head_quantization",
                Scope::DraftWeight,
                param_string(quant.clone()),
            );
        }
        if let Some(quant) = &speculation.markov_head_quantization {
            insert(
                "speculation.markov_head_quantization",
                Scope::DraftWeight,
                param_string(quant.clone()),
            );
        }
    }

    wire::PreparationManifest {
        version: PREPARATION_MANIFEST_VERSION,
        recipe: legacy.recipe.clone(),
        source_identity: legacy.source_identity.clone(),
        checkpoint_repository: legacy.checkpoint_repository.clone(),
        checkpoint_revision: legacy.checkpoint_revision.clone(),
        local_checkpoint: legacy.local_checkpoint,
        reproduction_args: legacy.reproduction_args.clone(),
        parameters: parameters
            .into_iter()
            .map(|(key, (scope, value))| wire::PreparationParameter {
                key,
                scope: scope as i32,
                value: Some(value),
            })
            .collect(),
    }
}

/// Resolve either a `PreparationManifest` or a deprecated `RebuildManifest` into
/// one validated preparation view for compatibility checks.
#[allow(deprecated)]
pub fn resolve_preparation<'a>(
    preparation: Option<&'a wire::PreparationManifest>,
    legacy: Option<&'a wire::RebuildManifest>,
) -> Result<Option<Cow<'a, wire::PreparationManifest>>> {
    match (preparation, legacy) {
        (Some(_), Some(_)) => Err(invalid(
            "deployment cannot carry both preparation and deprecated rebuild metadata",
        )),
        (Some(manifest), None) => {
            validate_preparation_manifest(manifest)?;
            Ok(Some(Cow::Borrowed(manifest)))
        }
        (None, Some(rebuild)) => Ok(Some(Cow::Owned(from_legacy_rebuild_manifest(rebuild)))),
        (None, None) => Ok(None),
    }
}

fn same_provenance(a: &wire::PreparationManifest, b: &wire::PreparationManifest) -> bool {
    a.recipe == b.recipe
        && a.checkpoint_repository == b.checkpoint_repository
        && a.checkpoint_revision == b.checkpoint_revision
        && a.local_checkpoint == b.local_checkpoint
}

fn scope_slice(
    manifest: &wire::PreparationManifest,
    include: impl Fn(i32) -> bool,
) -> Vec<(&str, i32, Option<&wire::PreparationValue>)> {
    manifest
        .parameters
        .iter()
        .filter(|parameter| include(parameter.scope))
        .map(|parameter| {
            (
                parameter.key.as_str(),
                parameter.scope,
                parameter.value.as_ref(),
            )
        })
        .collect()
}

/// Compare two resolved preparation manifests for `ignition combine`.
///
/// Provenance (`recipe`, checkpoint repository/revision, and local-checkpoint flag)
/// and all non-deployment-scoped parameters (`combine`, `target_weight`, and
/// `draft_weight`) must agree. Preparer `source_identity`, `reproduction_args`, and
/// `deployment`-scoped parameters may differ between combined deployments.
pub fn compatible_for_combine(
    a: Option<&wire::PreparationManifest>,
    b: Option<&wire::PreparationManifest>,
) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return a.is_none() && b.is_none();
    };
    let deployment = wire::PreparationParameterScope::Deployment as i32;
    same_provenance(a, b)
        && scope_slice(a, |scope| scope != deployment)
            == scope_slice(b, |scope| scope != deployment)
}

/// Compare a cached deployment's preparation manifest against the current export's
/// preparation manifest for `--reuse-weights`.
///
/// Provenance and `target_weight`-scoped parameters must agree. `draft_weight`-scoped
/// parameters must agree whenever the current export declares any draft-weight
/// parameter; target-only exports (for example `--no-mtp`) omit draft-weight
/// parameters and can reuse target weights from a cached core that also has a draft.
pub fn compatible_for_weight_reuse(
    previous: &wire::PreparationManifest,
    expected: &wire::PreparationManifest,
) -> bool {
    if !same_provenance(previous, expected) {
        return false;
    }
    let target = wire::PreparationParameterScope::TargetWeight as i32;
    if scope_slice(previous, |scope| scope == target)
        != scope_slice(expected, |scope| scope == target)
    {
        return false;
    }
    let draft = wire::PreparationParameterScope::DraftWeight as i32;
    let expected_draft = scope_slice(expected, |scope| scope == draft);
    expected_draft.is_empty()
        || scope_slice(previous, |scope| scope == draft) == expected_draft
}

/// Render a `PreparationManifest` into a structured JSON object for `flare inspect`.
/// Never fails on unfamiliar versions or scopes.
pub(crate) fn preparation_json(manifest: &wire::PreparationManifest) -> Value {
    json!({
        "version": manifest.version,
        "recipe": manifest.recipe,
        "source_identity": manifest.source_identity,
        "checkpoint_repository": manifest.checkpoint_repository,
        "checkpoint_revision": manifest.checkpoint_revision,
        "local_checkpoint": manifest.local_checkpoint,
        "reproduction_args": manifest.reproduction_args,
        "parameters": manifest
            .parameters
            .iter()
            .map(|parameter| {
                let scope = match wire::PreparationParameterScope::try_from(parameter.scope) {
                    Ok(wire::PreparationParameterScope::Deployment) => "deployment",
                    Ok(wire::PreparationParameterScope::Combine) => "combine",
                    Ok(wire::PreparationParameterScope::TargetWeight) => "target_weight",
                    Ok(wire::PreparationParameterScope::DraftWeight) => "draft_weight",
                    Ok(wire::PreparationParameterScope::Unspecified) | Err(_) => "unspecified",
                };
                json!({
                    "key": parameter.key,
                    "scope": scope,
                    "value": parameter
                        .value
                        .as_ref()
                        .map(preparation_value_json)
                        .unwrap_or(Value::Null),
                })
            })
            .collect::<Vec<_>>(),
    })
}

fn preparation_value_json(value: &wire::PreparationValue) -> Value {
    use wire::preparation_value::Value as Kind;
    match value.value.as_ref() {
        Some(Kind::BoolValue(value)) => Value::Bool(*value),
        Some(Kind::IntValue(value)) => json!(*value),
        Some(Kind::UintValue(value)) => json!(*value),
        Some(Kind::FloatValue(value)) => serde_json::Number::from_f64(*value)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(Kind::StringValue(value)) => Value::String(value.clone()),
        Some(Kind::ListValue(list)) => {
            Value::Array(list.values.iter().map(preparation_value_json).collect())
        }
        None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn param(
        key: &str,
        scope: wire::PreparationParameterScope,
        value: wire::PreparationValue,
    ) -> wire::PreparationParameter {
        wire::PreparationParameter {
            key: key.into(),
            scope: scope as i32,
            value: Some(value),
        }
    }

    fn base_manifest() -> wire::PreparationManifest {
        wire::PreparationManifest {
            version: PREPARATION_MANIFEST_VERSION,
            recipe: "fixture".into(),
            source_identity: "rev-a".into(),
            checkpoint_repository: "owner/model".into(),
            checkpoint_revision: "abc123".into(),
            local_checkpoint: false,
            reproduction_args: vec!["ignition".into(), "export".into(), "owner/model@abc123".into()],
            parameters: vec![
                param(
                    "language_model_only",
                    wire::PreparationParameterScope::Combine,
                    param_bool(false),
                ),
                param(
                    "max_concurrency",
                    wire::PreparationParameterScope::Deployment,
                    wire::PreparationValue {
                        value: Some(wire::preparation_value::Value::UintValue(8)),
                    },
                ),
                param(
                    "mla_compute_mode",
                    wire::PreparationParameterScope::Combine,
                    param_string("auto"),
                ),
                param(
                    "online_mxfp8",
                    wire::PreparationParameterScope::TargetWeight,
                    param_bool(true),
                ),
                param(
                    "online_mxfp8_scope",
                    wire::PreparationParameterScope::TargetWeight,
                    param_string("all"),
                ),
            ],
        }
    }

    #[test]
    fn validation_accepts_sorted_typed_parameters_and_rejects_malformed_inputs() {
        let valid = base_manifest();
        validate_preparation_manifest(&valid).unwrap();

        let encoded = valid.encode_to_vec();
        let decoded = wire::PreparationManifest::decode(encoded.as_slice()).unwrap();
        assert_eq!(decoded, valid);
        validate_preparation_manifest(&decoded).unwrap();

        let mut bad_version = valid.clone();
        bad_version.version = 2;
        assert!(validate_preparation_manifest(&bad_version).is_err());

        let mut unsorted = valid.clone();
        unsorted.parameters.swap(0, 1);
        assert!(validate_preparation_manifest(&unsorted).is_err());

        let mut duplicate = valid.clone();
        duplicate.parameters.push(duplicate.parameters.last().unwrap().clone());
        assert!(validate_preparation_manifest(&duplicate).is_err());

        let mut unspecified_scope = valid.clone();
        unspecified_scope.parameters[0].scope =
            wire::PreparationParameterScope::Unspecified as i32;
        assert!(validate_preparation_manifest(&unspecified_scope).is_err());

        let mut missing_value = valid.clone();
        missing_value.parameters[0].value = None;
        assert!(validate_preparation_manifest(&missing_value).is_err());

        let mut nan_float = valid.clone();
        nan_float.parameters[0].value = Some(wire::PreparationValue {
            value: Some(wire::preparation_value::Value::FloatValue(f64::NAN)),
        });
        assert!(validate_preparation_manifest(&nan_float).is_err());

        let mut empty_arg = valid.clone();
        empty_arg.reproduction_args.push(String::new());
        assert!(validate_preparation_manifest(&empty_arg).is_err());
    }

    #[test]
    fn scope_compatibility_ignores_deployment_params_and_distinguishes_draft_reuse() {
        let mut first = base_manifest();
        first.parameters.push(param(
            "speculation.draft_body_quantization",
            wire::PreparationParameterScope::DraftWeight,
            param_string("mxfp8"),
        ));
        first.parameters.push(param(
            "speculation.has_draft",
            wire::PreparationParameterScope::DraftWeight,
            param_bool(true),
        ));
        validate_preparation_manifest(&first).unwrap();

        let mut second = first.clone();
        second.source_identity = "rev-b".into();
        second.reproduction_args.push("--row-shard-prefill".into());
        // Adding a new deployment-only Ignition parameter does not break combine or reuse.
        second.parameters.push(param(
            "z_future_ignition_flag",
            wire::PreparationParameterScope::Deployment,
            param_bool(true),
        ));
        validate_preparation_manifest(&second).unwrap();
        assert!(compatible_for_combine(Some(&first), Some(&second)));
        assert!(compatible_for_weight_reuse(&first, &second));

        // Changing a target-weight parameter rejects both combine and reuse.
        let mut changed_target = first.clone();
        changed_target
            .parameters
            .iter_mut()
            .find(|p| p.key == "online_mxfp8_scope")
            .unwrap()
            .value = Some(param_string("kda-input"));
        assert!(!compatible_for_combine(Some(&first), Some(&changed_target)));
        assert!(!compatible_for_weight_reuse(&first, &changed_target));

        // Changing a combine-only parameter rejects combine while preserving target weight reuse.
        let mut changed_combine = first.clone();
        changed_combine
            .parameters
            .iter_mut()
            .find(|p| p.key == "language_model_only")
            .unwrap()
            .value = Some(param_bool(true));
        assert!(!compatible_for_combine(Some(&first), Some(&changed_combine)));
        assert!(compatible_for_weight_reuse(&first, &changed_combine));

        // Target-only export (--no-mtp) can reuse target weights from a draft-bearing cache,
        // while a draft-bearing export cannot reuse from a target-only cache.
        let target_only = base_manifest();
        assert!(!compatible_for_combine(Some(&first), Some(&target_only)));
        assert!(compatible_for_weight_reuse(&first, &target_only));
        assert!(!compatible_for_weight_reuse(&target_only, &first));
    }

    #[test]
    #[allow(deprecated)]
    fn legacy_rebuild_projection_matches_equivalent_preparation_manifest() {
        let legacy = wire::RebuildManifest {
            version: 1,
            recipe: "fixture".into(),
            source_identity: "old-rev".into(),
            checkpoint_repository: "owner/model".into(),
            checkpoint_revision: "abc123".into(),
            online_mxfp8: true,
            online_mxfp8_scope: String::new(),
            mla_compute_mode: "auto".into(),
            speculation: Some(wire::RebuildSpeculation {
                method: "dflash2".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let projected = from_legacy_rebuild_manifest(&legacy);
        validate_preparation_manifest(&projected).unwrap();

        let mut modern = base_manifest();
        modern.parameters.push(param(
            "speculation.draft_body_quantization",
            wire::PreparationParameterScope::DraftWeight,
            param_string("mxfp8"),
        ));
        modern.parameters.push(param(
            "speculation.has_draft",
            wire::PreparationParameterScope::DraftWeight,
            param_bool(true),
        ));
        modern.parameters.push(param(
            "speculation.vocabulary_parallel_sampling",
            wire::PreparationParameterScope::Combine,
            param_bool(false),
        ));
        validate_preparation_manifest(&modern).unwrap();
        assert!(compatible_for_combine(Some(&projected), Some(&modern)));
        assert!(compatible_for_weight_reuse(&projected, &modern));
        assert!(compatible_for_weight_reuse(&modern, &projected));
    }
}
