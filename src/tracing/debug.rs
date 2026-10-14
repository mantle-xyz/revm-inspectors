use crate::tracing::{
    FourByteInspector, MuxInspector, TracingInspector, TracingInspectorConfig, TransactionContext,
};
#[cfg(feature = "js-tracer")]
use alloc::boxed::Box;
use alloy_rpc_types_eth::TransactionInfo;
use alloy_rpc_types_trace::geth::{
    erc7562::Erc7562Config, mux::MuxConfig, CallConfig, FourByteFrame, GethDebugBuiltInTracerType,
    GethDebugTracerType, GethDebugTracingOptions, GethDefaultTracingOptions, GethTrace, NoopFrame,
    PreStateConfig,
};
use revm::{
    context_interface::{
        result::{HaltReasonTr, ResultAndState},
        Block, ContextTr, Transaction,
    },
    inspector::JournalExt,
    interpreter::{CallInputs, CallOutcome, CreateInputs, CreateOutcome, Interpreter},
    primitives::{Address, Log, U256},
    DatabaseRef, Inspector,
};
use thiserror::Error;

/// The timeout go-ethereum applies to a JS tracer when the request does not set one.
#[cfg(feature = "js-tracer")]
const DEFAULT_JS_TRACER_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(5);

/// Parses a duration the way Go's `time.ParseDuration` does, the form `debug_trace*`'s `timeout`
/// takes: an optional sign, then one or more decimal numbers each with a unit (`ns`,
/// `us`/`µs`/`μs`, `ms`, `s`, `m`, `h`), as in `"1m30s"` or `"1.5h"`; a bare `"0"` is also
/// accepted. Returns `None` for anything Go rejects. A negative duration has already expired, so it
/// comes back as zero.
fn parse_go_duration(s: &str) -> Option<core::time::Duration> {
    // Go's limit: the magnitude of an `int64` count of nanoseconds.
    const MAX_NANOS: u128 = 1 << 63;

    let (negative, mut rest) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    if rest == "0" {
        return Some(core::time::Duration::ZERO);
    }
    if rest.is_empty() {
        return None;
    }

    let mut total: u128 = 0;
    while !rest.is_empty() {
        // The integer part, then an optional fraction; at least one digit overall.
        let int_len = rest.bytes().take_while(u8::is_ascii_digit).count();
        let (int_digits, after) = rest.split_at(int_len);
        let (frac_digits, after) = match after.strip_prefix('.') {
            Some(after) => {
                let frac_len = after.bytes().take_while(u8::is_ascii_digit).count();
                after.split_at(frac_len)
            }
            None => ("", after),
        };
        if int_digits.is_empty() && frac_digits.is_empty() {
            return None;
        }

        // The unit runs up to the next number.
        let unit_len = after.find(|c: char| c == '.' || c.is_ascii_digit()).unwrap_or(after.len());
        let (unit, after) = after.split_at(unit_len);
        let nanos_per_unit: u128 = match unit {
            "ns" => 1,
            "us" | "µs" | "μs" => 1_000,
            "ms" => 1_000_000,
            "s" => 1_000_000_000,
            "m" => 60_000_000_000,
            "h" => 3_600_000_000_000,
            _ => return None,
        };

        let mut int_part: u128 = 0;
        for digit in int_digits.bytes() {
            int_part = int_part * 10 + u128::from(digit - b'0');
            if int_part > MAX_NANOS {
                return None;
            }
        }
        // Like Go, digits past what can matter are dropped rather than rejected.
        let (mut frac, mut scale) = (0u128, 1u128);
        for digit in frac_digits.bytes().take(18) {
            frac = frac * 10 + u128::from(digit - b'0');
            scale *= 10;
        }

        total += int_part.checked_mul(nanos_per_unit)? + frac * nanos_per_unit / scale;
        if total > MAX_NANOS {
            return None;
        }
        rest = after;
    }

    if negative {
        return Some(core::time::Duration::ZERO);
    }
    if total == MAX_NANOS {
        return None;
    }
    Some(core::time::Duration::from_nanos(total as u64))
}

/// Inspector for the `debug` API
///
/// This inspector is used to trace the execution of a transaction or call and supports all variants
/// of [`GethDebugTracerType`].
///
/// This inspector can be re-used for tracing multiple transactions. This is supported by
/// requiring caller to invoke [`DebugInspector::fuse`] after each transaction. See method
/// documentation for more details.
#[derive(Debug)]
pub enum DebugInspector {
    /// FourByte tracer
    FourByte(FourByteInspector),
    /// CallTracer
    CallTracer(TracingInspector, CallConfig),
    /// PreStateTracer
    PreStateTracer(TracingInspector, PreStateConfig),
    /// Noop tracer
    Noop(revm::inspector::NoOpInspector),
    /// Mux tracer
    Mux(MuxInspector, MuxConfig),
    /// FlatCallTracer
    FlatCallTracer(TracingInspector),
    /// Erc7562Tracer
    Erc7562Tracer(TracingInspector, Erc7562Config),
    /// Default tracer
    Default(TracingInspector, GethDefaultTracingOptions),
    #[cfg(feature = "js-tracer")]
    /// JS tracer
    Js(Box<crate::tracing::js::JsInspector>),
}

impl DebugInspector {
    /// Creates a fresh copy of this inspector, resetting all execution state.
    pub fn try_clone(&self) -> Result<Self, DebugInspectorError> {
        Ok(match self {
            Self::FourByte(inspector) => Self::FourByte(inspector.clone()),
            Self::CallTracer(inspector, config) => Self::CallTracer(inspector.clone(), *config),
            Self::PreStateTracer(inspector, config) => {
                Self::PreStateTracer(inspector.clone(), *config)
            }
            Self::Noop(inspector) => Self::Noop(*inspector),
            Self::Mux(inspector, config) => Self::Mux(inspector.clone(), config.clone()),
            Self::FlatCallTracer(inspector) => Self::FlatCallTracer(inspector.clone()),
            Self::Erc7562Tracer(inspector, config) => {
                Self::Erc7562Tracer(inspector.clone(), config.clone())
            }
            Self::Default(inspector, config) => Self::Default(inspector.clone(), *config),
            #[cfg(feature = "js-tracer")]
            Self::Js(inspector) => Self::Js(inspector.try_clone()?.into()),
        })
    }

    /// Create a new `DebugInspector` from the given tracing options.
    pub fn new(opts: GethDebugTracingOptions) -> Result<Self, DebugInspectorError> {
        let GethDebugTracingOptions { config, tracer, tracer_config, timeout, .. } = opts;
        // go-ethereum rejects a `timeout` it cannot parse whichever tracer is asked for.
        let timeout = timeout
            .as_deref()
            .map(|timeout| {
                parse_go_duration(timeout).ok_or(DebugInspectorError::InvalidTracerConfig)
            })
            .transpose()?;
        // Only the JS tracer enforces it.
        #[cfg(not(feature = "js-tracer"))]
        let _ = timeout;

        let this = if let Some(tracer) = tracer {
            #[allow(unreachable_patterns)]
            match tracer {
                GethDebugTracerType::BuiltInTracer(tracer) => match tracer {
                    GethDebugBuiltInTracerType::FourByteTracer => {
                        Self::FourByte(FourByteInspector::default())
                    }
                    GethDebugBuiltInTracerType::CallTracer => {
                        let config = tracer_config
                            .into_call_config()
                            .map_err(|_| DebugInspectorError::InvalidTracerConfig)?;

                        Self::CallTracer(
                            TracingInspector::new(TracingInspectorConfig::from_geth_call_config(
                                &config,
                            )),
                            config,
                        )
                    }
                    GethDebugBuiltInTracerType::PreStateTracer => {
                        let config = tracer_config
                            .into_pre_state_config()
                            .map_err(|_| DebugInspectorError::InvalidTracerConfig)?;

                        Self::PreStateTracer(
                            TracingInspector::new(
                                TracingInspectorConfig::from_geth_prestate_config(&config),
                            ),
                            config,
                        )
                    }
                    GethDebugBuiltInTracerType::NoopTracer => {
                        Self::Noop(revm::inspector::NoOpInspector)
                    }
                    GethDebugBuiltInTracerType::MuxTracer => {
                        let config = tracer_config
                            .into_mux_config()
                            .map_err(|_| DebugInspectorError::InvalidTracerConfig)?;

                        Self::Mux(MuxInspector::try_from_config(config.clone())?, config)
                    }
                    GethDebugBuiltInTracerType::FlatCallTracer => {
                        let flat_call_config = tracer_config
                            .into_flat_call_config()
                            .map_err(|_| DebugInspectorError::InvalidTracerConfig)?;

                        Self::FlatCallTracer(TracingInspector::new(
                            TracingInspectorConfig::from_flat_call_config(&flat_call_config),
                        ))
                    }
                    GethDebugBuiltInTracerType::Erc7562Tracer => {
                        let config = if tracer_config.is_null() {
                            Erc7562Config::default()
                        } else {
                            tracer_config
                                .from_value()
                                .map_err(|_| DebugInspectorError::InvalidTracerConfig)?
                        };

                        Self::Erc7562Tracer(
                            TracingInspector::new(
                                TracingInspectorConfig::from_geth_erc7562_config(&config),
                            ),
                            config,
                        )
                    }
                    _ => {
                        // Note: this match is non-exhaustive in case we need to add support for
                        // additional tracers
                        return Err(DebugInspectorError::UnsupportedTracer);
                    }
                },
                #[cfg(not(feature = "js-tracer"))]
                GethDebugTracerType::JsTracer(_) => {
                    return Err(DebugInspectorError::JsTracerNotEnabled);
                }
                #[cfg(feature = "js-tracer")]
                GethDebugTracerType::JsTracer(code) => {
                    let config = tracer_config.into_json();
                    // go-ethereum applies a five-second timeout to JS tracers unless the request
                    // overrides it.
                    let timeout = timeout.unwrap_or(DEFAULT_JS_TRACER_TIMEOUT);
                    let inspector =
                        crate::tracing::js::JsInspector::new(code, config)?.with_timeout(timeout);
                    Self::Js(inspector.into())
                }
                _ => {
                    // Note: this match is non-exhaustive in case we need to add support for
                    // additional tracers
                    return Err(DebugInspectorError::UnsupportedTracer);
                }
            }
        } else {
            Self::Default(
                TracingInspector::new(TracingInspectorConfig::from_geth_config(&config)),
                config,
            )
        };

        Ok(this)
    }

    /// Prepares inspector for executing the next transaction. This will remove any state from
    /// previous transactions.
    pub fn fuse(&mut self) -> Result<(), DebugInspectorError> {
        match self {
            Self::FourByte(inspector) => {
                core::mem::take(inspector);
            }
            Self::CallTracer(inspector, _)
            | Self::PreStateTracer(inspector, _)
            | Self::FlatCallTracer(inspector)
            | Self::Erc7562Tracer(inspector, _)
            | Self::Default(inspector, _) => inspector.fuse(),
            Self::Noop(_) => {}
            Self::Mux(inspector, config) => {
                *inspector = MuxInspector::try_from_config(config.clone())?;
            }
            #[cfg(feature = "js-tracer")]
            Self::Js(inspector) => {
                *inspector = inspector.try_clone()?.into();
            }
        }

        Ok(())
    }

    /// Should be invoked after each transaction to obtain the resulting [`GethTrace`].
    pub fn get_result<DB: DatabaseRef>(
        &mut self,
        tx_context: Option<TransactionContext>,
        tx_env: &impl Transaction,
        block_env: &impl Block,
        res: &ResultAndState<impl HaltReasonTr>,
        db: &mut DB,
    ) -> Result<GethTrace, DebugInspectorError<DB::Error>> {
        #[allow(clippy::needless_update)]
        let tx_info = TransactionInfo {
            hash: tx_context.as_ref().and_then(|c| c.tx_hash),
            index: tx_context.as_ref().and_then(|c| c.tx_index.map(|i| i as u64)),
            block_hash: tx_context.as_ref().and_then(|c| c.block_hash),
            block_number: Some(block_env.number().saturating_to()),
            base_fee: Some(block_env.basefee()),
            ..Default::default()
        };

        let res = match self {
            Self::FourByte(inspector) => FourByteFrame::from(&*inspector).into(),
            Self::CallTracer(inspector, config) => {
                inspector.set_transaction_gas_limit(tx_env.gas_limit());
                inspector.set_transaction_caller(tx_env.caller());
                inspector.geth_builder().geth_call_traces(*config, res.result.tx_gas_used()).into()
            }
            Self::PreStateTracer(inspector, config) => {
                inspector.set_transaction_gas_limit(tx_env.gas_limit());
                inspector
                    .geth_builder()
                    .geth_prestate_traces(res, config, db)
                    .map_err(DebugInspectorError::Database)?
                    .into()
            }
            Self::Noop(_) => NoopFrame::default().into(),
            Self::Mux(inspector, _) => {
                // The wrapped tracers need the same transaction-level root frame every other
                // branch here sets, otherwise e.g. `callTracer` reports a different `gas` under
                // `muxTracer` than it does on its own.
                inspector.set_transaction_gas_limit(tx_env.gas_limit());
                inspector.set_transaction_caller(tx_env.caller());
                inspector
                    .try_into_mux_frame(res, db, tx_info)
                    .map_err(DebugInspectorError::Database)?
                    .into()
            }
            Self::FlatCallTracer(inspector) => {
                inspector.set_transaction_gas_limit(tx_env.gas_limit());
                inspector.set_transaction_caller(tx_env.caller());
                inspector
                    .clone()
                    .into_parity_builder()
                    .with_transaction_gas_used(res.result.tx_gas_used())
                    .into_localized_transaction_traces(tx_info)
                    .into()
            }
            Self::Erc7562Tracer(inspector, config) => {
                inspector.set_transaction_gas_limit(tx_env.gas_limit());
                inspector.set_transaction_caller(tx_env.caller());
                inspector
                    .geth_builder()
                    .geth_erc7562_traces(config.clone(), res.result.tx_gas_used(), db)
                    .into()
            }
            Self::Default(inspector, config) => {
                inspector.set_transaction_gas_limit(tx_env.gas_limit());
                inspector.set_transaction_caller(tx_env.caller());
                inspector
                    .geth_builder()
                    .geth_traces(
                        res.result.tx_gas_used(),
                        res.result.output().unwrap_or_default().clone(),
                        *config,
                    )
                    .into()
            }
            #[cfg(feature = "js-tracer")]
            Self::Js(inspector) => {
                inspector.set_transaction_context(tx_context.unwrap_or_default());
                let res = inspector
                    .json_result(res.clone(), tx_env, block_env, db)
                    .map_err(DebugInspectorError::JsInspector)?;

                GethTrace::JS(res)
            }
        };

        Ok(res)
    }
}

macro_rules! delegate {
    ($self:expr => $insp:ident.$method:ident($($arg:expr),*)) => {
        match $self {
            Self::FourByte($insp) => Inspector::<CTX>::$method($insp, $($arg),*),
            Self::CallTracer($insp, _) => Inspector::<CTX>::$method($insp, $($arg),*),
            Self::PreStateTracer($insp, _) => Inspector::<CTX>::$method($insp, $($arg),*),
            Self::FlatCallTracer($insp) => Inspector::<CTX>::$method($insp, $($arg),*),
            Self::Erc7562Tracer($insp, _) => Inspector::<CTX>::$method($insp, $($arg),*),
            Self::Default($insp, _) => Inspector::<CTX>::$method($insp, $($arg),*),
            Self::Noop($insp) => Inspector::<CTX>::$method($insp, $($arg),*),
            Self::Mux($insp, _) => Inspector::<CTX>::$method($insp, $($arg),*),
            #[cfg(feature = "js-tracer")]
            Self::Js($insp) => Inspector::<CTX>::$method($insp, $($arg),*),
        }
    };
}

impl<CTX> Inspector<CTX> for DebugInspector
where
    CTX: ContextTr<Journal: JournalExt, Db: DatabaseRef>,
{
    fn initialize_interp(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        delegate!(self => inspector.initialize_interp(interp, context))
    }

    fn step(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        delegate!(self => inspector.step(interp, context))
    }

    fn step_end(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        delegate!(self => inspector.step_end(interp, context))
    }

    fn log(&mut self, context: &mut CTX, log: Log) {
        delegate!(self => inspector.log(context, log))
    }

    fn log_full(&mut self, interp: &mut Interpreter, context: &mut CTX, log: Log) {
        delegate!(self => inspector.log_full(interp, context, log))
    }

    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        delegate!(self => inspector.call(context, inputs))
    }

    fn call_end(&mut self, context: &mut CTX, inputs: &CallInputs, outcome: &mut CallOutcome) {
        delegate!(self => inspector.call_end(context, inputs, outcome))
    }

    fn create(&mut self, context: &mut CTX, inputs: &mut CreateInputs) -> Option<CreateOutcome> {
        delegate!(self => inspector.create(context, inputs))
    }

    fn create_end(
        &mut self,
        context: &mut CTX,
        inputs: &CreateInputs,
        outcome: &mut CreateOutcome,
    ) {
        delegate!(self => inspector.create_end(context, inputs, outcome))
    }

    fn selfdestruct(&mut self, contract: Address, target: Address, value: U256) {
        delegate!(self => inspector.selfdestruct(contract, target, value))
    }
}

/// Error type for [DebugInspector]
#[derive(Debug, Error)]
pub enum DebugInspectorError<DBError = core::convert::Infallible> {
    /// Invalid tracer configuration
    #[error("invalid tracer config")]
    InvalidTracerConfig,
    /// Unsupported tracer
    #[error("unsupported tracer")]
    UnsupportedTracer,
    /// JS tracer is not enabled
    #[error("JS Tracer is not enabled")]
    JsTracerNotEnabled,
    /// Error from MuxInspector
    #[error(transparent)]
    MuxInspector(#[from] crate::tracing::MuxError),
    /// Error from JS inspector
    #[cfg(feature = "js-tracer")]
    #[error(transparent)]
    JsInspector(#[from] crate::tracing::js::JsInspectorError),
    /// Database error
    #[error("database error: {0}")]
    Database(DBError),
}

#[cfg(test)]
mod tests {
    use super::{parse_go_duration, DebugInspector, DebugInspectorError};
    use alloy_rpc_types_trace::geth::{
        GethDebugBuiltInTracerType, GethDebugTracerType, GethDebugTracingOptions,
    };
    use core::time::Duration;

    /// A `timeout` that does not parse is rejected, as geth rejects it, instead of being replaced
    /// by the default, for the JS tracer and the built-in ones alike.
    #[test]
    fn rejects_an_unparseable_timeout() {
        let tracers = [
            GethDebugTracerType::BuiltInTracer(GethDebugBuiltInTracerType::CallTracer),
            #[cfg(feature = "js-tracer")]
            GethDebugTracerType::JsTracer("{fault: function() {}, result: function() {}}".into()),
        ];
        for tracer in tracers {
            let opts = |timeout: &str| GethDebugTracingOptions {
                tracer: Some(tracer.clone()),
                timeout: Some(timeout.into()),
                ..Default::default()
            };
            assert!(
                matches!(
                    DebugInspector::new(opts("nonsense")),
                    Err(DebugInspectorError::InvalidTracerConfig)
                ),
                "{tracer:?}"
            );
            assert!(DebugInspector::new(opts("1m30s")).is_ok(), "{tracer:?}");
        }
    }

    #[test]
    fn parses_go_durations() {
        assert_eq!(parse_go_duration("5s"), Some(Duration::from_secs(5)));
        assert_eq!(parse_go_duration("200ms"), Some(Duration::from_millis(200)));
        assert_eq!(parse_go_duration("2m"), Some(Duration::from_secs(120)));
        assert_eq!(parse_go_duration("1.5s"), Some(Duration::from_millis(1500)));
        assert_eq!(parse_go_duration(".5s"), Some(Duration::from_millis(500)));
        assert_eq!(parse_go_duration("1.s"), Some(Duration::from_secs(1)));
        assert_eq!(parse_go_duration("+3s"), Some(Duration::from_secs(3)));
        // The forms Go's `Duration.String()` itself produces.
        assert_eq!(parse_go_duration("1m30s"), Some(Duration::from_secs(90)));
        assert_eq!(parse_go_duration("1h2m3.5s"), Some(Duration::from_millis(3_723_500)));
        assert_eq!(parse_go_duration("100us"), Some(Duration::from_micros(100)));
        assert_eq!(parse_go_duration("100µs"), Some(Duration::from_micros(100)));
        assert_eq!(parse_go_duration("100μs"), Some(Duration::from_micros(100)));
        assert_eq!(parse_go_duration("0"), Some(Duration::ZERO));
        assert_eq!(parse_go_duration("0s"), Some(Duration::ZERO));
        assert_eq!(parse_go_duration("-0"), Some(Duration::ZERO));
        assert_eq!(parse_go_duration("+0"), Some(Duration::ZERO));
        assert_eq!(parse_go_duration("1.0000000000000000000001s"), Some(Duration::from_secs(1)));
        assert_eq!(
            parse_go_duration("9223372036854775807ns"),
            Some(Duration::from_nanos(i64::MAX as u64))
        );
        // Already expired.
        assert_eq!(parse_go_duration("-1s"), Some(Duration::ZERO));
        assert_eq!(
            parse_go_duration("2562047h47m16.854775807s"),
            Some(Duration::from_nanos(i64::MAX as u64))
        );
    }

    #[test]
    fn rejects_what_go_rejects() {
        for s in [
            "",
            "abc",
            "10",
            "10x",
            "s",
            ".s",
            "-",
            " 3s ",
            "1m30",
            "2562047h47m16.854775808s",
            "9223372036854775808ns",
        ] {
            assert_eq!(parse_go_duration(s), None, "{s:?}");
        }
    }
}
