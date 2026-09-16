//! Javascript inspector

use crate::tracing::{
    config::TraceStyle,
    js::{
        bindings::{
            CallFrame, Contract, EvmDbRef, FrameResult, GcGuard, JsEvmContext, MemoryRef,
            MemorySnapshot, MemoryView, StackRef, StepLog,
        },
        builtins::{register_builtins, to_serde_value, PrecompileList},
    },
    types::CallKind,
    utils, CallInputExt, TransactionContext,
};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use alloy_primitives::{Address, Bytes, U256};
pub use boa_engine::vm::RuntimeLimits;
use boa_engine::{js_string, Context, JsError, JsNativeError, JsObject, JsResult, JsValue, Source};
use core::borrow::Borrow;
use revm::{
    context::JournalTr,
    context_interface::{
        result::{ExecutionResult, HaltReasonTr, Output, ResultAndState},
        Block, ContextTr, TransactTo, Transaction,
    },
    inspector::JournalExt,
    interpreter::{
        interpreter_types::{Jumps, LoopControl},
        CallInputs, CallOutcome, CallScheme, CreateInputs, CreateOutcome, Gas, InstructionResult,
        Interpreter, InterpreterAction, InterpreterResult,
    },
    DatabaseRef, Inspector,
};

pub(crate) mod bindings;
pub(crate) mod builtins;

/// The maximum number of iterations in a loop.
///
/// Once exceeded, the loop will throw an error.
// An empty loop with this limit takes around 50ms to fail.
pub const LOOP_ITERATION_LIMIT: u64 = 200_000;

/// The recursion limit for function calls.
///
/// Once exceeded, the function will throw an error.
pub const RECURSION_LIMIT: usize = 10_000;

/// A javascript inspector that will delegate inspector functions to javascript functions
///
/// See also <https://geth.ethereum.org/docs/developers/evm-tracing/custom-tracer#custom-javascript-tracing>
#[derive(Debug)]
pub struct JsInspector {
    ctx: Context,
    /// The original javascript code used to create this inspector.
    code: String,
    /// The input config object.
    config: serde_json::Value,
    /// The evaluated object that contains the inspector functions.
    obj: JsObject,
    /// The context of the transaction that is being inspected.
    transaction_context: TransactionContext,

    /// The javascript function that will be called when the result is requested.
    result_fn: JsObject,
    fault_fn: JsObject,

    // EVM inspector hook functions
    /// Invoked when the EVM enters a new call that is _NOT_ the top level call.
    ///
    /// Corresponds to [Inspector::call] and [Inspector::create_end] but is also invoked on
    /// [Inspector::selfdestruct].
    enter_fn: Option<JsObject>,
    /// Invoked when the EVM exits a call that is _NOT_ the top level call.
    ///
    /// Corresponds to [Inspector::call_end] and [Inspector::create_end] but also invoked after
    /// selfdestruct.
    exit_fn: Option<JsObject>,
    /// Executed before each instruction is executed.
    step_fn: Option<JsObject>,
    /// Keeps track of the current call stack.
    call_stack: Vec<CallStackItem>,
    /// Marker to track whether the precompiles have been registered.
    precompiles_registered: bool,
    /// Tracker for PC recorded in start_step
    last_start_step_pc: Option<usize>,
    /// Opcode recorded in start_step, so `fault` can report the instruction that failed.
    last_start_step_op: Option<u8>,
    /// Snapshot taken in `step`, consumed by `step_end` once the instruction's cost is known.
    pending_step: Option<PendingStep>,
    /// The first error thrown by one of the tracer's hooks, if any. Mirrors go-ethereum's
    /// `jsTracer.err`: once a hook throws the rest are skipped and [`Self::result`] reports
    /// the error instead of a silently incomplete trace.
    hook_error: Option<JsInspectorError>,
}

impl JsInspector {
    /// Creates a new inspector from a javascript code snipped that evaluates to an object with the
    /// expected fields and a config object.
    ///
    /// The object must have the following fields:
    ///  - `result`: a function that will be called when the result is requested.
    ///  - `fault`: a function that will be called when the transaction fails.
    ///
    /// Optional functions are invoked during inspection:
    /// - `setup`: a function that will be called before the inspection starts.
    /// - `enter`: a function that will be called when the execution enters a new call.
    /// - `exit`: a function that will be called when the execution exits a call.
    /// - `step`: a function that will be called when the execution steps to the next instruction.
    ///
    /// This also accepts a sender half of a channel to communicate with the database service so the
    /// DB can be queried from inside the inspector.
    pub fn new(code: String, config: serde_json::Value) -> Result<Self, JsInspectorError> {
        Self::with_transaction_context(code, config, Default::default())
    }

    /// Creates a new inspector from a javascript code snippet. See also [Self::new].
    ///
    /// This also accepts a [TransactionContext] that gives the JS code access to some contextual
    /// transaction infos.
    pub fn with_transaction_context(
        code: String,
        config: serde_json::Value,
        transaction_context: TransactionContext,
    ) -> Result<Self, JsInspectorError> {
        // Instantiate the execution context
        let mut ctx = Context::default();

        // Apply the default runtime limits
        // This is a safe guard to prevent infinite loops
        ctx.runtime_limits_mut().set_loop_iteration_limit(LOOP_ITERATION_LIMIT);
        ctx.runtime_limits_mut().set_recursion_limit(RECURSION_LIMIT);

        register_builtins(&mut ctx)?;

        // evaluate the code
        let wrapped = format!("({code})");
        let obj =
            ctx.eval(Source::from_bytes(wrapped.as_bytes())).map_err(JsInspectorError::EvalCode)?;

        let obj = obj.as_object().ok_or(JsInspectorError::ExpectedJsObject)?;

        // ensure all the fields are callables, if present

        let result_fn = obj
            .get(js_string!("result"), &mut ctx)?
            .as_object()
            .ok_or(JsInspectorError::ResultFunctionMissing)?;
        if !result_fn.is_callable() {
            return Err(JsInspectorError::ResultFunctionMissing);
        }

        let fault_fn = obj
            .get(js_string!("fault"), &mut ctx)?
            .as_object()
            .ok_or(JsInspectorError::FaultFunctionMissing)?;
        if !fault_fn.is_callable() {
            return Err(JsInspectorError::FaultFunctionMissing);
        }

        let enter_fn =
            obj.get(js_string!("enter"), &mut ctx)?.as_object().filter(|o| o.is_callable());
        let exit_fn =
            obj.get(js_string!("exit"), &mut ctx)?.as_object().filter(|o| o.is_callable());
        // Frame tracing needs both halves: a tracer with only one of them silently records
        // half a call tree. geth rejects it at construction for the same reason.
        if enter_fn.is_some() != exit_fn.is_some() {
            return Err(JsInspectorError::UnpairedEnterExit);
        }
        let step_fn =
            obj.get(js_string!("step"), &mut ctx)?.as_object().filter(|o| o.is_callable());

        // Validate the config converts to a JS value, even when no `setup` consumes it.
        JsValue::from_json(&config, &mut ctx).map_err(JsInspectorError::InvalidJsonConfig)?;

        if let Some(setup_fn) = obj.get(js_string!("setup"), &mut ctx)?.as_object() {
            if !setup_fn.is_callable() {
                return Err(JsInspectorError::SetupFunctionNotCallable);
            }

            // geth hands `setup` the raw JSON text rather than a parsed object, defaulting to
            // "{}" when absent, so the documented `JSON.parse(config)` idiom works.
            let cfg = if config.is_null() { String::from("{}") } else { config.to_string() };
            let cfg = JsValue::from(js_string!(cfg));
            setup_fn
                .call(&(obj.clone().into()), core::slice::from_ref(&cfg), &mut ctx)
                .map_err(JsInspectorError::SetupCallFailed)?;
        }

        Ok(Self {
            ctx,
            code,
            config,
            obj,
            transaction_context,
            result_fn,
            fault_fn,
            enter_fn,
            exit_fn,
            step_fn,
            call_stack: Default::default(),
            precompiles_registered: false,
            last_start_step_pc: None,
            last_start_step_op: None,
            pending_step: None,
            hook_error: None,
        })
    }

    /// Returns the config object.
    pub const fn config(&self) -> &serde_json::Value {
        &self.config
    }

    /// Creates a fresh inspector from the same code and config, resetting all execution state.
    pub fn try_clone(&self) -> Result<Self, JsInspectorError> {
        Self::new(self.code.clone(), self.config.clone())
    }

    /// Returns the transaction context.
    pub const fn transaction_context(&self) -> &TransactionContext {
        &self.transaction_context
    }

    /// Sets the transaction context.
    pub fn set_transaction_context(&mut self, transaction_context: TransactionContext) {
        self.transaction_context = transaction_context;
    }

    /// Applies the runtime limits to the JS context.
    ///
    /// By default
    pub fn set_runtime_limits(&mut self, limits: RuntimeLimits) {
        self.ctx.set_runtime_limits(limits);
    }

    /// Calls the result function and returns the result as [serde_json::Value].
    ///
    /// Note: This is supposed to be called after the inspection has finished.
    pub fn json_result<DB>(
        &mut self,
        res: ResultAndState<impl HaltReasonTr>,
        tx: &impl Transaction,
        block: &impl Block,
        db: &DB,
    ) -> Result<serde_json::Value, JsInspectorError>
    where
        DB: DatabaseRef,
        <DB as DatabaseRef>::Error: core::fmt::Display,
    {
        let result = self.result(res, tx, block, db)?;
        Ok(to_serde_value(result, &mut self.ctx)?)
    }

    /// Calls the result function and returns the result.
    pub fn result<TX, DB>(
        &mut self,
        res: ResultAndState<impl HaltReasonTr>,
        tx: &TX,
        block: &impl Block,
        db: &DB,
    ) -> Result<JsValue, JsInspectorError>
    where
        TX: Transaction,
        DB: DatabaseRef,
        <DB as DatabaseRef>::Error: core::fmt::Display,
    {
        // A hook that threw leaves the tracer's state half-built, so report the failure
        // instead of a trace that silently omits whatever the hook did not record. The error
        // is rebuilt rather than taken, so asking for the result twice answers the same way.
        if let Some(JsInspectorError::JsError(err)) = &self.hook_error {
            return Err(JsInspectorError::JsError(err.clone()));
        }

        let ResultAndState { result, state } = res;
        let (db, _db_guard) = EvmDbRef::new(&state, db);

        let gas_used = result.tx_gas_used();
        let mut to = None;
        let mut output_bytes = None;
        let mut error = None;
        match result {
            ExecutionResult::Success { output, .. } => match output {
                Output::Call(out) => {
                    output_bytes = Some(out);
                }
                Output::Create(out, addr) => {
                    to = addr;
                    output_bytes = Some(out);
                }
            },
            ExecutionResult::Revert { output, .. } => {
                error = Some("execution reverted".to_string());
                output_bytes = Some(output);
            }
            ExecutionResult::Halt { reason, .. } => {
                error = Some(format!("execution halted: {reason:?}"));
            }
        };

        if let TransactTo::Call(target) = tx.kind() {
            to = Some(target);
        }

        let ctx = JsEvmContext {
            r#type: match tx.kind() {
                TransactTo::Call(_) => "CALL",
                TransactTo::Create => "CREATE",
            }
            .to_string(),
            from: tx.caller(),
            // geth takes this from the `OnEnter` callback, which for a creation carries the
            // computed address whether or not the deployment succeeded.
            to: to.or_else(|| Some(tx.caller().create(tx.nonce()))),
            input: tx.input().clone(),
            gas: tx.gas_limit(),
            gas_used,
            gas_price: U256::from(effective_gas_tip(tx, block.basefee() as u128)),
            intrinsic_gas: 0,
            value: tx.value(),
            block: block.number().try_into().unwrap_or(u64::MAX),
            coinbase: block.beneficiary(),
            output: output_bytes.unwrap_or_default(),
            time: block.timestamp().to_string(),
            transaction_ctx: self.transaction_context,
            error,
        };
        let ctx = ctx.into_js_object(&mut self.ctx)?;
        let db = db.into_js_object(&mut self.ctx)?;
        Ok(self.result_fn.call(
            &(self.obj.clone().into()),
            &[ctx.into(), db.into()],
            &mut self.ctx,
        )?)
    }

    /// Records the first error thrown by a tracer hook, tagged with the hook's name.
    ///
    /// Later errors are dropped: the first one is the one that explains the rest. The hook name
    /// goes into the message rather than a dedicated variant so that callers mapping
    /// [`JsInspectorError`] onto their own error types treat it as the JavaScript failure it is
    /// - reth's `EthApiError` sends every other variant to `InvalidParams`, which this is not.
    fn record_hook_error(&mut self, hook: &'static str, err: JsError) {
        if self.hook_error.is_none() {
            let message = format!("{err}    in server-side tracer function '{hook}'");
            self.hook_error = Some(JsInspectorError::JsError(JsError::from_native(
                JsNativeError::error().with_message(message),
            )));
        }
    }

    /// Whether a hook has already thrown, in which case the remaining hooks are skipped.
    const fn hook_errored(&self) -> bool {
        self.hook_error.is_some()
    }

    fn try_fault(&mut self, step: StepLog, db: EvmDbRef) -> JsResult<()> {
        if self.hook_errored() {
            return Ok(());
        }
        let step = step.into_js_object(&mut self.ctx)?;
        let db = db.into_js_object(&mut self.ctx)?;
        self.fault_fn.call(&(self.obj.clone().into()), &[step.into(), db.into()], &mut self.ctx)?;
        Ok(())
    }

    fn try_step(&mut self, step: StepLog, db: EvmDbRef) -> JsResult<()> {
        if self.hook_errored() {
            return Ok(());
        }
        if let Some(step_fn) = &self.step_fn {
            let step = step.into_js_object(&mut self.ctx)?;
            let db = db.into_js_object(&mut self.ctx)?;
            step_fn.call(&(self.obj.clone().into()), &[step.into(), db.into()], &mut self.ctx)?;
        }
        Ok(())
    }

    fn try_enter(&mut self, frame: CallFrame) -> JsResult<()> {
        if self.hook_errored() {
            return Ok(());
        }
        if let Some(enter_fn) = &self.enter_fn {
            let frame = frame.into_js_object(&mut self.ctx)?;
            enter_fn.call(&(self.obj.clone().into()), &[frame.into()], &mut self.ctx)?;
        }
        Ok(())
    }

    fn try_exit(&mut self, frame: FrameResult) -> JsResult<()> {
        if self.hook_errored() {
            return Ok(());
        }
        if let Some(exit_fn) = &self.exit_fn {
            let frame = frame.into_js_object(&mut self.ctx)?;
            exit_fn.call(&(self.obj.clone().into()), &[frame.into()], &mut self.ctx)?;
        }
        Ok(())
    }

    /// Returns the currently active call
    ///
    /// Panics: if there's no call yet
    #[track_caller]
    fn active_call(&self) -> &CallStackItem {
        self.call_stack.last().expect("call stack is empty")
    }

    #[inline]
    fn pop_call(&mut self) {
        self.call_stack.pop();
    }

    /// Returns true whether the active call is the root call.
    #[inline]
    fn is_root_call_active(&self) -> bool {
        self.call_stack.len() == 1
    }

    /// Returns true if there's an enter function and the active call is not the root call.
    #[inline]
    fn can_call_enter(&self) -> bool {
        self.enter_fn.is_some() && !self.is_root_call_active()
    }

    /// Returns true if there's an exit function and the active call is not the root call.
    #[inline]
    fn can_call_exit(&mut self) -> bool {
        self.exit_fn.is_some() && !self.is_root_call_active()
    }

    /// Pushes a new call to the stack
    fn push_call(
        &mut self,
        contract: Address,
        input: Bytes,
        value: U256,
        kind: CallKind,
        caller: Address,
        gas_limit: u64,
    ) -> &CallStackItem {
        let call = CallStackItem {
            contract: Contract { caller, contract, value, input },
            kind,
            gas_limit,
        };
        self.call_stack.push(call);
        self.active_call()
    }

    /// Registers the precompiles in the JS context
    fn register_precompiles<CTX: ContextTr<Journal: JournalExt>>(&mut self, context: &mut CTX) {
        if self.precompiles_registered {
            return;
        }
        let precompiles =
            PrecompileList(context.journal().precompile_addresses().iter().copied().collect());

        let _ = precompiles.register_callable(&mut self.ctx);

        self.precompiles_registered = true
    }
}

impl<CTX> Inspector<CTX> for JsInspector
where
    CTX: ContextTr<Journal: JournalExt, Db: DatabaseRef>,
{
    fn step(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        // Recorded unconditionally: `step_end` reports these for a faulting instruction even
        // when the tracer has no `step` hook, which is what geth does.
        self.last_start_step_pc = Some(interp.bytecode.pc());
        self.last_start_step_op = Some(interp.bytecode.opcode());

        if self.step_fn.is_none() {
            return;
        }

        // geth reports the cost of the instruction about to run, which revm only knows once it
        // has run, so the hook fires from `step_end`. By then the live stack and memory describe
        // the wrong point in time, hence the stack copy and the record of the bytes about to be
        // overwritten; `step_end` rebuilds the pre-execution view from those and the live memory.
        let active_call = self.active_call();
        let op = interp.bytecode.opcode();
        let memory = {
            let mem = interp.memory.borrow();
            MemorySnapshot::record(op, interp.stack.data(), &mem.context_memory())
        };
        self.pending_step = Some(PendingStep {
            stack: interp.stack.data().clone(),
            memory,
            op,
            pc: interp.bytecode.pc() as u64,
            gas_remaining: interp.gas.remaining(),
            depth: context.journal_ref().depth() as u64,
            refund: refunded_gas(interp),
            contract: Contract {
                caller: interp.input.caller_address,
                contract: interp.input.target_address,
                value: active_call.contract.value,
                input: active_call.contract.input.clone(),
            },
        });
    }

    fn step_end(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        let fault = interp
            .bytecode
            .action()
            .as_ref()
            .and_then(|a| a.instruction_result())
            .filter(|r| is_fault(*r));

        // The instruction has run, so its cost is now the difference in remaining gas. geth
        // reports this from `OnOpcode`, which it emits after metering but before executing.
        if let Some(pending) = self.pending_step.take() {
            let cost = pending.gas_remaining.saturating_sub(interp.gas.remaining());
            let (db, _db_guard) =
                EvmDbRef::new(context.journal_ref().evm_state(), context.db_ref());
            // Scoped so the guards and the memory borrow are released before `interp` is used
            // mutably below.
            let called = {
                let mem = interp.memory.borrow();
                let post_memory = mem.context_memory();
                let (step, _stack_guard, _memory_guard) =
                    pending.into_step_log(cost, None, &post_memory);
                self.try_step(step, db)
            };
            if let Err(err) = called {
                self.record_hook_error("step", err);
                // Only if the instruction did not already end the frame: setting a second
                // action panics, and a frame that is ending anyway needs no halt.
                if interp.bytecode.action().is_none() {
                    interp.bytecode.set_action(InterpreterAction::new_halt(
                        InstructionResult::Revert,
                        interp.gas,
                    ));
                }
                return;
            }
        }

        let Some(result) = fault else {
            return;
        };

        let (db, _db_guard) = EvmDbRef::new(context.journal_ref().evm_state(), context.db_ref());
        let active_call = self.active_call();
        let mem = interp.memory.borrow();
        let post_memory = mem.context_memory();
        let post_memory: &[u8] = &post_memory;
        // The instruction failed, so there is no "before" to rebuild: geth's `OnFault` likewise
        // reports the state as it stands. An empty snapshot serves the live memory unchanged.
        let (step, _stack_guard, _memory_guard) = PendingStep {
            stack: interp.stack.data().clone(),
            memory: MemorySnapshot::unchanged(post_memory.len()),
            op: self.last_start_step_op.unwrap_or_default(),
            pc: self.last_start_step_pc.unwrap_or_default() as u64,
            gas_remaining: interp.gas.remaining(),
            depth: context.journal_ref().depth() as u64,
            refund: refunded_gas(interp),
            contract: Contract {
                caller: interp.input.caller_address,
                contract: interp.input.target_address,
                value: active_call.contract.value,
                input: active_call.contract.input.clone(),
            },
        }
        .into_step_log(0, utils::fmt_error_msg(result, TraceStyle::Geth), post_memory);

        if let Err(err) = self.try_fault(step, db) {
            self.record_hook_error("fault", err);
        }
    }

    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        self.register_precompiles(context);

        // determine contract and caller based on the call scheme
        let (caller, contract) = match inputs.scheme {
            CallScheme::DelegateCall | CallScheme::CallCode => {
                (inputs.target_address, inputs.bytecode_address)
            }
            _ => (inputs.caller, inputs.target_address),
        };

        // A delegate call carries the parent frame's value as its apparent value, which is what
        // geth reports for it. A static call has none at all.
        let value = inputs.transfer_value().or_else(|| inputs.apparent_value()).unwrap_or_default();
        self.push_call(
            contract,
            inputs.input_data(context),
            value,
            inputs.scheme.into(),
            caller,
            inputs.gas_limit,
        );

        if self.can_call_enter() {
            let call = self.active_call();
            let frame = CallFrame {
                contract: call.contract.clone(),
                kind: call.kind.to_str(),
                gas: inputs.gas_limit,
                // geth passes nil here for a static call, so `getValue()` is `undefined`.
                value: (!matches!(inputs.scheme, CallScheme::StaticCall)).then_some(value),
            };
            if let Err(err) = self.try_enter(frame) {
                self.record_hook_error("enter", err.clone());
                return Some(CallOutcome::new(
                    js_error_to_revert(err),
                    inputs.return_memory_offset.clone(),
                ));
            }
        }

        None
    }

    fn call_end(&mut self, _context: &mut CTX, _inputs: &CallInputs, outcome: &mut CallOutcome) {
        if self.can_call_exit() {
            let frame_result = FrameResult {
                gas_used: outcome.result.gas.total_gas_spent(),
                output: outcome.result.output.clone(),
                error: utils::fmt_error_msg(outcome.result.result, TraceStyle::Geth),
            };
            if let Err(err) = self.try_exit(frame_result) {
                self.record_hook_error("exit", err.clone());
                outcome.result = js_error_to_revert(err);
            }
        }

        self.pop_call();
    }

    fn create(&mut self, context: &mut CTX, inputs: &mut CreateInputs) -> Option<CreateOutcome> {
        self.register_precompiles(context);

        let nonce = context.journal_mut().load_account(inputs.caller()).unwrap().info.nonce;
        let contract = inputs.created_address(nonce);
        self.push_call(
            contract,
            inputs.init_code().clone(),
            inputs.value(),
            inputs.scheme().into(),
            inputs.caller(),
            inputs.gas_limit(),
        );

        if self.can_call_enter() {
            let call = self.active_call();
            let frame = CallFrame {
                contract: call.contract.clone(),
                kind: call.kind.to_str(),
                gas: call.gas_limit,
                value: Some(call.contract.value),
            };
            if let Err(err) = self.try_enter(frame) {
                self.record_hook_error("enter", err.clone());
                return Some(CreateOutcome::new(js_error_to_revert(err), None));
            }
        }

        None
    }

    fn create_end(
        &mut self,
        _context: &mut CTX,
        _inputs: &CreateInputs,
        outcome: &mut CreateOutcome,
    ) {
        if self.can_call_exit() {
            let frame_result = FrameResult {
                gas_used: outcome.result.gas.total_gas_spent(),
                output: outcome.result.output.clone(),
                // geth's `OnExit` reports the error for creates as it does for calls, so a
                // failed deployment must not look successful to the tracer.
                error: utils::fmt_error_msg(outcome.result.result, TraceStyle::Geth),
            };
            if let Err(err) = self.try_exit(frame_result) {
                self.record_hook_error("exit", err.clone());
                outcome.result = js_error_to_revert(err);
            }
        }

        self.pop_call();
    }

    fn selfdestruct(&mut self, contract: Address, target: Address, value: U256) {
        // This is exempt from the root call constraint, because selfdestruct is treated as a
        // new scope that is entered and immediately exited.
        if self.enter_fn.is_some() {
            // geth reports the destroyed contract as the caller and the beneficiary as the
            // callee, with no input and no gas: `OnEnter(depth, SELFDESTRUCT, this,
            // beneficiary, []byte{}, 0, balance)`.
            let frame = CallFrame {
                contract: Contract {
                    caller: contract,
                    contract: target,
                    value,
                    input: Bytes::new(),
                },
                kind: "SELFDESTRUCT",
                gas: 0,
                value: Some(value),
            };
            if let Err(err) = self.try_enter(frame) {
                self.record_hook_error("enter", err);
            }
        }

        // exit with empty frame result ref <https://github.com/ethereum/go-ethereum/blob/0004c6b229b787281760b14fb9460ffd9c2496f1/core/vm/instructions.go#L829-L829>
        if self.exit_fn.is_some() {
            let frame_result = FrameResult { gas_used: 0, output: Bytes::new(), error: None };
            if let Err(err) = self.try_exit(frame_result) {
                self.record_hook_error("exit", err);
            }
        }
    }
}

/// Represents an active call
#[derive(Debug)]
struct CallStackItem {
    contract: Contract,
    kind: CallKind,
    gas_limit: u64,
}

/// Error variants that can occur during JavaScript inspection.
#[derive(Debug, thiserror::Error)]
pub enum JsInspectorError {
    /// Error originating from a JavaScript operation.
    #[error(transparent)]
    JsError(#[from] JsError),

    /// Failure during the evaluation of JavaScript code.
    #[error("failed to evaluate JS code: {0}")]
    EvalCode(JsError),

    /// The evaluated code is not a JavaScript object.
    #[error("the evaluated code is not a JS object")]
    ExpectedJsObject,

    /// The trace object must expose a function named `result()`.
    #[error("trace object must expose a function result()")]
    ResultFunctionMissing,

    /// The trace object must expose a function named `fault()`.
    #[error("trace object must expose a function fault()")]
    FaultFunctionMissing,

    /// The trace object exposed only one of `enter()` and `exit()`.
    #[error("trace object must expose either both or none of enter() and exit()")]
    UnpairedEnterExit,

    /// The setup object must be a callable function.
    #[error("setup object must be a function")]
    SetupFunctionNotCallable,

    /// Failure during the invocation of the `setup()` function.
    #[error("failed to call setup(): {0}")]
    SetupCallFailed(JsError),

    /// Invalid JSON configuration encountered.
    #[error("invalid JSON config: {0}")]
    InvalidJsonConfig(JsError),
}

/// A snapshot of the interpreter taken before an instruction runs.
///
/// The `step` hook needs pre-execution stack and memory but also the instruction's cost, which
/// is only known afterwards, so the two are bridged by copying rather than borrowing.
#[derive(Debug)]
struct PendingStep {
    stack: Vec<U256>,
    memory: MemorySnapshot,
    op: u8,
    pc: u64,
    gas_remaining: u64,
    depth: u64,
    refund: u64,
    contract: Contract,
}

impl PendingStep {
    /// Builds the log handed to JavaScript.
    ///
    /// The returned guards revoke the snapshot's JS-visible handles when dropped, so they must
    /// outlive the hook call.
    fn into_step_log<'a>(
        self,
        cost: u64,
        error: Option<String>,
        post_memory: &'a [u8],
    ) -> (StepLog, GcGuard<'a, Vec<U256>>, GcGuard<'a, MemoryView>) {
        let Self { stack, memory, op, pc, gas_remaining, depth, refund, contract } = self;
        let (stack, stack_guard) = StackRef::new(stack);
        let (memory, memory_guard) = MemoryRef::new(memory, post_memory);
        let step = StepLog {
            stack,
            op: op.into(),
            memory,
            pc,
            gas_remaining,
            cost,
            depth,
            refund,
            error,
            contract,
        };
        (step, stack_guard, memory_guard)
    }
}

/// Whether a terminating instruction result reaches the tracer's `fault` hook.
///
/// geth splits on where the error arose: its interpreter validates the stack and charges gas,
/// then emits `OnOpcode` and sets `logged`, then executes. The deferred handler routes an
/// error to `OnFault` only when `logged` is set, so faults raised while metering never get
/// there. revm meters and executes in one step, so the split has to be reconstructed here.
const fn is_fault(result: InstructionResult) -> bool {
    use InstructionResult::*;
    match result {
        // Normal termination.
        Stop | Return | SelfDestruct => false,
        // Rejected before geth emits `OnOpcode`: stack validation and every flavour of
        // running out of gas, including the memory-sizing overflow geth reports as
        // `ErrGasUintOverflow`.
        StackUnderflow | StackOverflow | OutOfGas | MemoryOOG | MemoryLimitOOG | PrecompileOOG
        | InvalidOperandOOG | ReentrancySentryOOG => false,
        // Everything else is raised while executing the instruction: REVERT, invalid jump,
        // undefined opcode, write protection, return-data overrun, and the create failures.
        _ => true,
    }
}

/// Returns the interpreter's refund counter as an unsigned value, clamping the negative values
/// `Gas::refunded` reports when a storage clear is reversed. Scope still differs from geth's
/// transaction-wide `StateDB.GetRefund()`; aligning needs the journal's transaction-level counter.
fn refunded_gas(interp: &Interpreter) -> u64 {
    interp.gas.refunded().max(0) as u64
}

/// Converts a JavaScript error into a [InstructionResult::Revert] [InterpreterResult].
#[inline]
fn js_error_to_revert(err: JsError) -> InterpreterResult {
    let output = err.to_string().as_bytes().to_vec();
    InterpreterResult { result: InstructionResult::Revert, output: output.into(), gas: Gas::new(0) }
}

/// Returns `min(gasTipCap, gasFeeCap - baseFee)`, what go-ethereum surfaces to a tracer as
/// `ctx.gasPrice`. It is the tip alone, so it is *less* than the sender paid per unit of gas.
/// Legacy and EIP-2930 transactions have no separate tip cap; geth uses their gas price as both.
fn effective_gas_tip(tx: &impl Transaction, base_fee: u128) -> u128 {
    let fee_cap = tx.max_fee_per_gas();
    effective_gas_tip_from(fee_cap, tx.max_priority_fee_per_gas().unwrap_or(fee_cap), base_fee)
}

/// The arithmetic behind [`effective_gas_tip`], split out so it can be tested directly.
fn effective_gas_tip_from(fee_cap: u128, tip_cap: u128, base_fee: u128) -> u128 {
    if fee_cap < base_fee {
        // Reproducing a quirk, not an oversight: geth subtracts in `uint256`, so this wraps to
        // a huge value, the `min` below always picks the tip cap, and `OnTxStart` discards the
        // `ErrGasFeeCapTooLow` that comes with it. Only reachable via `debug_traceCall`.
        tip_cap
    } else {
        tip_cap.min(fee_cap - base_fee)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloy_primitives::{bytes, hex, Address};
    use revm::{
        context::TxEnv,
        database::CacheDB,
        database_interface::EmptyDB,
        inspector::InspectorEvmTr,
        primitives::hardfork::SpecId,
        state::{AccountInfo, Bytecode},
        InspectEvm, MainBuilder, MainContext,
    };
    //use revm_inspector::{inspector_handler, InspectorContext, InspectorMainEvm};
    use serde_json::json;

    #[test]
    fn test_loop_iteration_limit() {
        let mut context = Context::default();
        context.runtime_limits_mut().set_loop_iteration_limit(LOOP_ITERATION_LIMIT);

        let code = "let i = 0; while (i++ < 69) {}";
        let result = context.eval(Source::from_bytes(code));
        assert!(result.is_ok());

        let code = "while (true) {}";
        let result = context.eval(Source::from_bytes(code));
        assert!(result.is_err());
    }

    #[test]
    fn test_fault_fn_not_callable() {
        let code = r#"
            {
                result: function() {},
                fault: {},
            }
        "#;
        let config = serde_json::Value::Null;
        let result = JsInspector::new(code.to_string(), config);
        assert!(matches!(result, Err(JsInspectorError::FaultFunctionMissing)));
    }

    // Helper function to run a trace and return the result
    fn run_trace(code: &str, contract: Option<Bytes>, success: bool) -> serde_json::Value {
        run_trace_with_gas(code, contract, success, 1_000_000)
    }

    fn run_trace_with_gas(
        code: &str,
        contract: Option<Bytes>,
        success: bool,
        gas_limit: u64,
    ) -> serde_json::Value {
        try_run_trace(code, contract, Some(success), gas_limit).expect("tracer should not fail")
    }

    /// Like [`run_trace`], but surfaces a tracer failure instead of panicking on it.
    fn try_run_trace(
        code: &str,
        contract: Option<Bytes>,
        success: Option<bool>,
        gas_limit: u64,
    ) -> Result<serde_json::Value, JsInspectorError> {
        let addr = Address::repeat_byte(0x01);
        let mut db = CacheDB::new(EmptyDB::default());

        // Insert the caller
        db.insert_account_info(
            Address::ZERO,
            AccountInfo { balance: U256::from(1e18), ..Default::default() },
        );
        // Insert the contract
        db.insert_account_info(
            addr,
            AccountInfo {
                code: Some(Bytecode::new_legacy(
                    /* PUSH1 1, PUSH1 1, STOP */
                    contract.unwrap_or_else(|| hex!("6001600100").into()),
                )),
                ..Default::default()
            },
        );

        let insp = JsInspector::new(code.to_string(), serde_json::Value::Null).unwrap();

        let mut evm = revm::Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.spec = SpecId::CANCUN)
            .with_db(db)
            .build_mainnet_with_inspector(insp);

        let res = evm
            .inspect_tx(TxEnv {
                gas_price: 1024,
                gas_limit,
                gas_priority_fee: None,
                kind: TransactTo::Call(addr),
                ..Default::default()
            })
            .expect("pass without error");

        if let Some(success) = success {
            assert_eq!(res.result.is_success(), success);
        }
        let (ctx, inspector) = evm.ctx_inspector();
        inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref())
    }

    /// Asserts that a throwing `step` hook fails the whole trace, naming the hook. geth does
    /// the same: out-of-range accessors call `vm.Interrupt`, the error lands in
    /// `jsTracer.err`, and `GetResult()` returns it rather than a partial trace.
    fn assert_step_hook_fails(code: &str, contract: Option<Bytes>) {
        // Whether the transaction itself fails depends on whether the throw landed before the
        // frame ended, so only the trace outcome is asserted.
        let err = try_run_trace(code, contract, None, 1_000_000)
            .expect_err("a throwing step hook must fail the trace");
        let msg = err.to_string();
        assert!(msg.contains("step"), "error should name the failing hook, got: {msg}");
    }

    /// `ctx.gasPrice` must be the effective *tip* and a big integer, as in go-ethereum. With
    /// base fee 300 and legacy gas price 1000 the candidates are far apart: geth reports
    /// `min(1000, 1000 - 300) = 700`, the price the sender actually pays is the full 1000.
    #[test]
    fn test_ctx_gas_price_is_effective_tip() {
        let addr = Address::repeat_byte(0x01);
        let mut db = CacheDB::new(EmptyDB::default());
        db.insert_account_info(
            Address::ZERO,
            AccountInfo { balance: U256::from(1e18), ..Default::default() },
        );
        db.insert_account_info(
            addr,
            AccountInfo {
                code: Some(Bytecode::new_legacy(hex!("6001600100").into())),
                ..Default::default()
            },
        );

        let code = r#"{
            step: function() {},
            fault: function() {},
            result: function(ctx) {
                return { price: ctx.gasPrice.toString(), kind: typeof ctx.gasPrice };
            }
        }"#;
        let insp = JsInspector::new(code.to_string(), serde_json::Value::Null).unwrap();

        let mut evm = revm::Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.spec = SpecId::CANCUN)
            .modify_block_chained(|block| block.basefee = 300)
            .with_db(db)
            .build_mainnet_with_inspector(insp);

        let res = evm
            .inspect_tx(TxEnv {
                gas_price: 1000,
                gas_limit: 1_000_000,
                gas_priority_fee: None,
                kind: TransactTo::Call(addr),
                ..Default::default()
            })
            .expect("pass without error");

        let (ctx, inspector) = evm.ctx_inspector();
        let res = inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref()).unwrap();

        assert_eq!(res["price"], json!("700"), "gasPrice must exclude the base fee");
        assert_eq!(res["kind"], json!("object"), "gasPrice must be a bigInt, not a JS number");
    }

    /// When the fee cap is below the base fee geth reports the full tip cap, because its
    /// `uint256` subtraction wraps. Saturating to zero is the intuitive reading and the one
    /// that diverges. Reachable via `debug_traceCall`, which does not enforce the base fee.
    #[test]
    fn test_effective_gas_tip_below_base_fee_matches_geth() {
        // (fee cap, tip cap, base fee) -> expected tip
        assert_eq!(effective_gas_tip_from(100, 50, 300), 50, "fee cap under base fee: tip cap");
        assert_eq!(effective_gas_tip_from(1000, 50, 300), 50, "tip cap binds");
        assert_eq!(effective_gas_tip_from(1000, 900, 300), 700, "headroom binds");
        // Legacy and EIP-2930: the gas price stands in for both caps.
        assert_eq!(effective_gas_tip_from(1000, 1000, 300), 700, "legacy");
        assert_eq!(effective_gas_tip_from(0, 0, 0), 0, "zero-priced transaction");
    }

    /// The lazily-defined `ctx` fields must be indistinguishable from ordinary properties:
    /// geth assigns them directly, so reading one twice gives the same object and assigning
    /// sticks. The deferring accessor has to replace itself on first read to preserve both.
    #[test]
    fn test_ctx_lazy_fields_behave_as_plain_properties() {
        let code = r#"{
            step: function() {},
            fault: function() {},
            result: function(ctx) {
                var same = ctx.value === ctx.value;
                ctx.value = 42;
                return { same: same, assigned: ctx.value, price: ctx.gasPrice === ctx.gasPrice };
            }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res["same"], json!(true), "reading twice must yield the same object");
        assert_eq!(res["assigned"], json!(42), "the field must be writable");
        assert_eq!(res["price"], json!(true), "gasPrice must memoize too");
    }

    /// The selfdestruct frame must describe the destruction, not the enclosing call.
    ///
    /// geth emits `OnEnter(depth, SELFDESTRUCT, this, beneficiary, []byte{}, 0, balance)`, so
    /// `getTo()` is the beneficiary. Reporting the enclosing frame misattributes the recipient.
    #[test]
    fn test_selfdestruct_frame_describes_the_destruction() {
        let contract_addr = Address::repeat_byte(0x01);
        let beneficiary = Address::repeat_byte(0x02);

        let mut db = CacheDB::new(EmptyDB::default());
        db.insert_account_info(
            Address::ZERO,
            AccountInfo { balance: U256::from(1e18), ..Default::default() },
        );
        // PUSH20 <beneficiary>, SELFDESTRUCT
        let mut code = vec![0x73];
        code.extend_from_slice(beneficiary.as_slice());
        code.push(0xff);
        db.insert_account_info(
            contract_addr,
            AccountInfo {
                balance: U256::from(1234u64),
                code: Some(Bytecode::new_legacy(code.into())),
                ..Default::default()
            },
        );

        let code = r#"{
            frames: [],
            step: function() {},
            fault: function() {},
            enter: function(frame) {
                this.frames.push({
                    type: frame.getType(),
                    from: toHex(frame.getFrom()),
                    to: toHex(frame.getTo()),
                    value: frame.getValue().toString(),
                    gas: frame.getGas(),
                });
            },
            exit: function() {},
            result: function() { return this.frames }
        }"#;
        let insp = JsInspector::new(code.to_string(), serde_json::Value::Null).unwrap();

        let mut evm = revm::Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.spec = SpecId::CANCUN)
            .with_db(db)
            .build_mainnet_with_inspector(insp);

        let res = evm
            .inspect_tx(TxEnv {
                gas_limit: 1_000_000,
                kind: TransactTo::Call(contract_addr),
                ..Default::default()
            })
            .expect("pass without error");

        let (ctx, inspector) = evm.ctx_inspector();
        let res = inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref()).unwrap();

        assert_eq!(
            res,
            json!([{
                "type": "SELFDESTRUCT",
                "from": contract_addr,
                "to": beneficiary,
                "value": "1234",
                "gas": 0,
            }])
        );
    }

    /// A failed deployment must reach `exit()` carrying an error.
    ///
    /// geth's `OnExit` reports errors for creates exactly as it does for calls; reporting
    /// `undefined` would let a tracer book a reverted deployment as successful.
    #[test]
    fn test_create_end_reports_the_error() {
        // PUSH5 <init code>, PUSH1 0, MSTORE, PUSH1 5, PUSH1 27, PUSH1 0, CREATE, STOP.
        // MSTORE right-aligns the 5-byte word, so the init code starts at offset 27. The init
        // code itself is PUSH1 0, PUSH1 0, REVERT.
        let contract = hex!("6460006000fd6000526005601b6000f000");

        let code = r#"{
            results: [],
            step: function() {},
            fault: function() {},
            enter: function() {},
            exit: function(res) { this.results.push(res.getError()) },
            result: function() { return this.results }
        }"#;
        let res = run_trace(code, Some(contract.into()), true);
        assert_eq!(res, json!(["execution reverted"]));
    }

    /// `setup` must receive the config as JSON text, as go-ethereum passes it.
    ///
    /// The documented idiom is `JSON.parse(config)`; handing over a parsed object instead
    /// makes that throw, which breaks every configurable third-party tracer.
    #[test]
    fn test_setup_receives_json_text() {
        let code = r#"{
            seen: null,
            setup: function(config) {
                this.seen = { kind: typeof config, raw: config, foo: JSON.parse(config).foo };
            },
            step: function() {},
            fault: function() {},
            result: function() { return this.seen }
        }"#;
        let insp = JsInspector::new(code.to_string(), json!({"foo": 42})).unwrap();
        let mut evm = revm::Context::mainnet()
            .with_db(CacheDB::new(EmptyDB::default()))
            .build_mainnet_with_inspector(insp);
        let res = evm.inspect_tx(TxEnv { gas_limit: 1_000_000, ..Default::default() }).unwrap();
        let (ctx, inspector) = evm.ctx_inspector();
        let res = inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref()).unwrap();

        assert_eq!(res["kind"], json!("string"));
        assert_eq!(res["raw"], json!(r#"{"foo":42}"#));
        assert_eq!(res["foo"], json!(42));
    }

    /// An absent config reaches `setup` as `"{}"`, matching geth's default for a nil config.
    #[test]
    fn test_setup_receives_empty_object_by_default() {
        let code = r#"{
            seen: null,
            setup: function(config) { this.seen = config; },
            step: function() {},
            fault: function() {},
            result: function() { return this.seen }
        }"#;
        let insp = JsInspector::new(code.to_string(), serde_json::Value::Null).unwrap();
        let mut evm = revm::Context::mainnet()
            .with_db(CacheDB::new(EmptyDB::default()))
            .build_mainnet_with_inspector(insp);
        let res = evm.inspect_tx(TxEnv { gas_limit: 1_000_000, ..Default::default() }).unwrap();
        let (ctx, inspector) = evm.ctx_inspector();
        let res = inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref()).unwrap();

        assert_eq!(res, json!("{}"));
    }

    /// `fault` must fire for errors raised while executing an instruction, not only reverts.
    ///
    /// geth routes anything raised after it emits `OnOpcode` to `OnFault`; only stack
    /// validation and gas metering, which happen earlier, go elsewhere.
    #[test]
    fn test_fault_fires_for_non_revert_execution_errors() {
        let code = r#"{
            faults: [],
            step: function() {},
            fault: function(log) { this.faults.push({ op: log.op.toString(), err: log.getError() }) },
            result: function() { return this.faults }
        }"#;

        // PUSH1 0xff, JUMP - an invalid jump destination.
        let res = run_trace(code, Some(hex!("60ff56").into()), false);
        assert_eq!(res, json!([{ "op": "JUMP", "err": "invalid jump destination" }]));

        // 0x0c is not a defined opcode.
        let res = run_trace(code, Some(hex!("0c").into()), false);
        assert_eq!(res, json!([{ "op": "opcode 0xc not defined", "err": "invalid opcode" }]));
    }

    /// Running out of gas is metered before geth emits `OnOpcode`, so it is not a fault.
    #[test]
    fn test_fault_skips_out_of_gas() {
        let code = r#"{
            faults: 0,
            step: function() {},
            fault: function() { this.faults++ },
            result: function() { return this.faults }
        }"#;
        // JUMPDEST, PUSH0, POP, PUSH1 0, JUMP - loops until the gas runs out.
        let res = run_trace_with_gas(code, Some(hex!("5b5f5060005600").into()), false, 100_000);
        assert_eq!(res, json!(0));
    }

    /// A tracer may define `fault` without `step`; geth delivers the hook either way.
    #[test]
    fn test_fault_fires_without_a_step_hook() {
        let code = r#"{
            seen: false,
            fault: function() { this.seen = true },
            result: function() { return this.seen }
        }"#;
        // PUSH1 0, PUSH1 0, REVERT
        let res = run_trace(code, Some(hex!("60006000fd").into()), false);
        assert_eq!(res, json!(true));
    }

    /// Measures what deferring the `step` hook to `step_end` would cost.
    ///
    /// Doing so requires snapshotting the stack and memory, because the hook would otherwise
    /// observe post-execution state. Run with
    /// `cargo test --release --all-features -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark, run explicitly with --ignored"]
    fn bench_step_snapshot_cost() {
        use revm::bytecode::opcode;
        use std::time::Instant;

        fn trace(code: &str, gas: u64) -> (serde_json::Value, f64) {
            let addr = Address::repeat_byte(0x01);
            let mut db = CacheDB::new(EmptyDB::default());
            db.insert_account_info(
                Address::ZERO,
                AccountInfo { balance: U256::from(1e18), ..Default::default() },
            );
            db.insert_account_info(
                addr,
                AccountInfo {
                    // JUMPDEST, PUSH1 1, PUSH1 0, MSTORE, PUSH1 0, MLOAD, POP, PUSH1 0, JUMP.
                    // Loops until the gas runs out, touching memory every iteration.
                    code: Some(Bytecode::new_legacy(hex!("5b60016000526000515060005600").into())),
                    ..Default::default()
                },
            );
            let insp = JsInspector::new(code.to_string(), serde_json::Value::Null).unwrap();
            let mut evm = revm::Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.spec = SpecId::CANCUN)
                .with_db(db)
                .build_mainnet_with_inspector(insp);

            let start = Instant::now();
            let res = evm
                .inspect_tx(TxEnv {
                    gas_limit: gas,
                    kind: TransactTo::Call(addr),
                    ..Default::default()
                })
                .unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            let (ctx, inspector) = evm.ctx_inspector();
            let out = inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref()).unwrap();
            (out, elapsed)
        }

        const GAS: u64 = 2_000_000;
        let counting = r#"{
            n: 0,
            step: function() { this.n++ },
            fault: function() {},
            result: function() { return this.n }
        }"#;
        let no_step = r#"{ fault: function() {}, result: function() { return 0 } }"#;

        // Warm up, then take the better of two runs to blunt scheduler noise.
        let (steps, _) = trace(counting, GAS);
        let steps = steps.as_u64().unwrap();
        let with_step = (0..3).map(|_| trace(counting, GAS).1).fold(f64::MAX, f64::min);
        let without_step = (0..3).map(|_| trace(no_step, GAS).1).fold(f64::MAX, f64::min);

        let per_step_js = (with_step - without_step) / steps as f64 * 1e9;
        println!("steps traced                {steps}");
        println!("per-step JS hook          {per_step_js:>9.0} ns");

        // What `step` adds per instruction, at a few stack depths and memory sizes. The stack is
        // copied whole; of the memory only the range the opcode overwrites is kept, so the cost
        // no longer tracks memory size. The full copy is measured alongside for comparison.
        for (depth, mem_len) in
            [(4usize, 32usize), (16, 1024), (64, 8192), (64, 262_144), (64, 1_048_576)]
        {
            let stack = vec![U256::from(1u64); depth];
            let memory = vec![0u8; mem_len];
            let iters = if mem_len > 100_000 { 2_000 } else { 200_000 };

            let start = Instant::now();
            for _ in 0..iters {
                core::hint::black_box((stack.clone(), memory.clone()));
            }
            let full = start.elapsed().as_secs_f64() / f64::from(iters) * 1e9;

            let start = Instant::now();
            for _ in 0..iters {
                core::hint::black_box((
                    stack.clone(),
                    MemorySnapshot::record(opcode::MSTORE, &stack, &memory),
                ));
            }
            let recorded = start.elapsed().as_secs_f64() / f64::from(iters) * 1e9;

            println!(
                "depth={depth:<3} mem={mem_len:<7} record {recorded:>6.0} ns ({:>5.1}% of hook) \
                 | full copy {full:>6.0} ns ({:.1}%)",
                recorded / per_step_js * 100.,
                full / per_step_js * 100.
            );
        }
    }

    /// `ctx.intrinsicGas` and `ctx.time` have no go-ethereum counterpart: geth dropped them in
    /// #26048 and #26291. They stay for the sake of scripts written against reth, which has
    /// exposed both since its tracer was forked. `intrinsicGas` remains the unimplemented 0.
    #[test]
    fn test_ctx_keeps_fields_geth_dropped() {
        let code = r#"{
            step: function() {},
            fault: function() {},
            result: function(ctx) {
                return {
                    gasType: typeof ctx.intrinsicGas,
                    gasIsZero: ctx.intrinsicGas === 0,
                    time: typeof ctx.time,
                };
            }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res, json!({ "gasType": "number", "gasIsZero": true, "time": "string" }));
    }

    /// `ctx.to` is the computed contract address even when the deployment failed.
    ///
    /// geth reads it from the `OnEnter` callback, which carries the address regardless of the
    /// outcome; deriving it only from a successful output leaves `null` behind.
    #[test]
    fn test_ctx_to_is_set_for_a_failed_creation() {
        let mut db = CacheDB::new(EmptyDB::default());
        db.insert_account_info(
            Address::ZERO,
            AccountInfo { balance: U256::from(1e18), ..Default::default() },
        );

        let code = r#"{
            step: function() {},
            fault: function() {},
            result: function(ctx) { return { to: toHex(ctx.to), type: ctx.type } }
        }"#;
        let insp = JsInspector::new(code.to_string(), serde_json::Value::Null).unwrap();
        let mut evm = revm::Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.spec = SpecId::CANCUN)
            .with_db(db)
            .build_mainnet_with_inspector(insp);

        let res = evm
            .inspect_tx(TxEnv {
                gas_limit: 1_000_000,
                kind: TransactTo::Create,
                // PUSH1 0, PUSH1 0, REVERT
                data: hex!("60006000fd").into(),
                ..Default::default()
            })
            .expect("pass without error");
        assert!(!res.result.is_success());

        let (ctx, inspector) = evm.ctx_inspector();
        let res = inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref()).unwrap();

        assert_eq!(res["type"], json!("CREATE"));
        assert_eq!(res["to"], json!(Address::ZERO.create(0)));
    }

    /// Every value `to_bigint` produces must carry the BigInteger.js API, not just the right
    /// digits. Asserting `toString()` alone would pass just as well on a native `BigInt`,
    /// whose `.add` does not exist, so each accessor is exercised through a chained call.
    #[test]
    fn test_to_bigint_results_are_chainable_everywhere() {
        let outer = Address::repeat_byte(0x01);
        let callee = Address::repeat_byte(0x02);

        // MSTORE 42 at offset 0, then DELEGATECALL the callee with gas 0xffff.
        let mut outer_code = hex!("602a6000526000600060006000").to_vec();
        outer_code.push(0x73); // PUSH20 <callee>
        outer_code.extend_from_slice(callee.as_slice());
        outer_code.extend_from_slice(&hex!("61fffff400")); // PUSH2 0xffff, DELEGATECALL, STOP

        let mut db = CacheDB::new(EmptyDB::default());
        db.insert_account_info(
            Address::ZERO,
            AccountInfo { balance: U256::from(1e18), ..Default::default() },
        );
        db.insert_account_info(
            outer,
            AccountInfo {
                code: Some(Bytecode::new_legacy(outer_code.into())),
                ..Default::default()
            },
        );
        db.insert_account_info(
            callee,
            AccountInfo {
                code: Some(Bytecode::new_legacy(hex!("00").into())),
                ..Default::default()
            },
        );

        let code = r#"{
            stepSeen: null,
            enterSeen: null,
            fault: function() {},
            step: function(log) {
                if (this.stepSeen !== null || log.op.toString() !== "DELEGATECALL") {
                    return;
                }
                this.stepSeen = {
                    peek: log.stack.peek(0).add(1).toString(),
                    getUint: log.memory.getUint(0).add(1).toString(),
                    contractValue: log.contract.getValue().add(1).toString(),
                };
            },
            enter: function(frame) {
                this.enterSeen = frame.getValue().add(1).toString();
            },
            exit: function() {},
            result: function(ctx, db) {
                return {
                    stack_peek: this.stepSeen.peek,
                    memory_getUint: this.stepSeen.getUint,
                    contract_getValue: this.stepSeen.contractValue,
                    frame_getValue: this.enterSeen,
                    ctx_value: ctx.value.add(1).toString(),
                    ctx_gasPrice: ctx.gasPrice.add(1).toString(),
                    db_getBalance: db.getBalance(ctx.to).add(1).toString(),
                };
            }
        }"#;

        let insp = JsInspector::new(code.to_string(), serde_json::Value::Null).unwrap();
        let mut evm = revm::Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.spec = SpecId::CANCUN)
            .with_db(db)
            .build_mainnet_with_inspector(insp);

        let res = evm
            .inspect_tx(TxEnv {
                gas_price: 7,
                gas_limit: 1_000_000,
                kind: TransactTo::Call(outer),
                value: U256::from(777u64),
                ..Default::default()
            })
            .expect("pass without error");
        assert!(res.result.is_success());

        let (ctx, inspector) = evm.ctx_inspector();
        let result = inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref()).unwrap();

        assert_eq!(
            result,
            json!({
                // DELEGATECALL's topmost argument is the gas it forwards, 0xffff.
                "stack_peek": "65536",
                "memory_getUint": "43",
                "contract_getValue": "778",
                "frame_getValue": "778",
                "ctx_value": "778",
                // Base fee is zero, so the effective tip is the full gas price.
                "ctx_gasPrice": "8",
                // `outer` starts empty and receives the 777 wei the transaction carries.
                "db_getBalance": "778",
            })
        );
    }

    /// A delegate call inherits the parent's value; a static call reports none at all.
    ///
    /// geth passes the parent's value to `OnEnter` for `DELEGATECALL` and nil for
    /// `STATICCALL`, so `frame.getValue()` is `undefined` only in the latter case.
    #[test]
    fn test_frame_value_for_delegate_and_static_calls() {
        // Pushes the six arguments both opcodes take, then the opcode itself and STOP.
        fn caller_code(op: u8, callee: Address) -> Bytes {
            let mut code = hex!("6000600060006000").to_vec(); // retLen, retOff, argLen, argOff
            code.push(0x73); // PUSH20 <callee>
            code.extend_from_slice(callee.as_slice());
            code.extend_from_slice(&hex!("61ffff")); // PUSH2 gas
            code.push(op);
            code.push(0x00); // STOP
            code.into()
        }

        fn frame_value(op: u8) -> serde_json::Value {
            let outer = Address::repeat_byte(0x01);
            let callee = Address::repeat_byte(0x02);
            let mut db = CacheDB::new(EmptyDB::default());
            db.insert_account_info(
                Address::ZERO,
                AccountInfo { balance: U256::from(1e18), ..Default::default() },
            );
            db.insert_account_info(
                outer,
                AccountInfo {
                    code: Some(Bytecode::new_legacy(caller_code(op, callee))),
                    ..Default::default()
                },
            );
            db.insert_account_info(
                callee,
                AccountInfo {
                    code: Some(Bytecode::new_legacy(hex!("00").into())),
                    ..Default::default()
                },
            );

            let code = r#"{
                seen: null,
                step: function() {},
                fault: function() {},
                enter: function(frame) {
                    var v = frame.getValue();
                    this.seen = { kind: typeof v, value: v === undefined ? null : v.toString() };
                },
                exit: function() {},
                result: function() { return this.seen }
            }"#;
            let insp = JsInspector::new(code.to_string(), serde_json::Value::Null).unwrap();
            let mut evm = revm::Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.spec = SpecId::CANCUN)
                .with_db(db)
                .build_mainnet_with_inspector(insp);

            let res = evm
                .inspect_tx(TxEnv {
                    gas_limit: 1_000_000,
                    kind: TransactTo::Call(outer),
                    value: U256::from(777u64),
                    ..Default::default()
                })
                .expect("pass without error");
            assert!(res.result.is_success());

            let (ctx, inspector) = evm.ctx_inspector();
            inspector.json_result(res, ctx.tx(), ctx.block(), ctx.db_ref()).unwrap()
        }

        // DELEGATECALL keeps the 777 wei the outer call received.
        assert_eq!(frame_value(0xf4), json!({ "kind": "object", "value": "777" }));
        // STATICCALL has no value of its own.
        assert_eq!(frame_value(0xfa), json!({ "kind": "undefined", "value": null }));
    }

    /// Frame tracing needs both `enter` and `exit`, as geth requires.
    #[test]
    fn test_enter_and_exit_must_be_paired() {
        let only_enter = r#"{
            enter: function() {}, fault: function() {}, result: function() { return null }
        }"#;
        assert!(matches!(
            JsInspector::new(only_enter.to_string(), serde_json::Value::Null),
            Err(JsInspectorError::UnpairedEnterExit)
        ));

        let only_exit = r#"{
            exit: function() {}, fault: function() {}, result: function() { return null }
        }"#;
        assert!(matches!(
            JsInspector::new(only_exit.to_string(), serde_json::Value::Null),
            Err(JsInspectorError::UnpairedEnterExit)
        ));

        let both = r#"{
            enter: function() {}, exit: function() {},
            fault: function() {}, result: function() { return null }
        }"#;
        assert!(JsInspector::new(both.to_string(), serde_json::Value::Null).is_ok());
    }

    /// `isPrecompiled` is deliberately absent during `setup`, unlike in geth.
    ///
    /// geth defines it up front and answers against an empty set, reporting every address as
    /// not precompiled - an undetectable wrong answer. Failing loudly is preferred here.
    #[test]
    fn test_is_precompiled_absent_during_setup() {
        let probe = r#"{
            seen: null,
            setup: function() { this.seen = typeof isPrecompiled },
            step: function() {},
            fault: function() {},
            result: function() { return this.seen }
        }"#;
        // Feature-detecting the global stays safe: `typeof` does not throw on an undeclared name.
        assert_eq!(run_trace(probe, None, true), json!("undefined"));

        let call = r#"{
            setup: function() { isPrecompiled("0x0000000000000000000000000000000000000001") },
            step: function() {},
            fault: function() {},
            result: function() { return null }
        }"#;
        let err = JsInspector::new(call.to_string(), serde_json::Value::Null)
            // Discarded so that a failure prints the error, not the whole `Context`.
            .map(|_| ())
            .expect_err("calling isPrecompiled in setup must fail");
        assert!(
            matches!(err, JsInspectorError::SetupCallFailed(_)),
            "expected the setup call to fail, got: {err}"
        );
    }

    /// A `result` hook that returns nothing serializes as `null`, as geth's `json.Marshal` does.
    #[test]
    fn test_result_returning_undefined_is_null() {
        let code = r#"{
            step: function() {},
            fault: function() {},
            result: function() {}
        }"#;
        assert_eq!(run_trace(code, None, true), serde_json::Value::Null);
    }

    /// A structure `JSON.stringify` rejects reports why, rather than a generic failure.
    #[test]
    fn test_unserializable_result_reports_the_reason() {
        let code = r#"{
            step: function() {},
            fault: function() {},
            result: function() { var a = {}; a.self = a; return a }
        }"#;
        let err = try_run_trace(code, None, Some(true), 1_000_000)
            .expect_err("a circular structure cannot be serialized");
        let msg = err.to_string();
        assert!(
            msg.contains("cyclic"),
            "error should name the cyclic reference rather than fail generically, got: {msg}"
        );
    }

    /// A hook failure must surface as [`JsInspectorError::JsError`], not a variant of its own.
    ///
    /// Callers map this enum onto their own error types by matching the variant: reth's
    /// `EthApiError` sends `JsError` to `InternalJsTracerError` and everything else to
    /// `InvalidParams`. A runtime failure inside the tracer is the former, not the latter.
    #[test]
    fn test_hook_failure_is_reported_as_a_js_error() {
        let code = r#"{
            step: function() { throw new Error("boom"); },
            fault: function() {},
            result: function() { return null }
        }"#;
        let err = try_run_trace(code, None, None, 1_000_000)
            .expect_err("a throwing step hook must fail the trace");

        assert!(
            matches!(err, JsInspectorError::JsError(_)),
            "hook failures must stay in the JsError variant, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("boom"), "the JS message must survive, got: {msg}");
        assert!(msg.contains("step"), "the hook must be named, got: {msg}");
    }

    #[test]
    fn test_general_counting() {
        let code = r#"{
            count: 0,
            step: function() { this.count += 1; },
            fault: function() {},
            result: function() { return this.count; }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res.as_u64().unwrap(), 3);
    }

    #[test]
    fn test_memory_access() {
        let code = r#"{
            depths: [],
            step: function(log) { this.depths.push(log.memory.slice(-1,-2)); },
            fault: function() {},
            result: function() { return this.depths; }
        }"#;
        assert_step_hook_fails(code, None);
    }

    #[test]
    fn test_stack_peek() {
        let code = r#"{
            depths: [],
            step: function(log) { this.depths.push(log.stack.peek(-1)); },
            fault: function() {},
            result: function() { return this.depths; }
        }"#;
        assert_step_hook_fails(code, None);
    }

    #[test]
    fn test_memory_get_uint() {
        let code = r#"{
            depths: [],
            step: function(log, db) { this.depths.push(log.memory.getUint(-64)); },
            fault: function() {},
            result: function() { return this.depths; }
        }"#;
        assert_step_hook_fails(code, None);
    }

    #[test]
    fn test_stack_depth() {
        let code = r#"{
            depths: [],
            step: function(log) { this.depths.push(log.stack.length()); },
            fault: function() {},
            result: function() { return this.depths; }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res, json!([0, 1, 2]));
    }

    #[test]
    fn test_memory_length() {
        let code = r#"{
            lengths: [],
            step: function(log) { this.lengths.push(log.memory.length()); },
            fault: function() {},
            result: function() { return this.lengths; }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res, json!([0, 0, 0]));
    }

    #[test]
    fn test_opcode_to_string() {
        let code = r#"{
             opcodes: [],
             step: function(log) { this.opcodes.push(log.op.toString()); },
             fault: function() {},
             result: function() { return this.opcodes; }
         }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res, json!(["PUSH1", "PUSH1", "STOP"]));
    }

    #[test]
    fn test_gas_used() {
        let code = r#"{
            depths: [],
            step: function() {},
            fault: function() {},
            result: function(ctx) { return ctx.gasPrice+'.'+ctx.gasUsed; }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res.as_str().unwrap(), "1024.21006");
    }

    #[test]
    fn test_to_word() {
        let code = r#"{
            res: null,
            step: function(log) {},
            fault: function() {},
            result: function() { return toWord('0xffaa') }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(
            res,
            json!({
                "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0,
                "9": 0, "10": 0, "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0,
                "17": 0, "18": 0, "19": 0, "20": 0, "21": 0, "22": 0, "23": 0, "24": 0,
                "25": 0, "26": 0, "27": 0, "28": 0, "29": 0, "30": 255, "31": 170,
            })
        );
    }

    #[test]
    fn test_to_address() {
        let code = r#"{
            res: null,
            step: function(log) { var address = log.contract.getAddress(); this.res = toAddress(address); },
            fault: function() {},
            result: function() { return toHex(this.res) }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res.as_str().unwrap(), "0x0101010101010101010101010101010101010101");
    }

    #[test]
    fn test_to_address_string() {
        let code = r#"{
            res: null,
            step: function(log) { var address = '0x0000000000000000000000000000000000000000'; this.res = toAddress(address); },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res.as_object().unwrap().values().map(|v| v.as_u64().unwrap()).sum::<u64>(), 0);
    }

    /// `getUint` must reject an offset no conversion can make safe, and must not wrap.
    ///
    /// `f64 as usize` saturates to `usize::MAX`, so an unchecked `offset + 32` wraps to 31 and
    /// passes any bounds test against a memory of 31 bytes or more.
    #[test]
    fn test_memory_get_uint_rejects_saturating_offset() {
        let code = r#"{
            step: function(log) { if (log.op.toString() === 'STOP') { log.memory.getUint(1e30) } },
            fault: function() {},
            result: function() { return null; }
        }"#;
        let contract = hex!("60ff60005300"); // expands memory to 32 bytes before STOP
        assert_step_hook_fails(code, Some(contract.into()));
    }

    /// Every opcode that writes memory must show the tracer the bytes as they were *before*
    /// the write. `step` records only the range about to be overwritten and `step_end` rebuilds
    /// the rest from live memory, so an opcode missing from `memory_write_range` would serve
    /// post-execution bytes instead, without any error.
    #[test]
    fn test_pre_execution_memory_for_writing_opcodes() {
        // Each program writes 0xaa into memory[0..32] first, then has the opcode under test
        // overwrite that word with zeros. The hook for that opcode must still report 0xaa.
        // RETURNDATACOPY has its own test below, it needs a preceding call to have return data.
        let programs: [(&str, &[u8]); 5] = [
            // PUSH1 0xbb, PUSH1 0, MSTORE
            ("MSTORE", &hex!("60aa60005260bb60005200")),
            // PUSH1 0xbb, PUSH1 0, MSTORE8
            ("MSTORE8", &hex!("60aa60005260bb60005300")),
            // PUSH1 32, PUSH1 0, PUSH1 0, CALLDATACOPY
            ("CALLDATACOPY", &hex!("60aa6000526020600060003700")),
            // PUSH1 32, PUSH1 0, PUSH1 0, CODECOPY
            ("CODECOPY", &hex!("60aa6000526020600060003900")),
            // PUSH1 32, PUSH1 32, PUSH1 0, MCOPY
            ("MCOPY", &hex!("60aa6000526020602060005e00")),
        ];

        for (op, program) in programs {
            let code = format!(
                r#"{{
                    seen: null,
                    fault: function() {{}},
                    step: function(log) {{
                        if (this.seen === null && log.op.toString() === "{op}"
                            && log.memory.length() >= 32) {{
                            this.seen = log.memory.getUint(0).toString();
                        }}
                    }},
                    result: function() {{ return this.seen }}
                }}"#
            );
            let res = run_trace(&code, Some(Bytes::from(program.to_vec())), true);
            assert_eq!(res, json!("170"), "{op} must report the pre-execution 0xaa");
        }
    }

    /// `RETURNDATACOPY` needs a preceding call to have any return data, so it gets its own
    /// program: a static call to the identity precompile echoes a marker back, and the copy
    /// then overwrites a different marker already in memory.
    #[test]
    fn test_pre_execution_memory_for_returndatacopy() {
        // PUSH1 0xbb, PUSH1 32, MSTORE           memory[32..64] = 0xbb
        // PUSH1 0, PUSH1 0, PUSH1 32, PUSH1 32, PUSH1 4, PUSH2 0xffff, STATICCALL, POP
        //                                        identity(memory[32..64]) -> return data 0xbb
        // PUSH1 0xaa, PUSH1 0, MSTORE            memory[0..32] = 0xaa
        // PUSH1 32, PUSH1 0, PUSH1 0, RETURNDATACOPY, STOP
        //                                        overwrites memory[0..32] with 0xbb
        let program = hex!("60bb6020526000600060206020600461fffffa5060aa6000526020600060003e00");
        let code = r#"{
            seen: null,
            fault: function() {},
            step: function(log) {
                if (this.seen === null && log.op.toString() === "RETURNDATACOPY") {
                    this.seen = log.memory.getUint(0).toString();
                }
            },
            result: function() { return this.seen }
        }"#;
        let res = run_trace(code, Some(Bytes::from(program.to_vec())), true);
        assert_eq!(res, json!("170"), "must report 0xaa, the byte before the copy");
    }

    /// `EXTCODECOPY` takes its destination from the second stack item, not the first.
    #[test]
    fn test_pre_execution_memory_for_extcodecopy() {
        // PUSH1 0xaa, PUSH1 0, MSTORE | PUSH1 32, PUSH1 0, PUSH1 0, ADDRESS, EXTCODECOPY, STOP
        let program = hex!("60aa600052602060006000303c00");
        let code = r#"{
            seen: null,
            fault: function() {},
            step: function(log) {
                if (this.seen === null && log.op.toString() === "EXTCODECOPY") {
                    this.seen = log.memory.getUint(0).toString();
                }
            },
            result: function() { return this.seen }
        }"#;
        let res = run_trace(code, Some(Bytes::from(program.to_vec())), true);
        assert_eq!(res, json!("170"));
    }

    /// `getUint` reads 32 bytes as a number, as geth's `memoryObj.GetUint` does.
    #[test]
    fn test_memory_get_uint_returns_big_integer() {
        let code = r#"{
            res: null,
            step: function(log) {
                if (log.op.toString() === 'STOP') {
                    var v = log.memory.getUint(0);
                    this.res = { hex: v.toString(16), kind: typeof v };
                }
            },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        let contract = hex!("60ff60005300"); // writes 0xff at offset 0
        let res = run_trace(code, Some(contract.into()), true);
        assert_eq!(res["kind"], json!("object"), "must be a bigInt, not a byte array");
        assert_eq!(
            res["hex"],
            json!("ff00000000000000000000000000000000000000000000000000000000000000")
        );
    }

    #[test]
    fn test_memory_slice() {
        let code = r#"{
            res: [],
            step: function(log) {
                var op = log.op.toString();
                if (op === 'MSTORE8' || op === 'STOP') {
                    this.res.push(log.memory.slice(0, 2))
                }
            },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        let contract = hex!("60ff60005300"); // PUSH1, 0xff, PUSH1, 0x00, MSTORE8, STOP
                                             // At MSTORE8 the memory is still empty, so both bytes are padding. By STOP the store
                                             // has expanded it and written 0xff at offset 0.
        let res = run_trace(code, Some(contract.into()), true);
        // A Uint8Array serializes as an object keyed by index.
        assert_eq!(res, json!([{"0": 0, "1": 0}, {"0": 255, "1": 0}]));
    }

    #[test]
    fn test_memory_limit() {
        let code = r#"{
            res: [],
            step: function(log) { if (log.op.toString() === 'STOP') { this.res.push(log.memory.slice(5, 1025 * 1024)) } },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        assert_step_hook_fails(code, None);
    }

    #[test]
    fn test_coinbase() {
        let code = r#"{
            lengths: [],
            step: function(log) { },
            fault: function() {},
            result: function(ctx) { var coinbase = ctx.coinbase; return toAddress(coinbase); }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(res.as_object().unwrap().values().map(|v| v.as_u64().unwrap()).sum::<u64>(), 0);
    }

    /// `getCost()` reports the cost of the instruction being stepped over.
    ///
    /// The default contract is PUSH1, PUSH1, STOP, which costs 3, 3 and 0. Reporting the
    /// previous instruction's cost instead would shift this to `[0, 3, 3]`.
    #[test]
    fn test_individual_opcode_costs() {
        let code = r#"{
            res: [],
            step: function(log) {
                this.res.push(log.getCost());
            },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        let res = run_trace(code, None, true);

        assert_eq!(
            res.as_array().unwrap().iter().map(|v| v.as_u64().unwrap_or(0)).collect::<Vec<u64>>(),
            vec![3, 3, 0]
        );
    }

    #[test]
    fn test_slice_builtin() {
        let code = r#"{
            res: [],
            step: function(log) {
                // A hex string is rejected: geth passes `allowString=false` to `slice`.
                var threw = false;
                try { slice('0xdeadbeef', 0, 2) } catch (e) { threw = true }
                this.res.push(threw);
                // Test slicing an array
                var arr = [0x01, 0x02, 0x03, 0x04, 0x05];
                this.res.push(toHex(slice(arr, 0, 3)));
                this.res.push(toHex(slice(arr, 1, 4)));
                // Test slicing a Uint8Array
                var uint8 = new Uint8Array([0xff, 0xee, 0xdd, 0xcc, 0xbb]);
                this.res.push(toHex(slice(uint8, 0, 2)));
                this.res.push(toHex(slice(uint8, 2, 5)));
            },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        let res = run_trace(code, Some(bytes!("0x00")), true);
        assert_eq!(res, json!([true, "0x010203", "0x020304", "0xffee", "0xddccbb"]));
    }

    /// The split between arguments that may be a hex string and those that may not.
    ///
    /// go-ethereum draws it per call site: the helpers that normalise a value into bytes take a
    /// string, the ones that consume bytes do not. Accepting a string where bytes are consumed
    /// turns a caller's mistake into a plausible answer instead of an error — `getBalance`
    /// would left-pad a short string into an address nobody asked about and report its balance.
    #[test]
    fn test_string_arguments_follow_geths_split() {
        let code = r#"{
            res: {},
            fault: function() {},
            step: function() {},
            result: function(ctx, db) {
                var self = this;
                function check(name, fn) {
                    try { fn(); self.res[name] = "accepted" }
                    catch (e) { self.res[name] = "rejected" }
                }
                // Normalising helpers take a string, as geth's allowString=true sites do.
                check("toWord",        function() { toWord("0x1234") });
                check("toAddress",     function() { toAddress("0x1234") });
                check("toContract",    function() { toContract("0x1234", 1) });
                check("isPrecompiled", function() { isPrecompiled("0x01") });
                // Consuming sites reject it, as geth's allowString=false sites do.
                check("toHex",         function() { toHex("0x1234") });
                check("slice",         function() { slice("0xaabbcc", 0, 2) });
                check("getBalance",    function() { db.getBalance("0x1234") });
                check("getNonce",      function() { db.getNonce("0x1234") });
                check("getCode",       function() { db.getCode("0x1234") });
                check("exists",        function() { db.exists("0x1234") });
                check("getState",      function() { db.getState("0x1234", "0x00") });
                return this.res;
            }
        }"#;
        let res = run_trace(code, None, true);
        assert_eq!(
            res,
            json!({
                "toWord": "accepted",
                "toAddress": "accepted",
                "toContract": "accepted",
                "isPrecompiled": "accepted",
                "toHex": "rejected",
                "slice": "rejected",
                "getBalance": "rejected",
                "getNonce": "rejected",
                "getCode": "rejected",
                "exists": "rejected",
                "getState": "rejected",
            })
        );
    }

    #[test]
    fn test_is_precompiled_builtin() {
        let code = r#"{
            res: [],
            step: function(log) {
                this.res.push(isPrecompiled("0x01"));
                this.res.push(isPrecompiled("0x0000000000000000000000000000000000000002"));
                this.res.push(isPrecompiled("0x0000000000000000000000000000000000000000"));
            },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        let res = run_trace(code, Some(bytes!("0x00")), true);
        assert_eq!(res, json!([true, true, false]));
    }

    #[test]
    fn test_has_own_property() {
        let code = r#"{
            res: [],
            step: function(log) {
                this.res.push(log.hasOwnProperty("stack"));
            },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        let res = run_trace(code, Some(bytes!("0x00")), true);
        assert_eq!(res, json!([true]));
    }

    #[test]
    fn test_slice_with_stack_values() {
        let code = r#"{
            res: [],
            step: function(log) {
                if ((log.stack.length() > 0) && log.memory.length() >= log.stack.peek(0)) {
                    this.res.push(log.memory.slice(0, log.stack.peek(0)));
                }
            },
            fault: function() {},
            result: function() { return this.res }
        }"#;
        let res = run_trace(code, Some(bytes!("0x5F5F52600100")), true);
        assert_eq!(res, json!([json!({}), json!({}), json!({"0": 0})]));
    }
}
