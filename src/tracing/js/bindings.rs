//! Type bindings for js tracing inspector

use crate::tracing::{
    js::builtins::{
        address_to_uint8_array, address_to_uint8_array_value, bytes_from_value, bytes_to_address,
        bytes_to_b256, to_bigint, to_uint8_array, to_uint8_array_value,
    },
    TransactionContext,
};
use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use alloy_primitives::{Address, Bytes, B256, U256};
use boa_engine::{
    js_string,
    native_function::NativeFunction,
    object::{builtins::JsUint8Array, FunctionObjectBuilder},
    property::PropertyDescriptor,
    Context, JsArgs, JsError, JsNativeError, JsObject, JsResult, JsString, JsValue,
};
use boa_gc::{empty_trace, Finalize, Gc, GcRefCell, Trace};
use core::cell::RefCell;
use revm::{
    bytecode::opcode::{self, OpCode, PUSH0, PUSH32},
    context_interface::DBErrorMarker,
    primitives::KECCAK_EMPTY,
    state::{AccountInfo, Bytecode, EvmState},
    Database, DatabaseRef,
};

/// Shared mutable state captured by JS native functions.
#[derive(Clone, Debug)]
struct Shared<T>(Rc<RefCell<T>>);

impl<T> Shared<T> {
    fn new(value: T) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }

    fn replace(&self, value: T) {
        *self.0.borrow_mut() = value;
    }
}

impl<T> Finalize for Shared<T> {}

unsafe impl<T> Trace for Shared<T> {
    empty_trace!();
}

/// Reusable JS log object for opcode steps.
#[derive(Debug)]
pub(crate) struct ReusableStepLog {
    state: Shared<StepLogState>,
    object: JsObject,
}

impl ReusableStepLog {
    pub(crate) fn new(ctx: &mut Context) -> JsResult<Self> {
        let state = Shared::new(StepLogState::default());
        let object = JsObject::with_object_proto(ctx.intrinsics());

        object.set(js_string!("op"), build_step_op_object(state.clone(), ctx)?, false, ctx)?;
        object.set(
            js_string!("memory"),
            build_step_memory_object(state.clone(), ctx)?,
            false,
            ctx,
        )?;
        object.set(
            js_string!("stack"),
            build_step_stack_object(state.clone(), ctx)?,
            false,
            ctx,
        )?;
        object.set(
            js_string!("contract"),
            build_step_contract_object(state.clone(), ctx)?,
            false,
            ctx,
        )?;

        let get_pc = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| Ok(JsValue::from(state.0.borrow().pc)),
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_gas = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| Ok(JsValue::from(state.0.borrow().gas_remaining)),
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_cost = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| Ok(JsValue::from(state.0.borrow().cost)),
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_depth = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| Ok(JsValue::from(state.0.borrow().depth)),
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_refund = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| Ok(JsValue::from(state.0.borrow().refund)),
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_error = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| {
                    Ok(state
                        .0
                        .borrow()
                        .error
                        .as_ref()
                        .map(|error| JsValue::from(js_string!(error.as_str())))
                        .unwrap_or_else(JsValue::undefined))
                },
                state.clone(),
            ),
        )
        .length(0)
        .build();

        object.set(js_string!("getPC"), get_pc, false, ctx)?;
        object.set(js_string!("getError"), get_error, false, ctx)?;
        object.set(js_string!("getGas"), get_gas, false, ctx)?;
        object.set(js_string!("getCost"), get_cost, false, ctx)?;
        object.set(js_string!("getDepth"), get_depth, false, ctx)?;
        object.set(js_string!("getRefund"), get_refund, false, ctx)?;

        Ok(Self { state, object })
    }

    pub(crate) fn update(&self, step: StepLog) {
        self.state.replace(step.into());
    }

    pub(crate) fn value(&self) -> JsValue {
        self.object.clone().into()
    }
}

/// Reusable JS call frame object for enter callbacks.
#[derive(Debug)]
pub(crate) struct ReusableCallFrame {
    state: Shared<CallFrameState>,
    object: JsObject,
}

impl ReusableCallFrame {
    pub(crate) fn new(ctx: &mut Context) -> JsResult<Self> {
        let state = Shared::new(CallFrameState::default());
        let object = JsObject::with_object_proto(ctx.intrinsics());

        let get_from = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, ctx| {
                    address_to_uint8_array_value(state.0.borrow().caller, ctx)
                },
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_to = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, ctx| {
                    address_to_uint8_array_value(state.0.borrow().contract, ctx)
                },
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_value = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, ctx| to_bigint(state.0.borrow().value, ctx),
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_input = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, ctx| {
                    to_uint8_array_value(state.0.borrow().input.clone(), ctx)
                },
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_gas = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| Ok(JsValue::from(state.0.borrow().gas)),
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_type = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| {
                    Ok(JsValue::from(js_string!(state.0.borrow().kind)))
                },
                state.clone(),
            ),
        )
        .length(0)
        .build();

        object.set(js_string!("getFrom"), get_from, false, ctx)?;
        object.set(js_string!("getTo"), get_to, false, ctx)?;
        object.set(js_string!("getValue"), get_value, false, ctx)?;
        object.set(js_string!("getInput"), get_input, false, ctx)?;
        object.set(js_string!("getGas"), get_gas, false, ctx)?;
        object.set(js_string!("getType"), get_type, false, ctx)?;

        Ok(Self { state, object })
    }

    pub(crate) fn update(&self, frame: CallFrame) {
        self.state.replace(frame.into());
    }

    pub(crate) fn value(&self) -> JsValue {
        self.object.clone().into()
    }
}

/// Reusable JS frame result object for exit callbacks.
#[derive(Debug)]
pub(crate) struct ReusableFrameResult {
    state: Shared<FrameResultState>,
    object: JsObject,
}

impl ReusableFrameResult {
    pub(crate) fn new(ctx: &mut Context) -> JsResult<Self> {
        let state = Shared::new(FrameResultState::default());
        let object = JsObject::with_object_proto(ctx.intrinsics());

        let get_gas_used = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| Ok(JsValue::from(state.0.borrow().gas_used)),
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_output = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, ctx| {
                    to_uint8_array_value(state.0.borrow().output.clone(), ctx)
                },
                state.clone(),
            ),
        )
        .length(0)
        .build();
        let get_error = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, _args, state, _ctx| {
                    Ok(state
                        .0
                        .borrow()
                        .error
                        .as_ref()
                        .map(|error| JsValue::from(js_string!(error.as_str())))
                        .unwrap_or_else(JsValue::undefined))
                },
                state.clone(),
            ),
        )
        .length(0)
        .build();

        object.set(js_string!("getGasUsed"), get_gas_used, false, ctx)?;
        object.set(js_string!("getOutput"), get_output, false, ctx)?;
        object.set(js_string!("getError"), get_error, false, ctx)?;

        Ok(Self { state, object })
    }

    pub(crate) fn update(&self, frame: FrameResult) {
        self.state.replace(frame.into());
    }

    pub(crate) fn value(&self) -> JsValue {
        self.object.clone().into()
    }
}

/// Reusable JS database object for step and fault callbacks.
#[derive(Debug)]
pub(crate) struct ReusableEvmDb {
    state: Shared<EvmDbState>,
    object: JsObject,
}

impl ReusableEvmDb {
    pub(crate) fn new(ctx: &mut Context) -> JsResult<Self> {
        let state = Shared::new(EvmDbState::default());
        let object = JsObject::with_object_proto(ctx.intrinsics());

        let exists = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, state, ctx| {
                    let db = current_db(state)?;
                    let acc = db.read_basic(args.get_or_undefined(0).clone(), ctx)?;
                    Ok(JsValue::from(acc.is_some()))
                },
                state.clone(),
            ),
        )
        .length(1)
        .build();
        let get_balance = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, state, ctx| {
                    let db = current_db(state)?;
                    let balance = db
                        .read_basic(args.get_or_undefined(0).clone(), ctx)?
                        .map(|acc| acc.balance)
                        .unwrap_or_default();
                    to_bigint(balance, ctx)
                },
                state.clone(),
            ),
        )
        .length(1)
        .build();
        let get_nonce = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, state, ctx| {
                    let db = current_db(state)?;
                    let nonce = db
                        .read_basic(args.get_or_undefined(0).clone(), ctx)?
                        .map(|acc| acc.nonce)
                        .unwrap_or_default();
                    Ok(JsValue::from(nonce))
                },
                state.clone(),
            ),
        )
        .length(1)
        .build();
        let get_code = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, state, ctx| {
                    let db = current_db(state)?;
                    Ok(db.read_code(args.get_or_undefined(0).clone(), ctx)?.into())
                },
                state.clone(),
            ),
        )
        .length(1)
        .build();
        let get_state = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, state, ctx| {
                    let db = current_db(state)?;
                    Ok(db
                        .read_state(
                            args.get_or_undefined(0).clone(),
                            args.get_or_undefined(1).clone(),
                            ctx,
                        )?
                        .into())
                },
                state.clone(),
            ),
        )
        .length(2)
        .build();

        object.set(js_string!("getBalance"), get_balance, false, ctx)?;
        object.set(js_string!("getNonce"), get_nonce, false, ctx)?;
        object.set(js_string!("getCode"), get_code, false, ctx)?;
        object.set(js_string!("getState"), get_state, false, ctx)?;
        object.set(js_string!("exists"), exists, false, ctx)?;

        Ok(Self { state, object })
    }

    pub(crate) fn update(&self, db: EvmDbRef) {
        self.state.replace(EvmDbState { current: Some(db) });
    }

    pub(crate) fn value(&self) -> JsValue {
        self.object.clone().into()
    }
}

#[derive(Clone, Debug, Default)]
struct EvmDbState {
    current: Option<EvmDbRef>,
}

fn current_db(state: &Shared<EvmDbState>) -> JsResult<EvmDbRef> {
    state.0.borrow().current.clone().ok_or_else(|| {
        JsError::from_native(
            JsNativeError::typ().with_message("tracer accessed db before it was initialized"),
        )
    })
}

#[derive(Clone, Debug, Default)]
struct StepLogState {
    stack: Option<StackRef>,
    op: u8,
    memory: Option<MemoryRef>,
    pc: u64,
    gas_remaining: u64,
    cost: u64,
    depth: u64,
    refund: u64,
    error: Option<String>,
    contract: Contract,
}

impl From<StepLog> for StepLogState {
    fn from(step: StepLog) -> Self {
        Self {
            stack: Some(step.stack),
            op: step.op.0,
            memory: Some(step.memory),
            pc: step.pc,
            gas_remaining: step.gas_remaining,
            cost: step.cost,
            depth: step.depth,
            refund: step.refund,
            error: step.error,
            contract: step.contract,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct CallFrameState {
    caller: Address,
    contract: Address,
    value: U256,
    input: Bytes,
    gas: u64,
    kind: &'static str,
}

impl From<CallFrame> for CallFrameState {
    fn from(frame: CallFrame) -> Self {
        Self {
            caller: frame.contract.caller,
            contract: frame.contract.contract,
            value: frame.contract.value,
            input: frame.contract.input,
            gas: frame.gas,
            kind: frame.kind,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct FrameResultState {
    gas_used: u64,
    output: Bytes,
    error: Option<String>,
}

impl From<FrameResult> for FrameResultState {
    fn from(frame: FrameResult) -> Self {
        Self { gas_used: frame.gas_used, output: frame.output, error: frame.error }
    }
}

fn build_step_op_object(state: Shared<StepLogState>, context: &mut Context) -> JsResult<JsObject> {
    let obj = JsObject::with_object_proto(context.intrinsics());
    let to_number = FunctionObjectBuilder::new(
        context.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, _ctx| Ok(JsValue::from(state.0.borrow().op)),
            state.clone(),
        ),
    )
    .length(0)
    .build();
    let is_push = FunctionObjectBuilder::new(
        context.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, _ctx| {
                Ok(JsValue::from((PUSH0..=PUSH32).contains(&state.0.borrow().op)))
            },
            state.clone(),
        ),
    )
    .length(0)
    .build();
    let to_string = FunctionObjectBuilder::new(
        context.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, _ctx| {
                let value = state.0.borrow().op;
                if let Some(op) = OpCode::new(value) {
                    Ok(JsValue::from(js_string!(op.as_str())))
                } else {
                    // <https://github.com/ethereum/go-ethereum/blob/7c107c2691fa66a1da60e2b95f5946c3a3921b00/core/vm/opcodes.go#L461-L461>
                    Ok(JsValue::from(js_string!(format!("opcode {:#x} not defined", value))))
                }
            },
            state.clone(),
        ),
    )
    .length(0)
    .build();

    obj.set(js_string!("toNumber"), to_number, false, context)?;
    obj.set(js_string!("toString"), to_string, false, context)?;
    obj.set(js_string!("isPush"), is_push, false, context)?;
    Ok(obj)
}

fn build_step_stack_object(
    state: Shared<StepLogState>,
    context: &mut Context,
) -> JsResult<JsObject> {
    let obj = JsObject::with_object_proto(context.intrinsics());
    let length = FunctionObjectBuilder::new(
        context.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, _ctx| {
                let len = state
                    .0
                    .borrow()
                    .stack
                    .as_ref()
                    .and_then(|stack| stack.0.with_inner(Vec::len))
                    .unwrap_or_default();
                Ok(JsValue::from(len))
            },
            state.clone(),
        ),
    )
    .length(0)
    .build();
    let peek = FunctionObjectBuilder::new(
        context.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, args, state, ctx| {
                let stack = state.0.borrow().stack.clone().ok_or_else(|| {
                    JsError::from_native(
                        JsNativeError::typ()
                            .with_message("tracer accessed stack before it was initialized"),
                    )
                })?;
                let len = stack.0.with_inner(Vec::len).unwrap_or_default();
                let idx = StackRef::parse_index(args.get_or_undefined(0), len, ctx)?;
                if len <= idx {
                    return Err(JsError::from_native(JsNativeError::typ().with_message(format!(
                        "tracer accessed out of bound stack: size {len}, index {idx}"
                    ))));
                }
                stack.peek(idx, ctx)
            },
            state.clone(),
        ),
    )
    .length(1)
    .build();

    obj.set(js_string!("length"), length, false, context)?;
    obj.set(js_string!("peek"), peek, false, context)?;
    Ok(obj)
}

fn build_step_memory_object(
    state: Shared<StepLogState>,
    context: &mut Context,
) -> JsResult<JsObject> {
    let obj = JsObject::with_object_proto(context.intrinsics());
    let length = FunctionObjectBuilder::new(
        context.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, _ctx| {
                let len = state.0.borrow().memory.as_ref().map(MemoryRef::len).unwrap_or_default();
                Ok(JsValue::from(len as u64))
            },
            state.clone(),
        ),
    )
    .length(0)
    .build();
    let slice = FunctionObjectBuilder::new(
        context.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, args, state, ctx| {
                let memory = state.0.borrow().memory.clone().ok_or_else(|| {
                    JsError::from_native(
                        JsNativeError::typ()
                            .with_message("tracer accessed memory before it was initialized"),
                    )
                })?;
                let (start, end) = (args.get_or_undefined(0), args.get_or_undefined(1));
                // geth answers an empty range before rejecting negatives, so `slice(-1, -1)`
                // yields an empty array rather than an error.
                if let (Some(start), Some(end)) = (start.as_number(), end.as_number()) {
                    if start == end {
                        return to_uint8_array_value(Vec::new(), ctx);
                    }
                }
                let start = MemoryRef::parse_index(start, "start", ctx)?;
                let end = MemoryRef::parse_index(end, "end", ctx)?;
                if end < start {
                    return Err(JsError::from_native(JsNativeError::typ().with_message(format!(
                        "tracer accessed out of bound memory: offset {start}, end {end}"
                    ))));
                }
                // Past the end reads as zeros, as geth pads it: `step` runs before the
                // instruction grows memory, so reading what it is about to write is routine.
                let slice = memory
                    .0
                    .with_inner(|view| view.copy_padded(start, end - start))
                    .unwrap_or_else(|| Ok(Vec::new()))
                    .map_err(|msg| JsError::from_native(JsNativeError::typ().with_message(msg)))?;
                to_uint8_array_value(slice, ctx)
            },
            state.clone(),
        ),
    )
    .length(2)
    .build();
    let get_uint = FunctionObjectBuilder::new(
        context.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, args, state, ctx| {
                let memory = state.0.borrow().memory.clone().ok_or_else(|| {
                    JsError::from_native(
                        JsNativeError::typ()
                            .with_message("tracer accessed memory before it was initialized"),
                    )
                })?;
                let len = memory.len();
                let offset = MemoryRef::parse_index(args.get_or_undefined(0), "offset", ctx)?;
                if offset.checked_add(32).is_none_or(|end| end > len) {
                    return Err(MemoryRef::out_of_bounds_error(len, offset, 32));
                }
                let slice: Vec<u8> = memory
                    .0
                    .with_inner(|view| {
                        (offset..offset + 32).map(|i| view.byte(i).unwrap_or(0)).collect()
                    })
                    .unwrap_or_default();
                // geth returns a big integer here, not a byte array.
                to_bigint(U256::from_be_slice(&slice), ctx)
            },
            state.clone(),
        ),
    )
    .length(1)
    .build();

    obj.set(js_string!("slice"), slice, false, context)?;
    obj.set(js_string!("getUint"), get_uint, false, context)?;
    obj.set(js_string!("length"), length, false, context)?;
    Ok(obj)
}

fn build_step_contract_object(
    state: Shared<StepLogState>,
    ctx: &mut Context,
) -> JsResult<JsObject> {
    let obj = JsObject::with_object_proto(ctx.intrinsics());
    let get_caller = FunctionObjectBuilder::new(
        ctx.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, ctx| {
                address_to_uint8_array_value(state.0.borrow().contract.caller, ctx)
            },
            state.clone(),
        ),
    )
    .length(0)
    .build();
    let get_address = FunctionObjectBuilder::new(
        ctx.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, ctx| {
                address_to_uint8_array_value(state.0.borrow().contract.contract, ctx)
            },
            state.clone(),
        ),
    )
    .length(0)
    .build();
    let get_value = FunctionObjectBuilder::new(
        ctx.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, ctx| to_bigint(state.0.borrow().contract.value, ctx),
            state.clone(),
        ),
    )
    .length(0)
    .build();
    let get_input = FunctionObjectBuilder::new(
        ctx.realm(),
        NativeFunction::from_copy_closure_with_captures(
            move |_this, _args, state, ctx| {
                to_uint8_array_value(state.0.borrow().contract.input.clone(), ctx)
            },
            state.clone(),
        ),
    )
    .length(0)
    .build();

    obj.set(js_string!("getCaller"), get_caller, false, ctx)?;
    obj.set(js_string!("getAddress"), get_address, false, ctx)?;
    obj.set(js_string!("getValue"), get_value, false, ctx)?;
    obj.set(js_string!("getInput"), get_input, false, ctx)?;
    Ok(obj)
}

/// A wrapper for a value that can be garbage collected, but will not give access to the value if
/// it has been dropped via its guard.
///
/// This is used to allow the JS tracer functions to access values at a certain point during
/// inspection by ref without having to clone them and capture them in the js object.
///
/// JS tracer functions get access to evm internals via objects or function arguments, for example
/// `function step(log,evm)` where log has an object `stack` that has a function `peek(number)` that
/// returns a value from the stack.
///
/// These functions could get garbage collected, however the data accessed by the function is
/// supposed to be ephemeral and only valid for the duration of the function call.
///
/// This type supports garbage collection of (rust) references and prevents access to the value if
/// it has been dropped.
#[derive(Clone, Debug)]
struct GuardedNullableGc<Val: 'static> {
    /// The lifetime is a lie to make it possible to use a reference in boa which requires 'static
    inner: Rc<RefCell<Option<Guarded<'static, Val>>>>,
}

impl<Val: 'static> GuardedNullableGc<Val> {
    /// Creates a garbage collectible value to the given reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the guard is dropped before the value is dropped.
    fn new_ref(val: &Val) -> (Self, GcGuard<'_, Val>) {
        Self::new(Guarded::Ref(val))
    }

    /// Creates a garbage collectible value to the given reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the guard is dropped before the value is dropped.
    fn new_owned<'a>(val: Val) -> (Self, GcGuard<'a, Val>) {
        Self::new(Guarded::Owned(val))
    }

    fn new(val: Guarded<'_, Val>) -> (Self, GcGuard<'_, Val>) {
        let inner = Rc::new(RefCell::new(Some(val)));
        let guard = GcGuard { inner: Rc::clone(&inner) };

        // SAFETY: guard enforces that the value is removed from the refcell before it is dropped.
        #[allow(clippy::missing_transmute_annotations)]
        let this = Self { inner: unsafe { core::mem::transmute(inner) } };

        (this, guard)
    }

    /// Executes the given closure with a reference to the inner value if it is still present.
    fn with_inner<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Val) -> R,
    {
        self.inner.borrow().as_ref().map(|guard| f(guard.as_ref()))
    }
}

impl<Val: 'static> Finalize for GuardedNullableGc<Val> {}

unsafe impl<Val: 'static> Trace for GuardedNullableGc<Val> {
    empty_trace!();
}

/// A value that is either a reference or an owned value.
#[derive(Debug)]
enum Guarded<'a, T> {
    Ref(&'a T),
    Owned(T),
}

impl<T> Guarded<'_, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        match self {
            Guarded::Ref(val) => val,
            Guarded::Owned(val) => val,
        }
    }
}

/// Guard the inner value, once this value is dropped the inner value is also removed.
///
/// This type guarantees that it never outlives the wrapped value.
#[derive(Debug)]
#[must_use]
pub(crate) struct GcGuard<'a, Val> {
    inner: Rc<RefCell<Option<Guarded<'a, Val>>>>,
}

impl<Val> Drop for GcGuard<'_, Val> {
    fn drop(&mut self) {
        self.inner.borrow_mut().take();
    }
}

/// The Log object that is passed to the javascript inspector.
#[derive(Debug)]
pub(crate) struct StepLog {
    /// Stack before step execution
    pub(crate) stack: StackRef,
    /// Opcode to be executed
    pub(crate) op: OpObj,
    /// All allocated memory in a step
    pub(crate) memory: MemoryRef,
    /// Program counter before step execution
    pub(crate) pc: u64,
    /// Remaining gas before step execution
    pub(crate) gas_remaining: u64,
    /// Gas cost of step execution
    pub(crate) cost: u64,
    /// Call depth
    pub(crate) depth: u64,
    /// Gas refund counter before step execution
    pub(crate) refund: u64,
    /// returns information about the error if one occurred, otherwise returns undefined
    pub(crate) error: Option<String>,
    /// The contract object available to the js inspector
    pub(crate) contract: Contract,
}

/// Upper bound on the zero padding `memory.slice` will synthesize, matching geth's
/// `memoryPadLimit`.
const MEMORY_PAD_LIMIT: usize = 1024 * 1024;

/// The range of memory an instruction overwrites, as `(offset, size)`.
///
/// Only these opcodes write to memory. Calls are absent because their return data lands after
/// the child frame returns, which is past `step_end`; `CREATE`, `RETURN`, `KECCAK256` and the
/// `LOG` family only read.
fn memory_write_range(op: u8, stack: &[U256]) -> Option<(usize, usize)> {
    let peek = |n: usize| stack.len().checked_sub(n + 1).map(|i| stack[i]);
    let (offset, size) = match op {
        opcode::MSTORE => (peek(0)?, U256::from(32)),
        opcode::MSTORE8 => (peek(0)?, U256::from(1)),
        opcode::MCOPY | opcode::CALLDATACOPY | opcode::CODECOPY | opcode::RETURNDATACOPY => {
            (peek(0)?, peek(2)?)
        }
        opcode::EXTCODECOPY => (peek(1)?, peek(3)?),
        _ => return None,
    };
    Some((usize::try_from(offset).ok()?, usize::try_from(size).ok()?))
}

/// What `step` records so that `step_end` can serve the pre-execution memory.
///
/// Copying the whole memory costs time proportional to its size, which reaches milliseconds per
/// instruction once a contract has grown it to megabytes. Only the bytes the instruction is
/// about to overwrite are kept; the rest is read back from the post-execution memory.
#[derive(Clone, Debug, Default)]
pub(crate) struct MemorySnapshot {
    /// Length before the instruction ran.
    len: usize,
    /// Where `patch` starts.
    patch_offset: usize,
    /// The pre-execution bytes of the range the instruction overwrites.
    patch: Vec<u8>,
}

impl MemorySnapshot {
    /// A snapshot of memory that the instruction does not modify, of the given length.
    pub(crate) fn unchanged(len: usize) -> Self {
        Self { len, ..Default::default() }
    }

    /// Records the bytes `op` is about to overwrite, given the pre-execution stack and memory.
    pub(crate) fn record(op: u8, stack: &[U256], memory: &[u8]) -> Self {
        let mut this = Self { len: memory.len(), ..Default::default() };
        if let Some((offset, size)) = memory_write_range(op, stack) {
            if offset < memory.len() && size > 0 {
                let end = offset.saturating_add(size).min(memory.len());
                this.patch_offset = offset;
                this.patch.extend_from_slice(&memory[offset..end]);
            }
        }
        this
    }
}

/// The pre-execution memory, reconstructed from a [`MemorySnapshot`] and the memory as it
/// stands after the instruction ran.
#[derive(Clone, Debug)]
pub(crate) struct MemoryView {
    snapshot: MemorySnapshot,
    /// Memory after the instruction ran. Kept alive by the [`GcGuard`] handed out with this view.
    post: &'static [u8],
}

impl MemoryView {
    /// The byte at `i` as it was before the instruction ran, or `None` past the end.
    fn byte(&self, i: usize) -> Option<u8> {
        if i >= self.snapshot.len {
            return None;
        }
        let patch_len = self.snapshot.patch.len();
        if i >= self.snapshot.patch_offset && i < self.snapshot.patch_offset + patch_len {
            return Some(self.snapshot.patch[i - self.snapshot.patch_offset]);
        }
        self.post.get(i).copied()
    }

    /// Copies `size` bytes from `offset`, zero-filling whatever lies past the end.
    fn copy_padded(&self, offset: usize, size: usize) -> Result<Vec<u8>, String> {
        let len = self.snapshot.len;
        let padding_needed = offset.saturating_add(size).saturating_sub(len);
        if padding_needed > MEMORY_PAD_LIMIT {
            return Err(format!("reached limit for padding memory slice: {padding_needed}"));
        }
        Ok((offset..offset.saturating_add(size)).map(|i| self.byte(i).unwrap_or(0)).collect())
    }
}

/// Represents the memory object
#[derive(Clone, Debug)]
pub(crate) struct MemoryRef(GuardedNullableGc<MemoryView>);

impl MemoryRef {
    /// Pairs the snapshot taken in `step` with the memory as it stands in `step_end`.
    ///
    /// The returned guard revokes JavaScript's access to `post`, so it must not outlive it.
    pub(crate) fn new(snapshot: MemorySnapshot, post: &[u8]) -> (Self, GcGuard<'_, MemoryView>) {
        // SAFETY: the guard is tied to `post`'s lifetime and clears the view when dropped, so
        // JavaScript can only reach the reference while it is valid.
        let post: &'static [u8] = unsafe { core::mem::transmute::<&[u8], &'static [u8]>(post) };
        let (inner, guard) = GuardedNullableGc::new_owned(MemoryView { snapshot, post });
        (Self(inner), guard)
    }

    fn len(&self) -> usize {
        self.0.with_inner(|view| view.snapshot.len).unwrap_or_default()
    }

    fn invalid_index_error(name: &str, index: impl core::fmt::Display) -> JsError {
        JsError::from_native(
            JsNativeError::typ().with_message(format!("invalid memory {name}: {index}")),
        )
    }

    /// Converts a memory index argument, rejecting what is not a non-negative integer index.
    /// Bounds are left to the caller: `slice` pads past the end, `getUint` does not.
    fn parse_index(value: &JsValue, name: &str, ctx: &mut Context) -> JsResult<usize> {
        if value.is_undefined() {
            return Err(Self::invalid_index_error(name, "undefined"));
        }
        if let Some(index) = value.as_number() {
            if !index.is_finite() || index < 0. {
                return Err(Self::invalid_index_error(name, index));
            }
        }
        // Boa's `ToIndex` rejects BigInt, which a script may pass as `10n`.
        if let Some(index) = value.as_bigint() {
            let index = index.to_string();
            return index.parse::<usize>().map_err(|_| Self::invalid_index_error(name, &index));
        }

        let index = value.to_index(ctx)?;
        usize::try_from(index).map_err(|_| Self::invalid_index_error(name, index))
    }

    fn out_of_bounds_error(len: usize, offset: usize, size: usize) -> JsError {
        JsError::from_native(JsNativeError::typ().with_message(format!(
            "tracer accessed out of bound memory: available {len}, offset {offset}, size {size}"
        )))
    }
}

impl Finalize for MemoryRef {}

unsafe impl Trace for MemoryRef {
    empty_trace!();
}

/// Represents the state object
#[derive(Clone, Debug)]
pub(crate) struct StateRef {
    state: GuardedNullableGc<EvmState>,
    /// Whether the transaction is over, so a self-destructed account is gone, as geth's
    /// `Finalise` deletes it. Until then it keeps its code, nonce and storage.
    finalised: bool,
}

impl StateRef {
    /// Creates a new state reference
    pub(crate) fn new(state: &EvmState, finalised: bool) -> (Self, GcGuard<'_, EvmState>) {
        let (inner, guard) = GuardedNullableGc::new_ref(state);
        (Self { state: inner, finalised }, guard)
    }

    /// The account as the transaction has left it so far: `Some(None)` once the transaction is
    /// over and it self-destructed, so the account no longer exists, or `None` if the
    /// transaction has not touched it and the database holds the answer.
    fn get_account(&self, address: &Address) -> Option<Option<AccountInfo>> {
        self.state.with_inner(|state| {
            let acc = state.get(address)?;
            Some((!(self.finalised && acc.is_selfdestructed())).then(|| acc.info.clone()))
        })?
    }

    /// The slot's value as the transaction has left it so far, or `None` if the transaction has
    /// not touched it and the database holds the answer. A contract created by this transaction
    /// starts with empty storage, so its untouched slots are zero rather than whatever the
    /// database has for that address.
    fn get_storage(&self, address: &Address, slot: U256) -> Option<U256> {
        self.state.with_inner(|state| {
            let acc = state.get(address)?;
            if self.finalised && acc.is_selfdestructed() {
                return Some(U256::ZERO);
            }
            acc.storage
                .get(&slot)
                .map(|slot| slot.present_value())
                .or_else(|| acc.is_created().then_some(U256::ZERO))
        })?
    }
}

impl Finalize for StateRef {}

unsafe impl Trace for StateRef {
    empty_trace!();
}

/// Represents the database
#[derive(Clone, Debug)]
pub(crate) struct GcDb<DB: 'static>(GuardedNullableGc<DB>);

impl<DB> GcDb<DB>
where
    DB: DatabaseRef + 'static,
{
    /// Creates a new stack reference
    fn new<'a>(db: DB) -> (Self, GcGuard<'a, DB>) {
        let (inner, guard) = GuardedNullableGc::new_owned(db);
        (Self(inner), guard)
    }
}

impl<DB: 'static> Finalize for GcDb<DB> {}

unsafe impl<DB: 'static> Trace for GcDb<DB> {
    empty_trace!();
}

/// Represents the opcode object
#[derive(Debug)]
pub(crate) struct OpObj(pub(crate) u8);

impl From<u8> for OpObj {
    fn from(op: u8) -> Self {
        Self(op)
    }
}

/// Represents the stack object
#[derive(Clone, Debug)]
pub(crate) struct StackRef(GuardedNullableGc<Vec<U256>>);

impl StackRef {
    /// Takes ownership of a snapshot of the stack, bottom-first.
    ///
    /// Owned for the same reason as [`MemoryRef::new`]: the hook runs after the instruction.
    pub(crate) fn new<'a>(stack: Vec<U256>) -> (Self, GcGuard<'a, Vec<U256>>) {
        let (inner, guard) = GuardedNullableGc::new_owned(stack);
        (Self(inner), guard)
    }

    fn parse_index(value: &JsValue, len: usize, ctx: &mut Context) -> JsResult<usize> {
        let index = value.to_numeric_number(ctx)?;
        if !index.is_finite() || index < 0. || index > usize::MAX as f64 {
            return Err(JsError::from_native(JsNativeError::typ().with_message(format!(
                "tracer accessed out of bound stack: size {len}, index {index}"
            ))));
        }
        Ok(index as usize)
    }

    fn peek(&self, idx: usize, ctx: &mut Context) -> JsResult<JsValue> {
        self.0
            .with_inner(|stack| {
                // `idx` counts from the top, as `Stack::peek` does.
                stack.len().checked_sub(idx + 1).map(|pos| stack[pos]).ok_or_else(|| {
                    JsError::from_native(JsNativeError::typ().with_message(format!(
                        "tracer accessed out of bound stack: size {}, index {}",
                        stack.len(),
                        idx
                    )))
                })
            })
            .ok_or_else(|| {
                JsError::from_native(
                    JsNativeError::typ()
                        .with_message("tracer accessed stack after it was dropped".to_string()),
                )
            })?
            .and_then(|value| to_bigint(value, ctx))
    }
}

impl Finalize for StackRef {}

unsafe impl Trace for StackRef {
    empty_trace!();
}

/// Represents the contract object
#[derive(Clone, Debug, Default)]
pub(crate) struct Contract {
    pub(crate) caller: Address,
    pub(crate) contract: Address,
    pub(crate) value: U256,
    pub(crate) input: Bytes,
}

/// Represents the call frame object for exit functions
pub(crate) struct FrameResult {
    pub(crate) gas_used: u64,
    pub(crate) output: Bytes,
    pub(crate) error: Option<String>,
}

/// Represents the call frame object for enter functions
pub(crate) struct CallFrame {
    pub(crate) contract: Contract,
    /// The value `getType()` reports. A plain string rather than a `CallKind` because
    /// selfdestruct is reported as its own frame type, which `CallKind` does not model.
    pub(crate) kind: &'static str,
    pub(crate) gas: u64,
}

/// Defines `name` on `obj` as a big integer materialised on first read, since converting
/// eagerly would evaluate the whole `bigInt` library per `result()` call. The accessor stores
/// what it materialises and what the script assigns, so it reads like the plain property
/// go-ethereum exposes: the same object on every read, and an assignment sticks, also one made
/// before any read.
///
/// It must stay an accessor: redefining the property from inside its own getter or setter
/// corrupts Boa's inline cache, which pairs the slot it went through with the object's shape
/// *after* the call, so the next access from the same expression misreads the property.
fn define_lazy_bigint(
    obj: &JsObject,
    name: &'static str,
    value: U256,
    ctx: &mut Context,
) -> JsResult<()> {
    let cell = Gc::new(GcRefCell::new(None::<JsValue>));
    let getter = NativeFunction::from_copy_closure_with_captures(
        move |_this, _args, cell: &Gc<GcRefCell<Option<JsValue>>>, ctx| {
            if let Some(current) = cell.borrow().clone() {
                return Ok(current);
            }
            let big_int = to_bigint(value, ctx)?;
            *cell.borrow_mut() = Some(big_int.clone());
            Ok(big_int)
        },
        cell.clone(),
    )
    .to_js_function(ctx.realm());
    let setter = NativeFunction::from_copy_closure_with_captures(
        |_this, args, cell: &Gc<GcRefCell<Option<JsValue>>>, _ctx| {
            *cell.borrow_mut() = Some(args.get_or_undefined(0).clone());
            Ok(JsValue::undefined())
        },
        cell,
    )
    .to_js_function(ctx.realm());

    obj.define_property_or_throw(
        JsString::from(name),
        PropertyDescriptor::builder()
            .get(getter)
            .set(setter)
            .enumerable(true)
            .configurable(true)
            .build(),
        ctx,
    )?;
    Ok(())
}

/// The `ctx` object that represents the context in which the transaction is executed.
pub(crate) struct JsEvmContext {
    /// String, one of the two values CALL and CREATE
    pub(crate) r#type: String,
    /// Sender of the transaction
    pub(crate) from: Address,
    /// Target of the transaction
    pub(crate) to: Option<Address>,
    pub(crate) input: Bytes,
    /// Gas limit
    pub(crate) gas: u64,
    /// Number, amount of gas used in executing the transaction (excludes txdata costs)
    pub(crate) gas_used: u64,
    /// big.int, the effective gas price of the transaction being executed: base fee plus tip,
    /// capped at the fee cap. This is the price the sender pays per gas, so `gasUsed * gasPrice`
    /// is the execution fee.
    pub(crate) gas_price: U256,
    /// Number, intrinsic gas of the transaction. go-ethereum dropped this field in #26048, so it
    /// has no geth counterpart. Not implemented here either: the value is always 0.
    pub(crate) intrinsic_gas: u64,
    /// big.int Amount to be transferred in wei
    pub(crate) value: U256,
    /// Number, block number
    pub(crate) block: u64,
    /// Address, miner of the block
    pub(crate) coinbase: Address,
    pub(crate) output: Bytes,
    /// Number, block timestamp. go-ethereum's `ctx.time` held the tracer's wall time and was
    /// dropped in #26291 for determinism; a block timestamp keeps the output deterministic.
    pub(crate) time: String,
    pub(crate) transaction_ctx: TransactionContext,
    /// returns information about the error if one occurred, otherwise returns undefined
    pub(crate) error: Option<String>,
}

impl JsEvmContext {
    pub(crate) fn into_js_object(self, ctx: &mut Context) -> JsResult<JsObject> {
        let Self {
            r#type,
            from,
            to,
            input,
            gas,
            gas_used,
            gas_price,
            intrinsic_gas,
            value,
            block,
            coinbase,
            output,
            time,
            transaction_ctx,
            error,
        } = self;
        let obj = JsObject::with_object_proto(ctx.intrinsics());

        // add properties

        obj.set(js_string!("type"), js_string!(r#type), false, ctx)?;
        obj.set(js_string!("from"), address_to_uint8_array(from, ctx)?, false, ctx)?;
        if let Some(to) = to {
            obj.set(js_string!("to"), address_to_uint8_array(to, ctx)?, false, ctx)?;
        } else {
            obj.set(js_string!("to"), JsValue::null(), false, ctx)?;
        }

        obj.set(js_string!("input"), to_uint8_array(input, ctx)?, false, ctx)?;
        obj.set(js_string!("gas"), gas, false, ctx)?;
        obj.set(js_string!("gasUsed"), gas_used, false, ctx)?;
        define_lazy_bigint(&obj, "gasPrice", gas_price, ctx)?;
        obj.set(js_string!("intrinsicGas"), intrinsic_gas, false, ctx)?;
        define_lazy_bigint(&obj, "value", value, ctx)?;
        obj.set(js_string!("block"), block, false, ctx)?;
        obj.set(js_string!("coinbase"), address_to_uint8_array(coinbase, ctx)?, false, ctx)?;
        obj.set(js_string!("output"), to_uint8_array(output, ctx)?, false, ctx)?;
        obj.set(js_string!("time"), js_string!(time), false, ctx)?;
        if let Some(block_hash) = transaction_ctx.block_hash {
            obj.set(js_string!("blockHash"), to_uint8_array(block_hash, ctx)?, false, ctx)?;
        }
        if let Some(tx_index) = transaction_ctx.tx_index {
            obj.set(js_string!("txIndex"), tx_index as u64, false, ctx)?;
        }
        if let Some(tx_hash) = transaction_ctx.tx_hash {
            obj.set(js_string!("txHash"), to_uint8_array(tx_hash, ctx)?, false, ctx)?;
        }
        if let Some(error) = error {
            obj.set(js_string!("error"), js_string!(error), false, ctx)?;
        }

        Ok(obj)
    }
}

/// The state an instruction is about to change, captured before it runs.
///
/// The `step` hook fires after the instruction has run, once its cost is known, but has to see
/// the state as it was before, like geth's `OnOpcode`. Within a single instruction revm changes
/// state only for `SSTORE` (one slot) and `SELFDESTRUCT` (the balances it moves); calls and
/// creations change it later, when the new frame starts. Recording those few values is enough
/// to answer `db` reads as of before the instruction.
#[derive(Clone, Debug, Default)]
pub(crate) struct PreStepState {
    /// Accounts as they were, `None` if the account did not exist.
    pub(crate) accounts: Vec<(Address, Option<AccountInfo>)>,
    /// A storage slot and its value as it was.
    pub(crate) slot: Option<(Address, U256, U256)>,
}

/// DB is the object that allows the js inspector to interact with the database.
#[derive(Clone)]
pub(crate) struct EvmDbRef {
    inner: Rc<EvmDbRefInner>,
}

impl core::fmt::Debug for EvmDbRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EvmDbRef").finish_non_exhaustive()
    }
}

impl EvmDbRef {
    /// Creates a new evm and db JS object.
    pub(crate) fn new<'a, 'b, DB>(state: &'a EvmState, db: &'b mut DB) -> (Self, EvmDbGuard<'a, 'b>)
    where
        DB: Database,
        DB::Error: core::fmt::Display,
    {
        Self::with_pre_state(state, db, PreStepState::default())
    }

    /// Like [`Self::new`], but for the state the transaction has left behind: an account it
    /// self-destructed reads as absent, as it does to geth's `GetResult`, which runs after
    /// `Finalise` has deleted it.
    pub(crate) fn finalised<'a, 'b, DB>(
        state: &'a EvmState,
        db: &'b mut DB,
    ) -> (Self, EvmDbGuard<'a, 'b>)
    where
        DB: Database,
        DB::Error: core::fmt::Display,
    {
        Self::build(state, db, PreStepState::default(), true)
    }

    /// Like [`Self::new`], but answers reads of what `pre_state` covers from it, so the view is
    /// the state before the instruction that `pre_state` was captured for.
    pub(crate) fn with_pre_state<'a, 'b, DB>(
        state: &'a EvmState,
        db: &'b mut DB,
        pre_state: PreStepState,
    ) -> (Self, EvmDbGuard<'a, 'b>)
    where
        DB: Database,
        DB::Error: core::fmt::Display,
    {
        Self::build(state, db, pre_state, false)
    }

    fn build<'a, 'b, DB>(
        state: &'a EvmState,
        db: &'b mut DB,
        pre_state: PreStepState,
        finalised: bool,
    ) -> (Self, EvmDbGuard<'a, 'b>)
    where
        DB: Database,
        DB::Error: core::fmt::Display,
    {
        let (state, state_guard) = StateRef::new(state, finalised);

        // SAFETY:
        //
        // boa requires 'static lifetime for all objects.
        // As mentioned in the `Safety` section of [GuardedNullableGc] the caller of this function
        // needs to guarantee that the passed-in lifetime is sufficiently long for the lifetime of
        // the guard.
        let db = JsDb(RefCell::new(db));
        let js_db = unsafe {
            core::mem::transmute::<
                Box<dyn DatabaseRef<Error = StringError> + '_>,
                Box<dyn DatabaseRef<Error = StringError> + 'static>,
            >(Box::new(db))
        };

        let (db, db_guard) = GcDb::new(js_db);

        let inner = EvmDbRefInner { state, db, pre_state };
        let this = Self { inner: Rc::new(inner) };
        let guard = EvmDbGuard { _state_guard: state_guard, _db_guard: db_guard };
        (this, guard)
    }

    fn read_basic(&self, address: JsValue, ctx: &mut Context) -> JsResult<Option<AccountInfo>> {
        let buf = bytes_from_value(address, ctx)?;
        let address = bytes_to_address(&buf);
        if let Some((_, acc)) = self.inner.pre_state.accounts.iter().find(|(a, _)| *a == address) {
            return Ok(acc.clone());
        }
        // A destroyed account is gone, not merely unknown: the database must not answer for it,
        // since before EIP-6780 it may still hold the account as it was before the transaction.
        if let Some(acc) = self.inner.state.get_account(&address) {
            return Ok(acc);
        }

        let res = self.inner.db.0.with_inner(|db| db.basic_ref(address));
        match res {
            Some(Ok(maybe_acc)) => Ok(maybe_acc),
            _ => Err(JsError::from_native(
                JsNativeError::error()
                    .with_message(format!("Failed to read address {address:?} from database",)),
            )),
        }
    }

    fn read_code(&self, address: JsValue, ctx: &mut Context) -> JsResult<JsUint8Array> {
        let acc = self.read_basic(address, ctx)?;
        let code_hash = acc.as_ref().map(|acc| acc.code_hash).unwrap_or(KECCAK_EMPTY);
        if code_hash == KECCAK_EMPTY {
            return JsUint8Array::from_iter(core::iter::empty(), ctx);
        }

        if let Some(bytecode) = acc.as_ref().and_then(|acc| acc.code.as_ref()) {
            return to_uint8_array(bytecode.original_bytes().to_vec(), ctx);
        }

        let Some(Ok(bytecode)) = self.inner.db.0.with_inner(|db| db.code_by_hash_ref(code_hash))
        else {
            return Err(JsError::from_native(
                JsNativeError::error()
                    .with_message(format!("Failed to read code hash {code_hash:?} from database")),
            ));
        };

        to_uint8_array(bytecode.original_bytes().to_vec(), ctx)
    }

    fn read_state(
        &self,
        address: JsValue,
        slot: JsValue,
        ctx: &mut Context,
    ) -> JsResult<JsUint8Array> {
        let buf = bytes_from_value(address, ctx)?;
        let address = bytes_to_address(&buf);

        let buf = bytes_from_value(slot, ctx)?;
        let slot = bytes_to_b256(&buf);

        let slot = U256::from_be_bytes(slot.0);
        if let Some((_, _, value)) =
            self.inner.pre_state.slot.filter(|(a, s, _)| *a == address && *s == slot)
        {
            return to_uint8_array(B256::from(value), ctx);
        }
        // geth reads the live state, so a slot this transaction has written reads back the
        // written value rather than the one committed before the transaction.
        if let Some(value) = self.inner.state.get_storage(&address, slot) {
            return to_uint8_array(B256::from(value), ctx);
        }

        let res = self.inner.db.0.with_inner(|db| db.storage_ref(address, slot));

        let value = match res {
            Some(Ok(value)) => value,
            _ => {
                return Err(JsError::from_native(JsNativeError::error().with_message(format!(
                    "Failed to read state for {address:?} at {:?} from database",
                    B256::from(slot)
                ))))
            }
        };
        to_uint8_array(B256::from(value), ctx)
    }

    pub(crate) fn into_js_object(self, ctx: &mut Context) -> JsResult<JsObject> {
        let obj = JsObject::with_object_proto(ctx.intrinsics());
        let exists = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, db, ctx| {
                    let val = args.get_or_undefined(0).clone();
                    let acc = db.read_basic(val, ctx)?;
                    let exists = acc.is_some();
                    Ok(JsValue::from(exists))
                },
                self.clone(),
            ),
        )
        .length(1)
        .build();

        let get_balance = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, db, ctx| {
                    let val = args.get_or_undefined(0).clone();
                    let acc = db.read_basic(val, ctx)?;
                    let balance = acc.map(|acc| acc.balance).unwrap_or_default();
                    to_bigint(balance, ctx)
                },
                self.clone(),
            ),
        )
        .length(1)
        .build();

        let get_nonce = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, db, ctx| {
                    let val = args.get_or_undefined(0).clone();
                    let acc = db.read_basic(val, ctx)?;
                    let nonce = acc.map(|acc| acc.nonce).unwrap_or_default();
                    Ok(JsValue::from(nonce))
                },
                self.clone(),
            ),
        )
        .length(1)
        .build();

        let get_code = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, db, ctx| {
                    let val = args.get_or_undefined(0).clone();
                    Ok(db.read_code(val, ctx)?.into())
                },
                self.clone(),
            ),
        )
        .length(1)
        .build();

        let get_state = FunctionObjectBuilder::new(
            ctx.realm(),
            NativeFunction::from_copy_closure_with_captures(
                move |_this, args, db, ctx| {
                    let addr = args.get_or_undefined(0).clone();
                    let slot = args.get_or_undefined(1).clone();
                    Ok(db.read_state(addr, slot, ctx)?.into())
                },
                self,
            ),
        )
        .length(2)
        .build();

        obj.set(js_string!("getBalance"), get_balance, false, ctx)?;
        obj.set(js_string!("getNonce"), get_nonce, false, ctx)?;
        obj.set(js_string!("getCode"), get_code, false, ctx)?;
        obj.set(js_string!("getState"), get_state, false, ctx)?;
        obj.set(js_string!("exists"), exists, false, ctx)?;
        Ok(obj)
    }
}

impl Finalize for EvmDbRef {}

unsafe impl Trace for EvmDbRef {
    empty_trace!();
}

/// DB is the object that allows the js inspector to interact with the database.
struct EvmDbRefInner {
    state: StateRef,
    db: GcDb<Box<dyn DatabaseRef<Error = StringError> + 'static>>,
    pre_state: PreStepState,
}

/// Guard the inner references, once this value is dropped the inner reference is also removed.
///
/// This ensures that the guards are dropped within the lifetime of the borrowed values.
#[must_use]
pub(crate) struct EvmDbGuard<'a, 'b> {
    _state_guard: GcGuard<'a, EvmState>,
    _db_guard: GcGuard<'b, Box<dyn DatabaseRef<Error = StringError> + 'static>>,
}

/// A wrapper Database for the JS context.
///
/// Wraps a `&mut DB: Database` in a `RefCell` so that the `&self` methods of
/// [`DatabaseRef`] can drive the underlying `Database` (which requires `&mut self`).
pub(crate) struct JsDb<'a, DB>(RefCell<&'a mut DB>);

#[derive(Clone, Debug)]
pub(crate) struct StringError(pub String);

impl core::error::Error for StringError {}
impl DBErrorMarker for StringError {}

impl core::fmt::Display for StringError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for StringError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl<DB> DatabaseRef for JsDb<'_, DB>
where
    DB: Database,
    DB::Error: core::fmt::Display,
{
    type Error = StringError;

    fn basic_ref(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        self.0.borrow_mut().basic(address).map_err(|e| e.to_string().into())
    }

    fn code_by_hash_ref(&self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        self.0.borrow_mut().code_by_hash(code_hash).map_err(|e| e.to_string().into())
    }

    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        self.0.borrow_mut().storage(address, index).map_err(|e| e.to_string().into())
    }

    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        self.0.borrow_mut().block_hash(number).map_err(|e| e.to_string().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::js::builtins::{json_stringify, register_builtins, to_serde_value};
    use boa_engine::Source;
    use revm::{database::CacheDB, database_interface::EmptyDB, interpreter::Stack};

    #[test]
    fn test_contract() {
        let mut ctx = Context::default();
        let contract = Contract {
            caller: Address::ZERO,
            contract: Address::ZERO,
            value: U256::from(1337u64),
            input: vec![0x01, 0x02, 0x03].into(),
        };
        register_builtins(&mut ctx).unwrap();

        let (stack_ref, _stack_guard) = StackRef::new(Vec::new());
        let (mem_ref, _mem_guard) = MemoryRef::new(MemorySnapshot::unchanged(0), &[]);
        let step = StepLog {
            stack: stack_ref,
            op: OpObj(0),
            memory: mem_ref,
            pc: 0,
            gas_remaining: 0,
            cost: 0,
            depth: 0,
            refund: 0,
            error: None,
            contract: contract.clone(),
        };
        let reusable_step = ReusableStepLog::new(&mut ctx).unwrap();
        reusable_step.update(step);

        let s = "({
                caller: function(log) { return log.contract.getCaller(); },
                value: function(log) { return log.contract.getValue(); },
                address: function(log) { return log.contract.getAddress(); },
                input: function(log) { return log.contract.getInput(); }
        })";

        let log_arg = reusable_step.value();
        let eval_obj = ctx.eval(Source::from_bytes(s)).unwrap();
        let call = eval_obj.as_object().unwrap().get(js_string!("caller"), &mut ctx).unwrap();
        let res = call
            .as_callable()
            .unwrap()
            .call(&JsValue::undefined(), core::slice::from_ref(&log_arg), &mut ctx)
            .unwrap();
        assert!(res.is_object());
        let obj = res.as_object().unwrap();
        let array_buf = JsUint8Array::from_object(obj.clone());
        assert!(array_buf.is_ok());

        let get_address =
            eval_obj.as_object().unwrap().get(js_string!("address"), &mut ctx).unwrap();
        let res = get_address
            .as_callable()
            .unwrap()
            .call(&JsValue::undefined(), core::slice::from_ref(&log_arg), &mut ctx)
            .unwrap();
        assert!(res.is_object());

        let buf = bytes_from_value(res, &mut ctx).unwrap();
        assert_eq!(buf, contract.contract.as_slice());

        let call = eval_obj.as_object().unwrap().get(js_string!("value"), &mut ctx).unwrap();
        let res = call
            .as_callable()
            .unwrap()
            .call(&JsValue::undefined(), core::slice::from_ref(&log_arg), &mut ctx)
            .unwrap();
        assert_eq!(
            res.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            contract.value.to_string()
        );

        let call = eval_obj.as_object().unwrap().get(js_string!("input"), &mut ctx).unwrap();
        let res =
            call.as_callable().unwrap().call(&JsValue::undefined(), &[log_arg], &mut ctx).unwrap();

        let buf = bytes_from_value(res, &mut ctx).unwrap();
        assert_eq!(buf, contract.input);
    }

    #[test]
    fn test_evm_db_gc() {
        let mut context = Context::default();

        let result = context
            .eval(Source::from_bytes(
                "(
                    function(db, addr) {return db.exists(addr) }
            )
        "
                .to_string()
                .as_bytes(),
            ))
            .unwrap();
        assert!(result.is_callable());

        let f = result.as_callable().unwrap();

        let mut db = CacheDB::new(EmptyDB::new());
        let state = EvmState::default();
        let reusable_db = ReusableEvmDb::new(&mut context).unwrap();
        {
            let (db, guard) = EvmDbRef::new(&state, &mut db);
            let addr = Address::default();
            let addr = JsValue::from(js_string!(addr.to_string()));
            reusable_db.update(db);
            let db = reusable_db.value();
            let res = f.call(&result, &[db.clone(), addr.clone()], &mut context).unwrap();
            assert!(!res.as_boolean().unwrap());

            // drop the db which also drops any GC values
            drop(guard);
            let res = f.call(&result, &[db.clone(), addr.clone()], &mut context);
            assert!(res.is_err());
        }
        let addr = Address::default();
        db.insert_account_info(addr, Default::default());

        {
            let (db, guard) = EvmDbRef::new(&state, &mut db);
            let addr = JsValue::from(js_string!(addr.to_string()));
            reusable_db.update(db);
            let db = reusable_db.value();
            let res = f.call(&result, &[db.clone(), addr.clone()], &mut context).unwrap();

            // account exists
            assert!(res.as_boolean().unwrap());

            // drop the db which also drops any GC values
            drop(guard);
            let res = f.call(&result, &[db.clone(), addr.clone()], &mut context);
            assert!(res.is_err());
        }
    }

    #[test]
    fn test_evm_db_gc_captures() {
        let mut context = Context::default();

        let res = context
            .eval(Source::from_bytes(
                r"({
                 setup: function(db) {this.db = db;},
                 result: function(addr) {return this.db.exists(addr) }
            })
        "
                .to_string()
                .as_bytes(),
            ))
            .unwrap();

        let obj = res.as_object().unwrap();

        let result_fn = obj.get(js_string!("result"), &mut context).unwrap().as_object().unwrap();
        let setup_fn = obj.get(js_string!("setup"), &mut context).unwrap().as_object().unwrap();

        let mut db = CacheDB::new(EmptyDB::new());
        let state = EvmState::default();
        {
            let (db_ref, guard) = EvmDbRef::new(&state, &mut db);
            let reusable_db = ReusableEvmDb::new(&mut context).unwrap();
            reusable_db.update(db_ref);
            let _res =
                setup_fn.call(&(obj.clone().into()), &[reusable_db.value()], &mut context).unwrap();
            assert!(obj.get(js_string!("db"), &mut context).unwrap().is_object());

            let addr = Address::default();
            let addr = JsValue::from(js_string!(addr.to_string()));
            let res = result_fn
                .call(&(obj.clone().into()), core::slice::from_ref(&addr), &mut context)
                .unwrap();
            assert!(!res.as_boolean().unwrap());

            // drop the guard which also drops any GC values
            drop(guard);
            let res = result_fn.call(&(obj.clone().into()), &[addr], &mut context);
            assert!(res.is_err());
        }
    }

    #[test]
    fn test_big_int() {
        let mut context = Context::default();
        register_builtins(&mut context).unwrap();

        let eval = context
            .eval(Source::from_bytes(
                r#"({data: [], fault: function(log) {}, step: function(log) { this.data.push({ value: log.stack.peek(2) }) }, result: function() { return this.data; }})"#
                .to_string()
                .as_bytes(),
            ))
            .unwrap();

        let obj = eval.as_object().unwrap();

        let result_fn = obj.get(js_string!("result"), &mut context).unwrap().as_object().unwrap();
        let step_fn = obj.get(js_string!("step"), &mut context).unwrap().as_object().unwrap();

        let mut stack = Stack::new();
        let _ = stack.push(U256::from(35000));
        let _ = stack.push(U256::from(35000));
        let _ = stack.push(U256::from(35000));
        let (stack_ref, _stack_guard) = StackRef::new(stack.data().clone());
        let (mem_ref, _mem_guard) = MemoryRef::new(MemorySnapshot::unchanged(0), &[]);

        let step = StepLog {
            stack: stack_ref,
            op: OpObj(0),
            memory: mem_ref,
            pc: 0,
            gas_remaining: 0,
            cost: 0,
            depth: 0,
            refund: 0,
            error: None,
            contract: Default::default(),
        };

        let reusable_step = ReusableStepLog::new(&mut context).unwrap();
        reusable_step.update(step);

        let _ = step_fn.call(&eval, &[reusable_step.value()], &mut context).unwrap();

        let res = result_fn.call(&eval, &[], &mut context).unwrap();
        let val = json_stringify(res.clone(), &mut context).unwrap().to_std_string().unwrap();
        assert_eq!(val, r#"[{"value":"35000"}]"#);

        let val = to_serde_value(res, &mut context).unwrap();
        assert!(val.is_array());
        let s = val.to_string();
        assert_eq!(s, r#"[{"value":"35000"}]"#);
    }

    #[test]
    fn test_object_functions() {
        let mut context = Context::default();
        register_builtins(&mut context).unwrap();

        let eval = context
            .eval(Source::from_bytes(
                r#"(
    {
        retVal: [],
        callStack: [],
        byte2Hex: function (byte) {
            if (byte < 0x10) return "0" + byte.toString(16);
            return byte.toString(16);
        },
        array2Hex: function (arr) {
            var retVal = "";
            for (var i = 0; i < arr.length; i++) retVal += this.byte2Hex(arr[i]);
            return retVal;
        },
        getAddr: function (log) {
            return this.array2Hex(log.contract.getAddress());
        },
        step: function (log, db) {
            var opcode = log.op.toNumber();
            if (opcode == 0x54) {
                this.retVal.push(this.getAddr(log) + ":" + log.stack.peek(0).toString(16));
            }
            if (opcode == 0x55)
                this.retVal.push(
                    this.getAddr(log) +
                        ":" +
                        log.stack.peek(0).toString(16) +
                        ";" +
                        log.stack.peek(1).toString(16)
                );
        },
        fault: function (log, db) {
            this.retVal.push("FAULT: ");
        },
        result: function (ctx, db) {
            return this.retVal;
        },
   }
)"#
                .to_string()
                .as_bytes(),
            ))
            .unwrap();

        let obj = eval.as_object().unwrap();

        let result_fn = obj.get(js_string!("result"), &mut context).unwrap().as_object().unwrap();
        let step_fn = obj.get(js_string!("step"), &mut context).unwrap().as_object().unwrap();

        let mut stack = Stack::new();
        let _ = stack.push(U256::from(35000));
        let _ = stack.push(U256::from(35000));
        let _ = stack.push(U256::from(35000));
        let (stack_ref, _stack_guard) = StackRef::new(stack.data().clone());
        let (mem_ref, _mem_guard) = MemoryRef::new(MemorySnapshot::unchanged(0), &[]);

        let step = StepLog {
            stack: stack_ref,
            op: OpObj(85),
            memory: mem_ref,
            pc: 0,
            gas_remaining: 0,
            cost: 0,
            depth: 0,
            refund: 0,
            error: None,
            contract: Default::default(),
        };

        let reusable_step = ReusableStepLog::new(&mut context).unwrap();
        reusable_step.update(step);

        let _ = step_fn.call(&eval, &[reusable_step.value()], &mut context).unwrap();

        let res = result_fn.call(&eval, &[], &mut context).unwrap();
        let val = json_stringify(res.clone(), &mut context).unwrap().to_std_string().unwrap();
        assert_eq!(val, r#"["0000000000000000000000000000000000000000:88b8;88b8"]"#);
    }
}
