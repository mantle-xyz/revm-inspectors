//! Builtin functions

use alloc::{borrow::Cow, format, string::ToString, vec::Vec};
use alloy_primitives::{hex, map::HashSet, Address, FixedBytes, B256, U256};
use boa_engine::{
    builtins::{array_buffer::ArrayBuffer, typed_array::TypedArray},
    js_string,
    object::builtins::{JsArray, JsArrayBuffer, JsTypedArray, JsUint8Array},
    property::PropertyDescriptor,
    Context, JsArgs, JsError, JsNativeError, JsResult, JsString, JsValue, NativeFunction, Source,
};
use boa_gc::{empty_trace, Finalize, Trace};
use core::borrow::Borrow;

/// Converts the given `JsValue` to a `serde_json::Value`.
///
/// Serialization goes through `JSON.stringify` so that `toJSON` properties are honoured, which
/// is what lets a big integer come out as a decimal string. Boa's own `to_json` is the
/// fallback; if it fails too, the `JSON.stringify` error is reported, being the informative
/// one - a circular structure names itself there and not in the fallback.
pub(crate) fn to_serde_value(val: JsValue, ctx: &mut Context) -> JsResult<serde_json::Value> {
    let stringify_err = match json_stringify(val.clone(), ctx) {
        Ok(json) => {
            let json = json.to_std_string().map_err(|err| {
                JsError::from_native(
                    JsNativeError::error()
                        .with_message(format!("failed to convert JSON to string: {err}")),
                )
            })?;
            return serde_json::from_str(&json).map_err(|err| {
                JsError::from_native(
                    JsNativeError::error().with_message(format!("failed to parse JSON: {err}")),
                )
            });
        }
        Err(err) => err,
    };

    val.to_json(ctx)?.ok_or(stringify_err)
}

/// Attempts to use the global `JSON` object to stringify the given value.
///
/// `JSON.stringify` answers the JavaScript value `undefined` for `undefined`, functions and
/// symbols. Rendering that as the *string* `"undefined"` would produce text no JSON parser
/// accepts, so it is mapped to `null`, as go-ethereum's `json.Marshal` does.
pub(crate) fn json_stringify(val: JsValue, ctx: &mut Context) -> JsResult<JsString> {
    let json = ctx.global_object().get(js_string!("JSON"), ctx)?;
    let json_obj = json.as_object().ok_or_else(|| {
        JsError::from_native(JsNativeError::typ().with_message("JSON is not an object"))
    })?;

    let stringify = json_obj.get(js_string!("stringify"), ctx)?;

    let stringify = stringify.as_callable().ok_or_else(|| {
        JsError::from_native(JsNativeError::typ().with_message("JSON.stringify is not callable"))
    })?;
    let res = stringify.call(&json, &[val], ctx)?;
    if res.is_undefined() {
        return Ok(js_string!("null"));
    }
    res.to_string(ctx)
}

/// The `bigInt` environment go-ethereum exposes to JS tracers, verbatim.
///
/// See `src/tracing/js/bigint.js` for provenance.
const BIG_INT_JS: &str = include_str!("bigint.js");

/// The global names the constructor is reachable under: `bigInt` is geth's, `bigint` is the
/// alias [`to_bigint`] looks up. Both resolve to the same object.
const BIG_INT_GLOBALS: [&str; 2] = ["bigInt", "bigint"];

/// Parses and evaluates the library, returning its constructor. The IIFE keeps its top-level
/// `var bigInt` function-scoped: at global scope it would not displace the accessor from
/// [`install_bigint`], and the trailing `bigInt` expression would re-enter it forever.
fn eval_bigint(ctx: &mut Context) -> JsResult<JsValue> {
    let src = format!("(function(){{\n{BIG_INT_JS}\n;return bigInt}})()");
    let big_int = ctx.eval(Source::from_bytes(src.as_str()))?;
    if !big_int.is_callable() {
        return Err(JsError::from_native(
            JsNativeError::typ().with_message("failed to install the bigInt environment"),
        ));
    }
    Ok(big_int)
}

/// Replaces a global with a plain writable data property holding `value`.
fn define_bigint_global(ctx: &mut Context, name: &str, value: JsValue) -> JsResult<()> {
    let desc = PropertyDescriptor::builder()
        .value(value)
        .writable(true)
        .enumerable(true)
        .configurable(true)
        .build();
    ctx.global_object().define_property_or_throw(JsString::from(name), desc, ctx)?;
    Ok(())
}

/// Evaluates the library on first access and replaces the accessors with the constructor.
fn bigint_lazy_getter(_this: &JsValue, _args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let big_int = eval_bigint(ctx)?;
    for name in BIG_INT_GLOBALS {
        define_bigint_global(ctx, name, big_int.clone())?;
    }
    Ok(big_int)
}

/// Accepts an assignment to one of the globals before the library is installed. Without a
/// setter `bigInt = x` would be dropped until something read the global and only take effect
/// afterwards; geth's `bigInt` is an ordinary writable global throughout.
fn bigint_lazy_setter(
    _this: &JsValue,
    args: &[JsValue],
    name: &&'static str,
    ctx: &mut Context,
) -> JsResult<JsValue> {
    // Only the assigned name is replaced; the sibling alias keeps its own binding.
    define_bigint_global(ctx, name, args.get_or_undefined(0).clone())?;
    Ok(JsValue::undefined())
}

/// Installs go-ethereum's `bigInt` environment lazily: parsing the 26 KB library dwarfs
/// building the [`Context`], and a fresh one is built per traced transaction. Triggering on
/// property access cannot miss a reference such as `globalThis['big' + 'Int']`.
fn install_bigint(ctx: &mut Context) -> JsResult<()> {
    let getter = NativeFunction::from_fn_ptr(bigint_lazy_getter).to_js_function(ctx.realm());
    for name in BIG_INT_GLOBALS {
        let setter = NativeFunction::from_copy_closure_with_captures(bigint_lazy_setter, name)
            .to_js_function(ctx.realm());
        let desc = PropertyDescriptor::builder()
            .get(getter.clone())
            .set(setter)
            .enumerable(true)
            .configurable(true)
            .build();
        ctx.global_object().define_property_or_throw(JsString::from(name), desc, ctx)?;
    }
    Ok(())
}

/// Registers all the builtin functions.
///
/// Note: this does not register the `isPrecompiled` builtin, as this requires the precompile
/// addresses, see [PrecompileList::register_callable].
pub(crate) fn register_builtins(ctx: &mut Context) -> JsResult<()> {
    install_bigint(ctx)?;
    // Additive shims on Boa's *native* `BigInt`, which goja lacks entirely; BigInteger.js
    // values sit on a disjoint prototype chain, so these cannot affect them. `toJSON` is
    // load-bearing: `JSON.stringify` in `to_serde_value` throws on a bare native BigInt.
    ctx.eval(Source::from_bytes(
        br#"
BigInt.prototype.toJSON = function() { return this.toString(); };
BigInt.prototype.equals = function(other) { return this == other; };
BigInt.prototype.toJSNumber = function() { return Number(this); };
BigInt.prototype.plus = function(other) { return this + BigInt(other); };
BigInt.prototype.minus = function(other) { return this - BigInt(other); };
"#,
    ))?;
    ctx.register_global_builtin_callable(
        js_string!("toHex"),
        1,
        NativeFunction::from_fn_ptr(to_hex),
    )?;
    ctx.register_global_callable(js_string!("toWord"), 1, NativeFunction::from_fn_ptr(to_word))?;
    ctx.register_global_callable(
        js_string!("toAddress"),
        1,
        NativeFunction::from_fn_ptr(to_address),
    )?;
    ctx.register_global_callable(
        js_string!("toContract"),
        2,
        NativeFunction::from_fn_ptr(to_contract),
    )?;
    ctx.register_global_callable(
        js_string!("toContract2"),
        3,
        NativeFunction::from_fn_ptr(to_contract2),
    )?;
    ctx.register_global_callable(js_string!("slice"), 3, NativeFunction::from_fn_ptr(slice))?;

    Ok(())
}

/// Converts an array or Uint8Array to a byte array, and a hex string too when `allow_string`.
///
/// go-ethereum draws the same line per call site: the functions that normalise a value into
/// bytes take a string, the ones that consume bytes do not. Accepting one where it consumes
/// bytes turns a caller's mistake into a plausible answer — `db.getBalance("0x1234")` would
/// left-pad to `0x00..1234` and report that account's balance.
pub(crate) fn bytes_from_value(
    val: JsValue,
    allow_string: bool,
    context: &mut Context,
) -> JsResult<Vec<u8>> {
    if let Some(obj) = val.as_object() {
        if obj.is::<TypedArray>() {
            let array: JsTypedArray = JsTypedArray::from_object(obj)?;
            let len = array.length(context)?;
            let mut buf = Vec::with_capacity(len);
            for i in 0..len {
                let val = array.get(i, context)?;
                buf.push(val.to_number(context)? as u8);
            }
            return Ok(buf);
        } else if obj.is::<ArrayBuffer>() {
            let buf = JsArrayBuffer::from_object(obj)?;
            let buf = buf.data().map(|data| data.to_vec()).ok_or_else(|| {
                JsNativeError::typ().with_message("ArrayBuffer was already detached")
            })?;
            return Ok(buf);
        } else if obj.is::<JsString>() {
            if !allow_string {
                return Err(invalid_buffer_type(&val));
            }
            let js_string = obj.downcast_ref::<JsString>().unwrap();
            return hex_decode_js_string(js_string.borrow());
        } else if obj.is_array() {
            let array = JsArray::from_object(obj)?;
            let len = array.length(context)?;
            let mut buf = Vec::with_capacity(len as usize);
            for i in 0..len {
                let val = array.get(i, context)?;
                buf.push(val.to_number(context)? as u8);
            }
            return Ok(buf);
        }
    }

    if allow_string {
        if let Some(js_string) = val.as_string() {
            return hex_decode_js_string(&js_string);
        }
    }

    Err(invalid_buffer_type(&val))
}

/// The error go-ethereum's `fromBuf` raises for an unusable argument.
fn invalid_buffer_type(val: &JsValue) -> JsError {
    JsError::from_native(
        JsNativeError::typ().with_message(format!("invalid buffer type: {}", val.type_of())),
    )
}

/// Create a new [JsUint8Array] array buffer from the address' bytes.
pub(crate) fn address_to_uint8_array(
    addr: Address,
    context: &mut Context,
) -> JsResult<JsUint8Array> {
    JsUint8Array::from_iter(addr, context)
}

/// Create a new [JsUint8Array] array buffer from the address' bytes.
pub(crate) fn address_to_uint8_array_value(
    addr: Address,
    context: &mut Context,
) -> JsResult<JsValue> {
    address_to_uint8_array(addr, context).map(Into::into)
}

/// Create a new [JsUint8Array] from byte block.
pub(crate) fn to_uint8_array<I>(bytes: I, context: &mut Context) -> JsResult<JsUint8Array>
where
    I: IntoIterator<Item = u8>,
{
    JsUint8Array::from_iter(bytes, context)
}

/// Create a new [JsUint8Array] object from byte block.
pub(crate) fn to_uint8_array_value<I>(bytes: I, context: &mut Context) -> JsResult<JsValue>
where
    I: IntoIterator<Item = u8>,
{
    to_uint8_array(bytes, context).map(Into::into)
}

/// Converts a slice of bytes to an address.
///
/// See [`bytes_to_fb`] for more information.
pub(crate) fn bytes_to_address(bytes: &[u8]) -> Address {
    Address(bytes_to_fb(bytes))
}

/// Converts a slice of bytes to a 32-byte fixed-size array.
///
/// See [`bytes_to_fb`] for more information.
pub(crate) fn bytes_to_b256(bytes: &[u8]) -> B256 {
    bytes_to_fb(bytes)
}

/// Converts a slice of bytes to a fixed-size array.
///
/// If the slice is larger than the array size, it will be trimmed from the left.
pub(crate) fn bytes_to_fb<const N: usize>(mut bytes: &[u8]) -> FixedBytes<N> {
    if bytes.len() > N {
        bytes = &bytes[bytes.len() - N..];
    }
    FixedBytes::left_padding_from(bytes)
}

/// Converts a U256 to a bigint using the global bigint alias. The value is passed as a decimal
/// string, matching geth, whose `toBig` is likewise handed `value.String()`; BigInteger.js
/// accepts only decimal in its single-argument form, so a hex string would throw.
pub(crate) fn to_bigint(value: U256, ctx: &mut Context) -> JsResult<JsValue> {
    let bigint = ctx.global_object().get(js_string!("bigint"), ctx)?;
    // Erroring beats `undefined`: this feeds `stack.peek()` and `db.getBalance()`, so a silent
    // `undefined` yields a uniformly wrong trace. geth cannot hit this, holding the
    // constructor as a Go-side handle rather than looking up a mutable global.
    let bigint = bigint.as_callable().ok_or_else(|| {
        JsError::from_native(
            JsNativeError::typ()
                .with_message("global `bigint` is not callable; it was overwritten"),
        )
    })?;
    bigint.call(&JsValue::undefined(), &[JsValue::from(js_string!(value.to_string()))], ctx)
}

/// Compute the address of a contract created using CREATE2.
///
/// Arguments:
/// 1. creator: The address of the contract creator
/// 2. salt: A 32-byte salt value
/// 3. initcode: The contract's initialization code
///
/// Returns: The computed contract address as an ArrayBuffer
pub(crate) fn to_contract2(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    // Extract the sender's address, salt and initcode from the arguments
    let from = args.get_or_undefined(0).clone();
    let salt = match args.get_or_undefined(1).to_string(ctx) {
        Ok(js_string) => {
            let buf = hex_decode_js_string(&js_string)?;
            bytes_to_b256(&buf)
        }
        Err(_) => {
            return Err(JsError::from_native(
                JsNativeError::typ().with_message("invalid salt type"),
            ))
        }
    };
    let initcode = args.get_or_undefined(2).clone();

    // Convert the sender's address to a byte buffer and then to an Address
    let buf = bytes_from_value(from, true, ctx)?;
    let addr = bytes_to_address(&buf);

    // Convert the initcode to a byte buffer
    let code_buf = bytes_from_value(initcode, true, ctx)?;

    // Compute the contract address
    let contract_addr = addr.create2_from_code(salt, code_buf);

    // Convert the contract address to a byte buffer and return it as an ArrayBuffer
    address_to_uint8_array_value(contract_addr, ctx)
}

/// Compute the address of a contract created by the sender with the given nonce.
///
/// Arguments:
/// 1. from: The address of the contract creator
/// 2. nonce: The creator's transaction count (optional, none is 0)
///
/// Returns: The computed contract address as an ArrayBuffer
pub(crate) fn to_contract(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    // Extract the sender's address and nonce from the arguments
    let from = args.get_or_undefined(0).clone();
    let nonce = args.get_or_undefined(1).to_number(ctx)? as u64;

    // Convert the sender's address to a byte buffer and then to an Address
    let buf = bytes_from_value(from, true, ctx)?;
    let addr = bytes_to_address(&buf);

    // Compute the contract address
    let contract_addr = addr.create(nonce);

    // Convert the contract address to a byte buffer and return it as an ArrayBuffer
    address_to_uint8_array_value(contract_addr, ctx)
}

/// Converts a buffer type to an address
pub(crate) fn to_address(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let val = args.get_or_undefined(0).clone();
    let buf = bytes_from_value(val, true, ctx)?;
    let address = bytes_to_address(&buf);
    address_to_uint8_array_value(address, ctx)
}

/// Converts a buffer type to a word
pub(crate) fn to_word(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let val = args.get_or_undefined(0).clone();
    let buf = bytes_from_value(val, true, ctx)?;
    let hash = bytes_to_b256(&buf);
    to_uint8_array_value(hash, ctx)
}

/// Converts a buffer type to a hex string
pub(crate) fn to_hex(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let val = args.get_or_undefined(0).clone();
    let buf = bytes_from_value(val, false, ctx)?;
    let s = js_string!(hex::encode_prefixed(buf));
    Ok(JsValue::from(s))
}

/// Decodes a hex decoded js-string
fn hex_decode_js_string(js_string: &JsString) -> JsResult<Vec<u8>> {
    match js_string.to_std_string() {
        Ok(s) => {
            // hex decoding strings is pretty relaxed in geth reference implementation, which allows uneven hex values <https://github.com/ethereum/go-ethereum/blob/355228b011ef9a85ebc0f21e7196f892038d49f0/common/bytes.go#L33-L35>
            // <https://github.com/paradigmxyz/reth/issues/16289>
            let mut s = Cow::Borrowed(s.strip_prefix("0x").unwrap_or(s.as_str()));
            if s.as_ref().len() % 2 == 1 {
                s = Cow::Owned(format!("0{s}"));
            }

            match hex::decode(s.as_ref()) {
                Ok(data) => Ok(data),
                Err(err) => Err(JsError::from_native(
                    JsNativeError::error()
                        .with_message(format!("invalid hex string: \"{s}\": {err}",)),
                )),
            }
        }
        Err(err) => Err(JsError::from_native(
            JsNativeError::error()
                .with_message(format!("invalid utf8 string {js_string:?}: {err}",)),
        )),
    }
}

/// Returns a slice of the given value.
pub(crate) fn slice(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let val = args.get_or_undefined(0).clone();

    let buf = bytes_from_value(val, false, ctx)?;
    // Test the floats before converting: `f64 as usize` saturates, so `-1.0` would silently
    // become `0` and return a slice geth rejects. `MemoryRef::slice` guards the same way.
    let start_f64 = args.get_or_undefined(1).to_numeric_number(ctx)?;
    let end_f64 = args.get_or_undefined(2).to_numeric_number(ctx)?;
    let start = start_f64 as usize;
    let end = end_f64 as usize;

    if start_f64 < 0. || end_f64 < 0. || start > end || end > buf.len() {
        Err(JsError::from_native(JsNativeError::error().with_message(format!(
            "Tracer accessed out of bound memory: available {}, start {}, end {}",
            buf.len(),
            start_f64,
            end_f64
        ))))
    } else {
        to_uint8_array_value(buf[start..end].iter().copied(), ctx)
    }
}

/// A container for all precompile addresses used for the `isPrecompiled` global callable.
#[derive(Clone, Debug)]
pub(crate) struct PrecompileList(pub(crate) HashSet<Address>);

impl PrecompileList {
    /// Registers the global callable `isPrecompiled`
    pub(crate) fn register_callable(self, ctx: &mut Context) -> JsResult<()> {
        let is_precompiled = NativeFunction::from_copy_closure_with_captures(
            move |_this, args, precompiles, ctx| {
                let val = args.get_or_undefined(0).clone();
                // geth passes `allowString=true` here, unlike the `db` accessors.
                let buf = bytes_from_value(val, true, ctx)?;
                let addr = bytes_to_address(&buf);
                Ok(precompiles.0.contains(&addr).into())
            },
            self,
        );

        ctx.register_global_callable(js_string!("isPrecompiled"), 1, is_precompiled)?;

        Ok(())
    }
}

impl Finalize for PrecompileList {}

unsafe impl Trace for PrecompileList {
    empty_trace!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_install_bigint() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        // Test that 'bigint' alias exists and works
        let bigint = ctx.global_object().get(js_string!("bigint"), &mut ctx).unwrap();
        assert!(bigint.is_callable());

        let value = JsValue::from(js_string!("100"));
        let result =
            bigint.as_callable().unwrap().call(&JsValue::undefined(), &[value], &mut ctx).unwrap();
        // BigInteger.js hands back an object, not a native bigint primitive - same as geth.
        assert!(result.is_object());
        assert!(!result.is_bigint());
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "100");

        // `bigInt` (what tracers call) and `bigint` (what `to_bigint` looks up) must be the
        // very same constructor, otherwise values from the two would not interoperate.
        let same = ctx.eval(Source::from_bytes(b"bigInt === bigint")).unwrap();
        assert!(same.as_boolean().unwrap());

        // The library supplies its own `toJSON`, which `to_serde_value` relies on.
        assert_eq!(json_stringify(result, &mut ctx).unwrap().to_std_string().unwrap(), "\"100\"");
    }

    #[test]
    fn test_to_bigint_function() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        // Test various U256 values through to_bigint
        let test_cases = vec![
            (U256::ZERO, "0"),
            (U256::from(1u64), "1"),
            (U256::from(42u64), "42"),
            (U256::from(u64::MAX), "18446744073709551615"),
            (
                U256::from_str_radix("123456789012345678901234567890", 10).unwrap(),
                "123456789012345678901234567890",
            ),
        ];

        for (value, expected) in test_cases {
            let result = to_bigint(value, &mut ctx).unwrap();
            assert!(result.is_object(), "Result should be a BigInteger object for value {value}");
            let result_str = result.to_string(&mut ctx).unwrap().to_std_string().unwrap();
            assert_eq!(result_str, expected, "BigInt conversion failed for {value}");
        }

        // Test that the result can be used in JavaScript operations
        let big_value = U256::from(999u64);
        let bigint_result = to_bigint(big_value, &mut ctx).unwrap();

        // Set it as a global variable
        ctx.global_object().set(js_string!("testBigInt"), bigint_result, false, &mut ctx).unwrap();

        // Arithmetic goes through the library's own methods, as it does in geth.
        let arithmetic_test =
            ctx.eval(Source::from_bytes(b"testBigInt.add(1).toString()")).unwrap();
        assert_eq!(arithmetic_test.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "1000");

        // Test comparison
        let comparison_test = ctx.eval(Source::from_bytes(b"testBigInt.greater(500)")).unwrap();
        assert!(comparison_test.as_boolean().unwrap());

        // Values are objects now, so `typeof` reports "object" and `+` coerces through
        // `valueOf` to a Number instead of throwing the way a native bigint would. Both
        // match go-ethereum; pinned here so the change is visible if it ever regresses.
        let type_of = ctx.eval(Source::from_bytes(b"typeof testBigInt")).unwrap();
        assert_eq!(type_of.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "object");
        let coerced = ctx.eval(Source::from_bytes(b"testBigInt + 1")).unwrap();
        assert_eq!(coerced.as_number().unwrap(), 1000.0);
    }

    /// Evaluates `src` in a context with the builtins installed and returns its string value.
    fn eval_str(src: &str) -> String {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();
        ctx.eval(Source::from_bytes(src.as_bytes()))
            .unwrap()
            .to_string(&mut ctx)
            .unwrap()
            .to_std_string()
            .unwrap()
    }

    /// The library must not be evaluated until something reaches for `bigInt`. Parsing it
    /// costs ~45x what building the whole [`Context`] does and is paid per traced
    /// transaction, so a regression here is expensive and otherwise invisible.
    #[test]
    fn test_bigint_is_installed_lazily() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        let descriptor_kind = |ctx: &mut Context| {
            ctx.eval(Source::from_bytes(
                b"typeof Object.getOwnPropertyDescriptor(globalThis, 'bigInt').get",
            ))
            .unwrap()
            .to_string(ctx)
            .unwrap()
            .to_std_string()
            .unwrap()
        };

        // Untouched: still an accessor.
        assert_eq!(descriptor_kind(&mut ctx), "function");

        // Reading it installs the library and swaps in a plain data property.
        let value = ctx.eval(Source::from_bytes(b"bigInt(7).toString()")).unwrap();
        assert_eq!(value.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "7");
        assert_eq!(descriptor_kind(&mut ctx), "undefined");

        // Both names must end up bound to the same constructor.
        let same = ctx.eval(Source::from_bytes(b"bigInt === bigint")).unwrap();
        assert!(same.as_boolean().unwrap());
    }

    /// Assigning to `bigInt` must not depend on whether the library is installed yet: a
    /// bare accessor would make it a no-op before the first read and an ordinary assignment
    /// after. geth's `bigInt` is a plain writable global throughout.
    #[test]
    fn test_bigint_global_is_assignable_before_and_after_install() {
        // Before: assignment goes through the setter.
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();
        let before = ctx.eval(Source::from_bytes(b"bigInt = 1; typeof bigInt")).unwrap();
        assert_eq!(before.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "number");
        // The sibling alias is independent, as two ordinary globals would be.
        let alias = ctx.eval(Source::from_bytes(b"typeof bigint")).unwrap();
        assert_eq!(alias.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "function");

        // After: the property is a data property and assignment still works.
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();
        let after = ctx
            .eval(Source::from_bytes(b"bigInt(1).toString(); bigInt = 1; typeof bigInt"))
            .unwrap();
        assert_eq!(after.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "number");

        // And strict mode must not throw in either order.
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();
        assert!(ctx.eval(Source::from_bytes(b"'use strict'; bigInt = 1;")).is_ok());
    }

    /// Touching the lowercase alias first must install the library just the same.
    #[test]
    fn test_bigint_lazy_install_via_lowercase_alias() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        let result = to_bigint(U256::from(7u64), &mut ctx).unwrap();
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "7");

        let same = ctx.eval(Source::from_bytes(b"bigInt === bigint")).unwrap();
        assert!(same.as_boolean().unwrap());
    }

    #[test]
    fn test_bigint_arithmetic() {
        // The canonical BigInteger.js names, not just the `plus`/`minus` aliases.
        assert_eq!(eval_str("bigInt(1).add(2).toString()"), "3");
        assert_eq!(eval_str("bigInt(10).subtract(3).toString()"), "7");
        assert_eq!(eval_str("bigInt(6).multiply(7).toString()"), "42");
        assert_eq!(eval_str("bigInt(84).divide(2).toString()"), "42");
        assert_eq!(eval_str("bigInt(17).mod(5).toString()"), "2");
        assert_eq!(eval_str("bigInt(2).pow(64).toString()"), "18446744073709551616");
        assert_eq!(eval_str("bigInt(5).negate().toString()"), "-5");
        assert_eq!(eval_str("bigInt(-5).abs().toString()"), "5");
        // Arbitrary precision must hold well past f64.
        assert_eq!(
            eval_str("bigInt('9007199254740993').add('9007199254740993').toString()"),
            "18014398509481986"
        );
    }

    #[test]
    fn test_bigint_compare_and_predicates() {
        assert_eq!(eval_str("bigInt(1).compare(2).toString()"), "-1");
        assert_eq!(eval_str("bigInt(2).compare(2).toString()"), "0");
        assert_eq!(eval_str("bigInt(3).compare(2).toString()"), "1");
        assert_eq!(eval_str("bigInt(3).greater(2).toString()"), "true");
        assert_eq!(eval_str("bigInt(1).lesser(2).toString()"), "true");
        assert_eq!(eval_str("bigInt(2).equals(2).toString()"), "true");
        assert_eq!(eval_str("bigInt(0).isZero().toString()"), "true");
        assert_eq!(eval_str("bigInt(-1).isNegative().toString()"), "true");
        assert_eq!(eval_str("bigInt(4).isEven().toString()"), "true");
    }

    #[test]
    fn test_bigint_radix_parsing() {
        // The two-argument form geth's prestate_tracer_legacy.js depends on.
        assert_eq!(eval_str("bigInt('ff', 16).toString()"), "255");
        assert_eq!(eval_str("bigInt('ff', 16).toString(16)"), "ff");
        assert_eq!(eval_str("bigInt('-ff', 16).toString()"), "-255");
        assert_eq!(eval_str("bigInt('de0b6b3a7640000', 16).toString()"), "1000000000000000000");
        assert_eq!(eval_str("bigInt('1010', 2).toString()"), "10");

        // A `0x` prefix is rejected: `x` is not a digit in base 16. geth's tracers strip it
        // with `.slice(2)` before calling, so preserving this rejection keeps us aligned.
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();
        assert!(ctx.eval(Source::from_bytes(b"bigInt('0xff', 16)")).is_err());
    }

    #[test]
    fn test_bigint_256bit_precision() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        let result = to_bigint(U256::MAX, &mut ctx).unwrap();
        assert_eq!(
            result.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            "115792089237316195423570985008687907853269984665640564039457584007913129639935"
        );

        ctx.global_object().set(js_string!("m"), result, false, &mut ctx).unwrap();
        let hex = ctx.eval(Source::from_bytes(b"m.toString(16)")).unwrap();
        assert_eq!(
            hex.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );
    }

    /// The payload must stay byte-for-byte identical to the one go-ethereum embeds. A 26 KB
    /// single-line file is easy prey for format-on-save, a prettier hook or CRLF
    /// normalization, any of which would silently desynchronize us from geth.
    #[test]
    fn test_bigint_source_is_geth_verbatim() {
        let payload = BIG_INT_JS.lines().next_back().unwrap();
        assert_eq!(payload.len(), 26497);
        assert!(payload.starts_with("var bigInt=function(undefined){\"use strict\";"));
        assert!(payload.ends_with("; bigInt"));
        assert_eq!(
            alloy_primitives::keccak256(payload),
            alloy_primitives::b256!(
                "0xae5b22e16550693c719bbe50e6f126c9dde23104c20662df89d3b41801686529"
            ),
        );
    }

    fn as_length<T>(array: T) -> usize
    where
        T: Borrow<JsValue>,
    {
        let array = array.borrow();
        let array = array.as_object().unwrap();
        let array = JsUint8Array::from_object(array.clone()).unwrap();
        array.length(&mut Context::default()).unwrap()
    }

    #[test]
    fn test_to_hex() {
        let mut ctx = Context::default();
        let value = to_uint8_array_value([0xde, 0xad, 0xbe, 0xefu8], &mut ctx).unwrap();
        let result = to_hex(&JsValue::undefined(), &[value], &mut ctx).unwrap();
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "0xdeadbeef");
    }

    /// `toHex` takes bytes, not a string. go-ethereum passes `allowString=false` here: the
    /// argument is already hex text in that case, so the call cannot have been intended.
    #[test]
    fn test_to_hex_rejects_a_string() {
        let mut ctx = Context::default();
        let value = JsValue::from(js_string!("0xdeadbeef"));
        let err = to_hex(&JsValue::undefined(), &[value], &mut ctx).unwrap_err();
        assert!(err.to_string().contains("invalid buffer type"), "got: {err}");
    }

    #[test]
    fn test_to_address() {
        let mut ctx = Context::default();
        let value = JsValue::from(js_string!("0xdeadbeef"));
        let result = to_address(&JsValue::undefined(), &[value], &mut ctx).unwrap();
        assert_eq!(as_length(&result), 20);
        assert_eq!(
            result.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,173,190,239"
        );
    }

    #[test]
    fn test_to_word() {
        let mut ctx = Context::default();
        let value = JsValue::from(js_string!("0xdeadbeef"));
        let result = to_word(&JsValue::undefined(), &[value], &mut ctx).unwrap();
        assert_eq!(as_length(&result), 32);
        assert_eq!(
            result.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,173,190,239"
        );
    }

    #[test]
    fn test_to_word_digit_string() {
        let mut ctx = Context::default();
        let value = JsValue::from(js_string!("1"));
        let result = to_word(&JsValue::undefined(), &[value], &mut ctx).unwrap();
        assert_eq!(as_length(&result), 32);
        assert_eq!(
            result.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
        );
    }

    #[test]
    fn test_to_contract() {
        let mut ctx = Context::default();
        let from = JsValue::from(js_string!("0xdeadbeef"));
        let nonce = JsValue::from(0);
        let result = to_contract(&JsValue::undefined(), &[from.clone(), nonce], &mut ctx).unwrap();
        assert_eq!(as_length(&result), 20);
        let addr = to_hex(&JsValue::undefined(), &[result], &mut ctx).unwrap();
        assert_eq!(
            addr.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            "0xe8279be14e9fe2ad2d8e52e42ca96fb33a813bbe",
        );

        // without nonce
        let result = to_contract(&JsValue::undefined(), &[from], &mut ctx).unwrap();
        let addr = to_hex(&JsValue::undefined(), &[result], &mut ctx).unwrap();
        assert_eq!(
            addr.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            "0xe8279be14e9fe2ad2d8e52e42ca96fb33a813bbe",
        );
    }
    #[test]
    fn test_to_contract2() {
        let mut ctx = Context::default();
        let from = JsValue::from(js_string!("0xdeadbeef"));
        let salt = JsValue::from(js_string!("0xdead4a17"));
        let code = JsValue::from(js_string!("0xdeadbeef"));
        let result = to_contract2(&JsValue::undefined(), &[from, salt, code], &mut ctx).unwrap();
        assert_eq!(as_length(&result), 20);
        let addr = to_hex(&JsValue::undefined(), &[result], &mut ctx).unwrap();
        assert_eq!(
            addr.to_string(&mut ctx).unwrap().to_std_string().unwrap(),
            "0x8a0d8a428b30200a296dfbe693310e5d6d2c64c5"
        );
    }

    #[test]
    fn test_bigint_camelcase_alias() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        let bigint = ctx.global_object().get(js_string!("bigInt"), &mut ctx).unwrap();
        assert!(bigint.is_callable());

        let result = ctx.eval(Source::from_bytes(b"bigInt(42).toString()")).unwrap();
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "42");

        let result = ctx.eval(Source::from_bytes(b"bigInt('100').toString(16)")).unwrap();
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "64");
    }

    #[test]
    fn test_bigint_equals_shim() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        let result = ctx.eval(Source::from_bytes(b"bigInt(1).equals(bigInt(1))")).unwrap();
        assert!(result.as_boolean().unwrap());

        let result = ctx.eval(Source::from_bytes(b"bigInt(1).equals(bigInt(2))")).unwrap();
        assert!(!result.as_boolean().unwrap());

        let result = ctx.eval(Source::from_bytes(b"BigInt(0).equals(0)")).unwrap();
        assert!(result.as_boolean().unwrap());
    }

    #[test]
    fn test_bigint_to_js_number_shim() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        let result = ctx.eval(Source::from_bytes(b"bigInt(42).toJSNumber()")).unwrap();
        assert_eq!(result.to_number(&mut ctx).unwrap(), 42.0);

        let result = ctx.eval(Source::from_bytes(b"typeof bigInt(42).toJSNumber()")).unwrap();
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "number");
    }

    #[test]
    fn test_bigint_plus_minus_shim() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        let result = ctx.eval(Source::from_bytes(b"bigInt(1).plus(bigInt(2)).toString()")).unwrap();
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "3");

        let result =
            ctx.eval(Source::from_bytes(b"bigInt(10).minus(bigInt(3)).toString()")).unwrap();
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "7");
    }

    #[test]
    fn test_bigint_geth_call_tracer_pattern() {
        let mut ctx = Context::default();
        register_builtins(&mut ctx).unwrap();

        // Simulates geth's call_tracer_legacy.js patterns:
        // bigInt(gasIn - gasCost - gas).toString(16)
        let result =
            ctx.eval(Source::from_bytes(b"'0x' + bigInt(1000 - 200 - 100).toString(16)")).unwrap();
        assert_eq!(result.to_string(&mut ctx).unwrap().to_std_string().unwrap(), "0x2bc");

        // !ret.equals(0) pattern
        let result = ctx.eval(Source::from_bytes(b"var ret = bigInt(1); !ret.equals(0)")).unwrap();
        assert!(result.as_boolean().unwrap());

        let result = ctx.eval(Source::from_bytes(b"var ret = bigInt(0); !ret.equals(0)")).unwrap();
        assert!(!result.as_boolean().unwrap());
    }
}
