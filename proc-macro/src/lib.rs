//! Proc Macros for the rlst library
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

use proc_macro::TokenStream;

mod dense;
mod tracing;

/// Allocate a dynamic column-major RLST array.
///
/// Syntax is `rlst_dynamic_array!(ItemType, shape_expression)`. An optional alignment expression
/// follows `|`, as in `rlst_dynamic_array!(f64, [2, 3] | 64)`. The number of dimensions is
/// inferred from the shape array; elements are initialized with `Default`.
///
/// A shape known to have rank zero, such as `[]` or a variable of type `[usize; 0]`,
/// creates a `ZeroDimArray<ItemType>` containing one scalar. Use `scalar()`,
/// `scalar_mut()`, and `into_scalar()` to access it; indexed access is unavailable.
/// Positive ranks produce the existing heap-backed `DynArray` type.
///
/// Dispatch takes place where the macro expands. Inside a function generic over
/// the rank, the result remains `DynArray<ItemType, NDIM>` and rank zero is rejected
/// at runtime. Generic reduction code can construct a scalar result explicitly
/// with `ZeroDimArray::from_scalar(value)`.
///
/// For the unaligned form, determine the shape rank through the shape expression
/// or an explicit expected result type. If neither constrains the rank during
/// dispatch, it is inferred as zero; subsequent uses cannot change that choice.
/// For example, annotate `let shape: [usize; 2] = Default::default()` rather than
/// relying on a later use of the resulting array to infer the shape rank.
/// `DynArray::from_shape` retains the original inference behavior for positive ranks.
///
/// # Panics
///
/// Panics if an alignment is requested for rank zero, or if an unresolved generic
/// rank is instantiated with zero.
#[proc_macro]
pub fn rlst_dynamic_array(items: TokenStream) -> TokenStream {
    dense::rlst_dynamic_array_impl(items)
}

/// Create a stack-backed column-major RLST array.
///
/// Syntax is `rlst_static_array!(ItemType, [literal_dimension, ...])`. Dimensions must be integer
/// literals because their product determines the backing Rust array type at compile time.
#[proc_macro]
pub fn rlst_static_array(items: TokenStream) -> TokenStream {
    dense::rlst_static_array_impl(items)
}

/// Expand to the type of a stack-backed RLST array.
///
/// Syntax is `rlst_static_type!(ItemType, literal_dimension, ...)`, without brackets around the
/// dimensions. At least one dimension is required.
#[proc_macro]
pub fn rlst_static_type(items: TokenStream) -> TokenStream {
    dense::rlst_static_type_impl(items)
}

/// Instrument a function with entry, exit, and elapsed-time log messages.
///
/// The attribute takes no arguments. Its expansion calls `rlst::rlst_print!`, which emits log
/// records when RLST's `enable_tracing` feature is enabled and otherwise compiles to a no-op.
#[proc_macro_attribute]
pub fn measure_duration(attr: TokenStream, item: TokenStream) -> TokenStream {
    tracing::measure_duration_impl(attr, item)
}
