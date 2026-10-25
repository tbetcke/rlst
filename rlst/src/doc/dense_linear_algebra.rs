//! An introduction to dense linear algebra with RLST.
//!
//! - [Basic concepts](#basic-concepts)
//! - [Element access](#element-access)
//! - [Stack based array allocations](#stack-based-array-allocations)
//! - [Arrays from memory slices](#arrays-from-memory-slices)
//! - [Strides](#strides)
//! - [References and ownership](#references-and-ownership)
//! - [Array slicing](#array-slicing)
//! - [Array iterators](#array-iterators)
//! - [Operations on arrays](#operations-on-arrays)
//! - [Matrix multiplication](#matrix-multiplication)
//! - [Direct tensor multiplication](#direct-tensor-multiplication)
//!
//! # Basic concepts
//!
//! Let us start by defining a random `(3, 5, 2)` array.
//! ```
//! let mut rng = rand::rng();
//! let mut arr1 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! arr1.fill_from_equally_distributed(&mut rng);
//! ```
//!
//! Alternatively, you can also create the array with
//! ```
//! let shape = [3, 5, 2];
//! let arr = rlst::DynArray::<f64, 3>::from_shape(shape);
//! ```
//! Note that the macro [`rlst_dynamic_array`](crate::rlst_dynamic_array) relies on the compiler to automatically infer the
//! number of dimensions from the type of the provided shape. To provide the dimension explicitly
//! use [DynArray::from_shape](crate::DynArray::from_shape) directly.
//!
//! To create a memory aligned array you can give the alignment as a third parameter.
//! ```
//! let shape = [3, 5, 2];
//! // Create a cache aligned array
//! let arr = rlst::rlst_dynamic_array!(f64, shape | rlst::CACHE_ALIGNED);
//! assert_eq!(arr.data().unwrap().as_ptr().addr() % rlst::CACHE_ALIGNED, 0);
//! // Create a page aligned array
//! let arr = rlst::rlst_dynamic_array!(f64, shape | rlst::PAGE_ALIGNED);
//! assert_eq!(arr.data().unwrap().as_ptr().addr() % rlst::PAGE_ALIGNED, 0);
//! // Create an array with 64 bytes alignment
//! let arr = rlst::rlst_dynamic_array!(f64, shape | 64);
//! assert_eq!(arr.data().unwrap().as_ptr().addr() % 64, 0);
//! ```
//!
//! Any type that supports the [std::marker::Copy] and [std::default::Default] traits is supported
//! for arrays. Hence, the following is a valid construct.
//! ```
//! let mut arr = rlst::rlst_dynamic_array!([usize; 3], [3, 7]);
//! arr[[0, 1]] = [1, 2, 3];
//! ```
//! This defines an array of dimension `[3, 7]`, where each element is of type `[usize; 3]`.
//! We then set the element at position `[0, 1]` to have the value `[1, 2, 3]`.
//!
//!
//! Let us define a second `(3, 5, 2)` array and do some operations.
//! ```
//! # let mut rng = rand::rng();
//! # let mut arr1 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! let mut arr2 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! arr2.fill_from_equally_distributed(&mut rng);
//!
//! let res = 3.0 * arr1 + 5.0 * arr2;
//! ```
//! The variable `res` is of a different type from `arr1` and `arr2`. It is an addition type that
//! represents the addition of two arrays. RLST internally uses an expression template arithmetic.
//! Array operations are only executed when requested by the user. We can evaluate the expression
//! into a new array using the following command.
//! ```
//! # let mut rng = rand::rng();
//! # let mut arr1 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! # let mut arr2 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr2.fill_from_equally_distributed(&mut rng);
//! # let res = 3.0_f64 * arr1 + 5.0_f64 * arr2;
//! use rlst::EvaluateArray;
//! let output = res.eval();
//! ```
//! The required trait is [EvaluateArray](crate::EvaluateArray). There is a quirk
//! in Rust's type inference. Without explicitly annotating the constants `3.0`
//! and `5.0` as being [f64] Rust will not compile the statement as it cannot infer the
//! correct item type for the new array.
//!
//! There is an important principle behind this. Algebraic operations on arrays do not allocate new memory on
//!  the heap without the user explicitly saying so. New memory is first allocated when
//! the expression `3.0_f64 * arr1 + 5.0_f64 * arr2` is evaluated. In the background, evaluation
//! instantiates one output array of the correct size and fills it componentwise.
//!
//! The operation `3.0 * arr1 + 5.0 * arr2` takes ownership of `arr1` and `arr2`. If this is not desired,
//! create array wrappers that behave like the original arrays but internally borrow them. This is done with `arr1.r()` for non-mutable reference
//! objects and with `arr1.r_mut()` for mutable reference objects.
//! ```
//! # let mut rng = rand::rng();
//! # let mut arr1 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! # let mut arr2 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr2.fill_from_equally_distributed(&mut rng);
//! let output = 3.0_f64 * arr1.r() + 5.0_f64 * arr2.r();
//! assert_eq!(output.shape(), arr1.shape());
//! ```
//!
//! # Element access
//!
//! To get the value of an element of an array at an arbitrary position the
//! method [get_value](crate::RandomAccessByValue::get_value)
//! is provided. [get_value](crate::RandomAccessByValue::get_value) does not return a reference but an actual value. It can //
//! therefore always be used even when the array represents an expression rather than values in memory. If it is possible
//! to have a reference to an element (i.e. the array is associated
//! with elements in memory) then direct index notation is also possible. Both are shown below.
//!
//! ```
//! let mut arr = rlst::rlst_dynamic_array!(f64, [3, 4, 2]);
//! arr.fill_from_seed_equally_distributed(0);
//! assert_eq!(arr.get_value([2, 3, 1]).unwrap(), arr[[2, 3, 1]]);
//! ```
//! The index notation `arr[[2, 3, 1]]` is just short for `arr.get([2, 3, 1]).unwrap()` and returns a reference.
//! The index notation `arr[[2, 3, 1]]` can also be used for a mutable reference, or equivalently use
//! `arr.get_mut([2, 3, 1]).unwrap()`. If the index does not exist the index notation returns an assertion error
//! while `get` and `get_mut` return `None`.
//!
//! # Stack based Array allocations
//!
//! RLST arrays can be allocated either on the heap or on the stack. Heap based allocations can be done dynamically at runtime.
//! Stack based allocations happens at compile time and the exact size of the array must be known. Stack based
//! arrays are of advantage for small fixed sizes when the overhead of allocating memory at runtime can impact performance.
//!
//! For heap-based allocation we have already seen the macro [rlst_dynamic_array](crate::rlst_dynamic_array) and the corresponding method [DynArray::from_shape](crate::DynArray::from_shape).
//! For stack based allocation the number of elements needs to be known at compile time. To make the allocation straight
//! forward the macro [rlst_static_array](crate::rlst_static_array). is provided. This works in the same way as
//! `rlst_dynamic_array`. However, the dimensions must be given explicitly as compile-time integers. The proc macro
//! then multiplies the dimensions and instantiates a compile time fixed size Rust array to hold the data. The following
//! shows an example.
//!
//! ```
//! # use rlst::EvaluateArray;
//! # let mut rng = rand::rng();
//! let mut arr1 = rlst::rlst_static_array!(f64, [1, 3, 2]);
//! let mut arr2 = rlst::rlst_static_array!(f64, [1, 3, 2]);
//! arr1.fill_from_equally_distributed(&mut rng);
//! arr2.fill_from_equally_distributed(&mut rng);
//! let output = (arr1 + arr2).eval();
//! ```
//! The associated static type can also be named by listing literal dimensions without brackets:
//! ```
//! type Matrix = rlst::rlst_static_type!(f64, 2, 3);
//! let matrix: Matrix = rlst::rlst_static_array!(f64, [2, 3]);
//! assert_eq!(matrix.shape(), [2, 3]);
//! ```
//! Both static macros require literal dimensions because the storage length is computed during
//! macro expansion.
//!
//! What is interesting about this code is that all the required memory is allocated at compile time, including
//! the array `output`. The `eval` method sees that all arrays of the expression are stack allocated and
//! hence uses a compile time allocation on the stack for the output array. The decision about whether a
//! runtime or compile time allocation is performed, is decided as follows:
//! - If all arrays in the expression are heap based runtime allocated arrays, then `eval` creates a runtime
//!   array.
//! - If at least one of the arrays is a stack allocated compile time array then a stack-based allocation for
//!   `output` at compile time is performed.
//!
//! The rationale is that if at least one array in the expression is stack allocated, size information is
//! available at compile time so can be used to create the output. If every array in the expression is generated
//! at runtime then no compile time size information is available.
//!
//! The mechanism that makes this selection possible is fully implemented through traits and type mechanics.
//! So there is no runtime overhead.
//!
//! ## Arrays from memory slices
//!
//! One can create an array from a given memory slice. The following example shows how to do this.
//! ```
//! let myvec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let arr = rlst::SliceArray::<f64, 2>::from_shape(myvec.as_slice(), [2, 3]);
//! ```
//! For mutable slices the method [SliceArrayMut::from_shape](crate::SliceArrayMut::from_shape)
//! updates the caller's storage without allocating:
//! ```
//! let mut storage = [1, 2, 3, 4];
//! {
//!     let mut arr = rlst::SliceArrayMut::<i32, 2>::from_shape(&mut storage, [2, 2]);
//!     arr[[1, 0]] = 20;
//! }
//! assert_eq!(storage, [1, 20, 3, 4]);
//! ```
//!
//! # Strides
//!
//! By default RLST uses a column major ordering, that is the left-most dimension is contiguous in memory.
//! This holds for memory storage order and also iterators over arrays. Any newly allocated array is automatically
//! using column-major order. However, other strides are possible. An array with a different stride can be initialized
//! with  [StridedDynArray::from_shape_and_stride](crate::StridedDynArray::from_shape_and_stride) or [StridedDynArray::row_major](crate::StridedDynArray::row_major).
//! to specifically create an array in row-major format. Sometimes one needs to convert an array from column-major to row-major format.
//! This can be done with [EvaluateRowMajorArray::eval_row_major](crate::EvaluateRowMajorArray::eval_row_major), which is implemented for all array types.
//!
//! All arrays with custom strides still iterate through in column-major order. This is to ensure that componentwise binary operations on two arrays are
//! possible for arbitrary stride arrays. The consequence is that iteration can be slower if an array has a custom stride.
//!
//! The following example contrasts logical iteration with physical storage order:
//! ```
//! use rlst::{MemoryLayout, StridedDynArray};
//!
//! let mut row_major = StridedDynArray::<i32, 2>::row_major([2, 3]);
//! row_major.fill_from_iter(0..6); // logical column-major order
//! assert_eq!(row_major.stride(), [3, 1]);
//! assert_eq!(row_major.memory_layout(), MemoryLayout::RowMajor);
//! assert!(row_major.is_contiguous());
//! assert_eq!(row_major.iter_value().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
//! assert_eq!(row_major.data().unwrap(), &[0, 2, 4, 1, 3, 5]);
//!
//! let custom = StridedDynArray::<i32, 2>::from_shape_and_stride([2, 3], [1, 3]);
//! assert_eq!(custom.stride(), [1, 3]);
//! assert_eq!(custom.memory_layout(), MemoryLayout::Unknown);
//! assert!(!custom.is_contiguous());
//! ```
//!
//! Explicit strides must map each logical element to a distinct storage location. Constructors
//! panic for overlapping layouts such as a zero stride on an axis with extent greater than one.
//!
//! [`data`](crate::Array::data) returns a slice that holds exactly the array's elements, starting
//! with the first one, and is therefore `None` for most views (see "Array slicing" below).
//! [`storage`](crate::Array::storage) and [`storage_mut`](crate::Array::storage_mut) also work
//! for views: they return the backing storage and the offset of the view's first element in
//! it, so that the element at multi-index `i` is `storage[offset + Σ i[k] · stride[k]]`. For a
//! view the slice is the parent's whole storage and may hold elements of other views before,
//! between and after the view's own. Use `data()` when the elements must be one contiguous
//! slice, and `storage()` to hand a strided view to code that addresses memory through
//! strides, such as an FFT library.
//! ```
//! let mut arr = rlst::rlst_dynamic_array!(i32, [4, 3]);
//! arr.fill_from_iter(0..12);
//!
//! let view = arr.r().into_subview([1, 1], [2, 2]);
//! let (storage, offset) = view.storage().unwrap();
//! assert_eq!((storage.len(), offset), (12, 5));
//! let stride = view.stride();
//! assert_eq!(storage[offset + stride[1]], view[[0, 1]]);
//! ```
//!
//!
//! # References and ownership
//!
//! Many operations in RLST take ownership of their arguments, e.g. in the expression `let out = a + b` for arrays `a` and `b` the object
//! `out` takes ownership of `a` and `b`. To avoid this one can instead say `let out = a.r() + b.r()`. The method [r](crate::dense::array::Array::r)
//! returns an array that is implemented through an [ArrayRef](crate::dense::array::reference::ArrayRef) struct. An `ArrayRef` is a simple container
//! that holds a Rust reference to an actual array. This allows us to hand over references to arrays to functions that expect to take ownership.
//! For mutable arrays a corresponding function exists in the form of [r_mut](crate::dense::array::Array::r_mut). These reference
//! arrays forward the capabilities of the borrowed array and can be used transparently in expressions and views.
//! The borrow remains active for the lifetime of the wrapper.
//!
//! # Array slicing
//!
//! RLST allows the slicing of arrays across a given dimension. Given the following array.
//! ```
//! let arr = rlst::rlst_dynamic_array!(f64, [5, 3, 2]);
//! ```
//! we can remove axis `1` and fix its coordinate at index `0` as follows. The arguments to
//! `slice` are `(axis, index_on_that_axis)`.
//! ```
//! # let arr = rlst::rlst_dynamic_array!(f64, [5, 3, 2]);
//! let slice = arr.r().slice::<2>(1, 0);
//! ```
//! This creates a two dimensional array of dimension `(5, 2)`. Any index `[a, b]` into this array
//! is mapped to an index `[a, 0, b]` of the original array. A slice is mutable if the original array was mutable.
//! A frequent example is access to a single column or row in a given matrix.
//! ```
//! let arr = rlst::rlst_dynamic_array!(f64, [4, 6]);
//! // This gives the third column
//! let slice = arr.r().slice::<1>(1, 2);
//! // This gives the fourth row
//! let slice = arr.r().slice::<1>(0, 3);
//! ```
//! The annotation with the dimension parameter is necessary if the compiler cannot automatically infer from
//! subsequent operations what the correct number of dimensions after slicing is.
//!
//! Slicing a vector gives a view of rank zero, a single element indexed by `[]`, and
//! [`insert_empty_axis`](crate::Array::insert_empty_axis) turns an array of rank zero into a
//! vector of length one:
//! ```
//! use rlst::dense::array::empty_axis::AxisPosition;
//!
//! let mut vec = rlst::rlst_dynamic_array!(f64, [3]);
//! vec.r_mut().slice::<0>(0, 1)[[]] = 2.0;
//! assert_eq!(vec[[1]], 2.0);
//!
//! let element = vec.r().slice::<0>(0, 1);
//! let one = element.insert_empty_axis::<1>(AxisPosition::Back);
//! assert_eq!(one.shape(), [1]);
//! assert_eq!(one[[0]], 2.0);
//! ```
//!
//! [`Array::multislice`](crate::Array::multislice) fixes several axes at once, while
//! [`Array::into_subview`](crate::Array::into_subview) keeps every axis and selects a rectangular
//! region. Views consume their array operand, so use `r()` or `r_mut()` to retain ownership:
//! ```
//! let mut arr = rlst::rlst_dynamic_array!(i32, [3, 4, 2]);
//! arr.fill_from_iter(0..24);
//!
//! {
//!     // Fix axes 0 and 2, leaving axis 1 as a one-dimensional view.
//!     let mut line = arr.r_mut().multislice::<2, 1>([0, 2], [1, 0]);
//!     assert_eq!(line.shape(), [4]);
//!     line[[2]] = 99;
//! }
//! assert_eq!(arr[[1, 2, 0]], 99);
//!
//! let view = arr.r().into_subview([1, 1, 0], [2, 2, 2]);
//! assert_eq!(view.shape(), [2, 2, 2]);
//! ```
//!
//! `transpose()` reverses axis order, while `permute_axes(permutation)` applies an explicit
//! permutation. `reverse_axis(axis)` presents indices on one axis in reverse order; these are
//! lazy views and do not rearrange the backing storage.
//!
//! # Array iterators
//!
//! The default iterator returns elements of an array in column-major order independent of the underlying stride.
//! The following example demonstrates the default iterator.
//! ```
//! let mut arr = rlst::rlst_dynamic_array!(f64, [2, 3]);
//! for elem in arr.iter_mut() {
//!     *elem = 2.0;
//! }
//!
//! ```
//! The possible iterators are `iter_value` for copied values, `iter_ref` for shared references,
//! and `iter_mut` for disjoint mutable references. Lazy expressions such as `a + b` support
//! `iter_value`; reference iterators require actual reference-backed storage. In every case the
//! logical order is column-major, independent of physical stride.
//!
//! It is possible to obtain the multi-index together with the iterator as is shown in the following example.
//! ```
//! # let mut arr = rlst::rlst_dynamic_array!(f64, [2, 3]);
//! # arr.fill_from_seed_equally_distributed(0);
//! use rlst::AsMultiIndex;
//! let shape = arr.shape();
//! for (multi_index, elem) in arr.iter_value().enumerate().multi_index(shape) {
//!     assert_eq!(elem, arr[multi_index]);
//! }
//! ```
//! Diagonal iteration visits `[i, i, ...]` for `i` up to the smallest axis extent. Row and
//! column iterators are available for matrices. In addition to the above iterators:
//! - [col_iter](crate::dense::array::Array::col_iter) - for 2-dimensional arrays to iterate through the columns.
//! - [row_iter](crate::dense::array::Array::row_iter) - for 2-dimensional arrays to iterate through the rows.
//! - [diag_iter_value](crate::dense::array::Array::diag_iter_value) - for n-dimensional arrays to iterate through the diagonal by value.
//! - [diag_iter_ref](crate::dense::array::Array::diag_iter_ref) - for n-dimensional arrays to iterate through the diagonal by reference.
//! - [diag_iter_mut](crate::dense::array::Array::diag_iter_mut) - for n-dimensional arrays to iterate through the diagonal mutably.
//!
//!
//! # Operations on arrays
//!
//! RLST uses a system of static lazy evaluation for array operations. Consider the following example.
//! ```
//! # let mut rng = rand::rng();
//! let mut arr = rlst::rlst_dynamic_array!(f64, [2, 3]);
//! arr.fill_from_equally_distributed(&mut rng);
//! let res = 5.0 * arr.r();
//! ```
//! The array `res` is not connected with a memory region. It internally takes ownership
//! of a reference into `arr`. The scalar multiplication with `5.0` is performed whenever an element of `res` is requested,
//! either by an iterator or by direct random access. Hence, the operation `let res = 5.0 * arr.r()` has no cost itself.
//! But calling `res.get_value([0, 0])` would multiply the first element of `arr` by `5.0` and return the result of that
//! operation. Calling `let out = res.eval()` allocates an output container and evaluates every element.
//!
//! RLST supports all the usual algebraic unary and binary operations on arrays. Results are stored as expressions and only evaluated
//! on element access or evaluation into a new array. In addition to standard algebraic operations the following special functions are
//! implemented: `conj, abs, square, abs_square, sqrt, exp, ln, recip, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh`.
//!
//! Hence, we can form an expression of the form `let out = (5.0 * a.sqrt() + b.exp()).eval()`. To evaluate the expression only a single iteration through
//! the arrays `a` and `b` is performed.
//!
//! Below is a list of further operations supported through the expression system in RLST.
//! - [cast](crate::dense::array::Array::cast): Cast from one type to another using [num::cast::cast].
//! - [coerce_dim](crate::dense::array::Array::coerce_dim): Coerce the dimension of an array from a generic parameter to a concrete integer.
//! - [mul_add](crate::traits::MulAdd): Multiply an array with a scalar and add to another array. This uses a componentwise `mul_add` that
//!   is usually converted into a single `mul_add` cpu instruction instead of a separate multiplication and addition.
//! - [reverse_axis](crate::dense::array::Array::reverse_axis): Reverse the elements in an array along a single axis.
//! - [transpose](crate::dense::array::Array::transpose): Reverse the order of the axes of an array.
//! - [with_eval_type](crate::dense::array::Array::with_container_type): Force the use of either a dynamic or static container for array evaluation.
//! - [unary_op](crate::dense::array::Array::unary_op): Apply an arbitrary unary function componentwise to an array.
//! - [into_type](crate::dense::array::Array::into_type): Use [std::convert::Into] to convert into another type.
//!
//! # Matrix multiplication
//!
//! [`dot_blas!`](crate::dot_blas) uses RLST's BLAS interface to perform matrix multiplication.
//! For general dense matrices, `dot_blas!` is significantly faster than
//! [`dot!`](crate::dot) and is recommended.
//! To use matrix products make sure to have a BLAS/Lapack provider as explained [here](crate::doc::getting_started). Furthermore,
//! BLAS requires the data stored in memory using column-major order. Hence, array expressions are not supported and need to first be
//! evaluated.
//!
//! ```
//! # extern crate blas_src;
//! let arr1 = rlst::rlst_dynamic_array!(f64, [4, 5]);
//! let arr2 = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! let res = rlst::dot_blas!(arr1.r(), arr2.r());
//! ```
//! This multiplies the matrices `arr1` and `arr2` into a new matrix `res`. The macro [`dot_blas!`](crate::dot_blas) initializes a new array and multiplies
//! `arr1` and `arr2` into it. `dot_blas!` supports the multiplication of multiple arrays. Hence, an expression of the form `dot_blas!(a1, a2, a3, a4)`
//! is possible. Arrays are multiplied with `dot_blas!` from right to left. For combined array multiplication/addition of the form `Y -> alpha * A B + beta * Y`
//! the [apply](crate::AsMatrixApply::apply) is implemented for all array types that support BLAS operations.
//!
//! # Direct tensor multiplication
//!
//! The shared [`dot!`](crate::dot) macro uses direct tensor contraction for dense
//! arrays, while preserving the existing multiplication behavior for
//! [sparse matrices](crate::doc::sparse_matrices) and
//! [abstract operators](crate::doc::abstract_linear_algebra).
//!
//! For dense arrays, [`dot!`](crate::dot) and [`dot_into!`](crate::dot_into)
//! contract the **last axis of A with the first axis of B**, without requiring BLAS.
//! For shapes `[a0, ..., aN, k]` and `[k, b0, ..., bM]`, the result has shape
//! `[a0, ..., aN, b0, ..., bM]`: the uncontracted axes of A followed by those of B.
//! Each result element is the sum over `k` of the corresponding products.
//! Complex values are multiplied without conjugating either operand.
//!
//! | A shape | B shape | Result shape |
//! | --- | --- | --- |
//! | `[k]` | `[k]` | `[]` |
//! | `[m, k]` | `[k]` | `[m]` |
//! | `[k]` | `[k, n]` | `[n]` |
//! | `[m, k]` | `[k, n]` | `[m, n]` |
//! | `[p, q, k]` | `[k, r, s]` | `[p, q, r, s]` |
//!
//! A vector-vector product returns a [`ZeroDimArray`](crate::ZeroDimArray), the
//! array `DynArray<_, 0>` of rank zero; read its scalar value with `result[[]]` or
//! `*result.scalar()`. Both inputs must have at least one axis.
//! Arbitrary positive input ranks are supported. Supply an output type annotation
//! when Rust cannot infer the result rank:
//!
//! ```
//! use rlst::DynArray;
//! let mut a = DynArray::<f64, 3>::from_shape([2, 3, 4]);
//! let mut b = DynArray::<f64, 3>::from_shape([4, 5, 2]);
//! a.fill_with_value(2.0);
//! b.fill_with_value(3.0);
//! let result: DynArray<f64, 4> = rlst::dot!(a, b);
//! assert_eq!(result.shape(), [2, 3, 5, 2]);
//! assert_eq!(result[[1, 2, 4, 1]], 24.0);
//! // The macros borrow named operands, so a and b remain usable.
//! assert_eq!(a.shape(), [2, 3, 4]);
//! ```
//!
//! ```
//! let mut a = rlst::rlst_dynamic_array!(f64, [3]);
//! a.fill_with_value(2.0);
//! let result: rlst::ZeroDimArray<f64> = rlst::dot!(a.r(), a.r());
//! assert_eq!(result.shape(), [0usize; 0]);
//! assert_eq!(*result.scalar(), 12.0);
//! ```
//!
//! For dense arrays, `dot!(a, b)` allocates a `DynArray` as
//! [`rlst_dynamic_array!`](crate::rlst_dynamic_array) does, then delegates to
//! `dot_into!(c, a, b)`. The latter returns `()`, overwrites every output
//! element, and neither allocates nor resizes the destination. Its shape and rank
//! must already match the result. A contraction of length zero writes zeros; an
//! output with any zero-length axis has no elements to write.
//!
//! ```
//! let mut a = rlst::rlst_static_array!(f64, [2, 3]);
//! let mut b = rlst::rlst_static_array!(f64, [3, 4]);
//! a.fill_with_value(2.0);
//! b.fill_with_value(3.0);
//! let mut storage = rlst::rlst_dynamic_array!(f64, [4, 6]);
//! storage.fill_with_value(-1.0);
//! rlst::dot_into!(storage.r_mut().into_subview([1, 1], [2, 4]), a, b);
//! assert_eq!(storage[[1, 1]], 18.0);
//! assert_eq!(storage[[0, 0]], -1.0);
//! ```
//!
//! Inputs can be normal or static arrays, strided or transposed views, or lazy
//! expressions supporting element access. Destinations must support mutable
//! element access, including mutable subviews. The direct kernel traverses logical
//! column-major coordinates while respecting each operand's layout.
//! Each macro argument expression is evaluated once and borrowed for the call.
//! When constructing views or lazy expressions, use `r()` and `r_mut()` to retain
//! ownership of their underlying arrays, as in the examples above.
//!
//! A mismatched contraction length, output rank, or output shape panics before
//! any output element is changed. Rank-zero inputs are unsupported and panic.
//! The allocating form also validates the inferred or annotated result rank;
//! an incorrect annotation is an error at runtime, not a reshape operation.
//!
// ! # Pivoted QR Decomposition
// !
// ! The pivted QR decomposition of a matrix `A` is given as `AP=QR`, where `P` is a permutation matrix, `R` is upper triangular, and
// ! `Q` has orthogonal columns. For details see the documentation of [MatrixQrDecomposition](crate::dense::linalg::qr::MatrixQr).
// !
// ! The pivoted QR decomposition of a matrix can be computed as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! let mut rand = rand::rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [8, 5]);
// ! let mut r_mat = rlst_dynamic_array2!(f64, [5, 5]);
// ! let mut p_mat = rlst_dynamic_array2!(f64, [5, 5]);
// ! let mut q_mat = rlst_dynamic_array2!(f64, [8, 5]);
//  ! arr.fill_from_equally_distributed(&mut rand);
// ! let qr = arr.into_qr_alloc().expect("QR Decomposition failed");
// ! qr.get_r(r_mat.r_mut());
// ! qr.get_q_alloc(q_mat.r_mut());
// ! qr.get_p(p_mat.r_mut());
// ! ````
// ! The content of `arr` is overwritten with the QR decomposition. The method [get_q_alloc](crate::dense::linalg::qr::QrDecomposition::get_q_alloc)
// ! needs to allocate additional temporary memory on the heap. This is why it is annoted with `_alloc`.
// !
// ! # Singular value decomposition
// !
// ! To compute the singular values of a two-dimensional array `arr` use
// ! ```
// ! # use rlst::prelude::*;
// ! let mut rand = rand::rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [8, 5]);
// ! let mut singvals = rlst_dynamic_array1!(f64, [5]);
// ! arr.into_singular_values_alloc(singvals.data_mut()).unwrap();
// ! ```
// ! This computes the singular values of `arr` into `singvals`. The method [into_singular_values_alloc](crate::MatrixSvd::into_singular_values_alloc)
// ! needs to allocate temporary memory on the heap. This is why it has the ending `_alloc`. To compute the whole
// ! singular value decomposition use the method [into_svd_alloc](crate::MatrixSvd::into_svd_alloc).
// ! ```
// ! # use rlst::prelude::*;
// ! let mut rand = rand::rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [8, 5]);
// ! let mut u = rlst_dynamic_array2!(f64, [8, 5]);
// ! let mut vt = rlst_dynamic_array2!(f64, [5, 5]);
// ! let mut sigma = rlst_dynamic_array1!(f64, [5]);
// ! arr.into_svd_alloc(u.r_mut(), vt.r_mut(), sigma.data_mut(), SvdMode::Reduced).unwrap();
// ! ```
// ! To compute the full SVD use the parameter [SvdMode::Full](crate::SvdMode::Full).
// !
// ! # Other vector functions
// !
// ! The following other functions for arrays with a single dimension (i.e. vectors) are provided.
// ! - [inner](crate::Array::inner): Compute the inner product with another vector.
// ! - [norm_1](crate::Array::norm_1): Compute the 1-norm of a vector.
// ! - [norm_2](crate::Array::norm_2): Compute the 2-norm of a vector.
// ! - [norm_inf](crate::Array::norm_inf): Compute the inf-norm of a vector.
// ! - [len](crate::Array::len): Convenience function to return the length of a vector.
// ! - [cross](crate::Array::cross): Compute the cross product with another vector.
// !
// ! # Other matrix functions
// !
// ! The following other functions for arrays with two dimensions (i.e. matrices) are provided.
// ! - [norm_1](crate::Array::norm_1): Compute the 1-norm of a matrix.
// ! - [norm_2_alloc](crate::Array::norm_2): Compute the 2-norm of a matrix. This method allocates temporary memory on the heap.
// ! - [norm_fro](crate::Array::norm_fro): Compute the Frobenius norm of a matrix.
// ! - [norm_inf](crate::Array::norm_inf): Compute the inf-norm of a matrix.
// ! - [into_det](crate::Array::into_det): Compute the determinant of a matrix.
// ! - [into_pseudo_inverse_alloc](crate::MatrixPseudoInverse::into_pseudo_inverse_alloc) and [into_pseudo_inverse_resize_alloc](crate::MatrixPseudoInverse::into_pseudo_inverse_resize_alloc) compute the pseudoinverse of a matrix.
