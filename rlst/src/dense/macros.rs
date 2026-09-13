//! A collection of macros for Rlst.

/// Implementation support for rank-dependent dynamic-array construction.
#[doc(hidden)]
pub mod dynamic {
    use std::marker::PhantomData;

    use crate::{Array, DynArray, ZeroDimArray};

    /// Preserve expected-result rank inference before call-site method dispatch.
    pub fn build<Item, const NDIM: usize, ArrayImpl>(
        shape: [usize; NDIM],
        allocate: impl FnOnce(Request<Item, NDIM>) -> Array<ArrayImpl, NDIM>,
    ) -> Array<ArrayImpl, NDIM> {
        allocate(Request::new(shape))
    }

    /// Reject aligned scalars while preserving the constructor's shape inference.
    pub fn aligned_shape<const NDIM: usize>(shape: [usize; NDIM]) -> [usize; NDIM] {
        assert!(
            NDIM > 0,
            "Aligned zero-dimensional arrays are not supported; use the unaligned macro form"
        );
        shape
    }

    /// A shape and item type awaiting call-site rank dispatch.
    pub struct Request<Item, const NDIM: usize> {
        shape: [usize; NDIM],
        item: PhantomData<Item>,
    }

    impl<Item, const NDIM: usize> Request<Item, NDIM> {
        /// Evaluate and retain the shape once.
        pub fn new(shape: [usize; NDIM]) -> Self {
            Self {
                shape,
                item: PhantomData,
            }
        }
    }

    /// Select storage using method lookup at the macro expansion site.
    pub trait Allocate {
        /// The concrete array type for this receiver.
        type Output;

        /// Construct a default-initialized array.
        fn allocate(self) -> Self::Output;
    }

    // Method lookup dereferences the macro's &&Request receiver to this fallback
    // for positive or unresolved generic ranks. No rank enumeration is needed.
    impl<Item: Copy + Default, const NDIM: usize> Allocate for &Request<Item, NDIM> {
        type Output = DynArray<Item, NDIM>;

        fn allocate(self) -> Self::Output {
            DynArray::from_shape(self.shape)
        }
    }

    // An exact receiver match wins for rank zero. Method lookup can also infer an
    // otherwise unconstrained rank as zero; later uses cannot revise that choice.
    // Keep this dispatch in the macro: a generic helper function would always
    // select the fallback before monomorphization, even when called with [].
    impl<Item: Copy + Default> Allocate for &&Request<Item, 0> {
        type Output = ZeroDimArray<Item>;

        fn allocate(self) -> Self::Output {
            ZeroDimArray::from_scalar(Item::default())
        }
    }
}

/// Implementation support for the direct tensor multiplication macros.
#[doc(hidden)]
pub mod direct {
    use crate::{Array, BaseItem, UnsafeRandomAccessMut, ZeroDimArray};

    /// Mutable output access for indexed arrays and dedicated scalar storage.
    pub trait Output<const R: usize>: BaseItem {
        /// Access an output element after validating its index.
        ///
        /// # Safety
        /// `index` must be within the output shape.
        unsafe fn output_mut(&mut self, index: [usize; R]) -> &mut Self::Item;
    }

    impl<Impl: UnsafeRandomAccessMut<R>, const R: usize> Output<R> for Array<Impl, R> {
        unsafe fn output_mut(&mut self, index: [usize; R]) -> &mut Self::Item {
            // SAFETY: the caller guarantees a valid index.
            unsafe { self.get_unchecked_mut(index) }
        }
    }

    impl<Item: Copy + Default> Output<0> for ZeroDimArray<Item> {
        unsafe fn output_mut(&mut self, _index: [usize; 0]) -> &mut Item {
            self.scalar_mut()
        }
    }

    /// Validate contraction ranks, dimensions, and size arithmetic before any writes.
    pub fn shape<const M: usize, const N: usize, const R: usize>(
        a: [usize; M],
        b: [usize; N],
    ) -> ([usize; R], usize, usize, usize) {
        assert!(
            M > 0 && N > 0,
            "direct multiplication requires positive ranks"
        );
        let rank = (M - 1)
            .checked_add(N - 1)
            .expect("direct multiplication rank overflow");
        assert_eq!(R, rank, "direct multiplication output rank mismatch");
        assert_eq!(a[M - 1], b[0], "direct multiplication contraction mismatch");

        fn size(shape: &[usize]) -> usize {
            shape.iter().fold(1usize, |size, &dim| {
                size.checked_mul(dim)
                    .expect("direct multiplication size overflow")
            })
        }

        // Check full input sizes as well as the independently flattened free axes.
        size(&a);
        size(&b);
        let rows = size(&a[..M - 1]);
        let cols = size(&b[1..]);
        rows.checked_mul(cols)
            .expect("direct multiplication output size overflow");
        let mut output = [1; R];
        output[..M - 1].copy_from_slice(&a[..M - 1]);
        output[M - 1..M + N - 2].copy_from_slice(&b[1..]);
        size(&output);
        (output, rows, a[M - 1], cols)
    }

    /// Obtain the additive identity without requiring downstream imports of `num`.
    pub fn zero<Item: num::Zero>() -> Item {
        Item::zero()
    }
}

/// Contract the last axis of `a` with the first axis of `b`, overwriting `c`.
///
/// The output shape concatenates the remaining axes of `a` and `b`. Two vectors
/// produce shape `[]`. Input ranks must be positive; there is no fixed rank limit.
/// The output rank is inferred independently from `c`.
///
/// Arguments are evaluated and borrowed once, including temporary `r()` and
/// `r_mut()` views. This macro returns `()`, does not allocate or resize, and
/// supports logical indexed access with arbitrary strides. Products are accumulated
/// in column-major `j-k-i` order without conjugation or a BLAS/GEMM requirement.
/// Items must support zero, multiplication, and addition assignment. An empty
/// contracted axis writes zeros; an empty output performs no element accesses.
///
/// # Panics
/// Panics before modifying `c` if ranks or shapes are incompatible, or checked
/// shape-size arithmetic overflows.
///
/// ```
/// let a = rlst::rlst_dynamic_array!(f64, [2, 3]);
/// let b = rlst::rlst_dynamic_array!(f64, [3, 4]);
/// let mut c = rlst::rlst_dynamic_array!(f64, [2, 4]);
/// rlst::dot_direct_into!(c.r_mut(), a.r(), b.r());
/// ```
#[macro_export]
macro_rules! dot_direct_into {
    ($c:expr, $a:expr, $b:expr $(,)?) => {{
        use $crate::dense::macros::direct::Output as _;
        let c = &mut $c;
        let a = &$a;
        let b = &$b;
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();
        let (expected, rows, inner, cols) = $crate::dense::macros::direct::shape(a_shape, b_shape);
        assert_eq!(
            c_shape, expected,
            "direct multiplication output shape mismatch"
        );

        // Copying shapes creates stack indices with the independently inferred ranks.
        let mut ai = a_shape.map(|_| 0);
        let mut bi = b_shape.map(|_| 0);
        let mut ci = c_shape.map(|_| 0);
        let last = a_shape.len() - 1;
        if rows != 0 && cols != 0 {
            for _ in 0..cols {
                for _ in 0..rows {
                    // SAFETY: validated output shape and bounded mixed-radix indices.
                    unsafe {
                        *c.output_mut(ci) = $crate::dense::macros::direct::zero();
                    }
                    // Advance axis zero first, wrapping all free A axes after
                    // the final row so the next sweep starts at zero.
                    for axis in 0..last {
                        ci[axis] += 1;
                        if ci[axis] < a_shape[axis] {
                            break;
                        }
                        ci[axis] = 0;
                    }
                }
                for k in 0..inner {
                    ai[last] = k;
                    bi[0] = k;
                    // SAFETY: free B indices are bounded by their dimensions and
                    // k is below the validated common contracted dimension.
                    let right = unsafe { b.get_value_unchecked(bi) };
                    for _ in 0..rows {
                        // SAFETY: all indices are within the validated shapes.
                        // The exclusive output borrow prevents aliasing with inputs.
                        unsafe {
                            *c.output_mut(ci) += a.get_value_unchecked(ai) * right;
                        }
                        // Carry the shared free A/output indices together.
                        for axis in 0..last {
                            ai[axis] += 1;
                            ci[axis] = ai[axis];
                            if ai[axis] < a_shape[axis] {
                                break;
                            }
                            ai[axis] = 0;
                            ci[axis] = 0;
                        }
                    }
                }
                // B's first free axis is axis one. Carry it together with
                // the corresponding trailing output axes.
                for axis in 1..b_shape.len() {
                    bi[axis] += 1;
                    ci[last + axis - 1] = bi[axis];
                    if bi[axis] < b_shape[axis] {
                        break;
                    }
                    bi[axis] = 0;
                    ci[last + axis - 1] = 0;
                }
            }
        }
    }};
}

/// Allocate a direct contraction of the last axis of `a` with the first of `b`.
///
/// Uses the same rank-dependent storage as [`rlst_dynamic_array!`](crate::rlst_dynamic_array) and
/// [`dot_direct_into!`](crate::dot_direct_into) for multiplication. Arguments are
/// evaluated and borrowed once. The output concatenates the remaining input axes,
/// with a [`ZeroDimArray`](crate::ZeroDimArray) of shape `[]` for two vectors;
/// multiplication does not conjugate either input.
///
/// Stable Rust requires the independent output rank to be inferred from context,
/// for example an explicit `DynArray<_, R>` or `ZeroDimArray<_>` annotation.
/// There is no rank cap.
///
/// # Panics
/// Panics for zero input ranks, incompatible contracted dimensions or output rank,
/// or overflowing shape-size arithmetic, before allocating the output.
///
/// ```
/// let a = rlst::rlst_dynamic_array!(f64, [2, 3, 4]);
/// let b = rlst::rlst_dynamic_array!(f64, [4, 5]);
/// let c: rlst::DynArray<f64, 3> = rlst::dot_direct!(a.r(), b.r());
/// assert_eq!(c.shape(), [2, 3, 5]);
/// ```
#[macro_export]
macro_rules! dot_direct {
    ($a:expr, $b:expr $(,)?) => {{
        let a = &$a;
        let b = &$b;
        let (shape, _, _, _) = $crate::dense::macros::direct::shape(a.shape(), b.shape());
        // Dispatch at the call site so a known zero rank selects scalar storage.
        use $crate::dense::macros::dynamic::Allocate as _;
        $crate::dense::macros::dynamic::build(shape, |request| {
            let mut output = (&&request).allocate();
            $crate::dot_direct_into!(output, a, b);
            output
        })
    }};
}

/// Dot product between compatible arrays.
///
/// It always allocates a new array for the result.
/// Call as `res = dot!(a, b, c)` to multiply the matrices
/// `a`, `b,`, and `c`. The product of arbitrary numbers of matrices
/// is supported. For more than two arguments the product is evaluated
/// from back to front.
#[macro_export]
macro_rules! dot {
    ($a:expr, $b:expr) => {
        $a.dot(&$b)
    };

    ($a:expr, $b:expr, $($c:expr),+) => {{
        dot!($a, dot!($b, $($c),+))
    }};

}

/// Create an array of `n` equidistant points with first value `a` and second value `b`.
///
/// The invocation `let arr = linspace!(dtype, a, b, n)` creates an array of n equidistant
/// points with start value `a` and final value `b`. `dtype` can be either `f32` or `f64`.
#[macro_export]
macro_rules! linspace {
    ($dtype:ty, $a:expr, $b:expr, $n:expr) => {{
        use $crate::traits::base_operations::Linspace;
        let mut arr = $crate::dense::array::DynArray::<$dtype, 1>::from_shape([$n]);
        arr.linspace($a, $b);
        arr
    }};
}

/// Create a new n-dimensional diagonal array from a list of diagonal entries.
///
/// Call with `diag!(d)` where `d` is a one-dimensional array to obtain a two-dimensional array with `d` on the
/// diagonal. For n-dimensional diagonal arrays, call with `diag!(d, NDIM)` where `NDIM` is the
/// number of dimensions and `d` is a list of diagonal entries.
#[macro_export]
macro_rules! diag {
    ($d:expr) => {
        $crate::diag!($d, 2)
    };

    ($d:expr, $ndim:literal) => {{
        use itertools::izip;

        let mut diag_array =
            $crate::dense::array::DynArray::<_, $ndim>::from_shape([$d.len(); $ndim]);
        izip!(diag_array.diag_iter_mut(), $d.iter_value()).for_each(|(diag_entry, entry)| {
            *diag_entry = entry;
        });
        diag_array
    }};
}

#[cfg(test)]
mod test {

    use rand::SeedableRng;

    use crate::{MultIntoResize, dense::array::DynArray, empty_array};

    #[test]
    fn test_dot() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut arr1 = DynArray::<f64, 2>::from_shape([2, 3]);
        let mut arr2 = DynArray::<f64, 2>::from_shape([3, 4]);
        let mut arr3 = DynArray::<f64, 2>::from_shape([4, 5]);
        arr1.fill_from_standard_normal(&mut rng);
        arr2.fill_from_standard_normal(&mut rng);
        arr3.fill_from_standard_normal(&mut rng);

        let expected = empty_array().simple_mult_into_resize(
            empty_array().simple_mult_into_resize(arr1.r(), arr2.r()),
            arr3.r(),
        );

        let actual = crate::dot!(arr1.r(), arr2.r(), arr3.r());

        crate::assert_array_relative_eq!(expected, actual, 1E-10);
    }

    #[test]
    fn test_diag() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut d = DynArray::<f64, 1>::from_shape([2]);
        d.fill_from_equally_distributed(&mut rng);

        let actual = diag!(d);
        let mut expected = DynArray::<f64, 2>::from_shape([2, 2]);

        expected[[0, 0]] = d[[0]];
        expected[[1, 1]] = d[[1]];

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }
}
