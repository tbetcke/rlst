//! Type-erased frame storage selected by a space.
//!
//! [`InnerProductSpace::frame`] returns the same type for
//! every space, so the storage backend is erased behind [`SpaceFrame`]. A space
//! selects its backend by overriding that method; generic code therefore reaches
//! the selected backend without Rust specialization. The cost is one virtual
//! call per frame operation, which is small next to the work each one performs.

use crate::operator::element::Element;
use crate::{
    Array, DynArray, FrameApply, FrameImpl, FrameQr, FrameQrDecomposition, FrameQrOptions,
    InnerProductSpace, LinearSpace, RlstResult, RlstScalar, Shape, SliceArray,
    UnsafeRandom1DAccessByValue,
};

/// The QR decomposition an erased frame returns, with Q erased as well.
pub type ErasedQrDecomposition<'a, Space> =
    FrameQrDecomposition<SpaceFrame<'a, Space>, <Space as LinearSpace>::F>;

/// Object-safe frame operations behind [`SpaceFrame`].
///
/// [`FrameApply`] is generic over the coefficient array and [`FrameQr`] requires
/// `Sized`, so neither is object safe. This trait restates both with packed
/// coefficients and an erased Q, and is blanket-implemented for every backend
/// supporting them. Implementations never need to write it themselves.
pub trait DynFrame<'a>: FrameImpl<'a>
where
    <Self::Space as LinearSpace>::F: RlstScalar,
{
    /// Form the linear combination of columns with packed coefficients.
    fn apply_dyn(
        &self,
        coefficients: SliceArray<'_, <Self::Space as LinearSpace>::F, 1>,
    ) -> RlstResult<Element<'a, Self::Space>>;

    /// Compute `output = alpha * self * coefficients + beta * output`.
    fn apply_into_dyn(
        &self,
        alpha: <Self::Space as LinearSpace>::F,
        coefficients: SliceArray<'_, <Self::Space as LinearSpace>::F, 1>,
        beta: <Self::Space as LinearSpace>::F,
        output: &mut Element<Self::Space>,
    ) -> RlstResult<()>;

    /// Compute pivoted QR, returning Q in the same erased form.
    fn qr_pivoted_dyn(
        &self,
        options: FrameQrOptions<<<Self::Space as LinearSpace>::F as RlstScalar>::Real>,
    ) -> RlstResult<ErasedQrDecomposition<'a, Self::Space>>;
}

impl<'a, T> DynFrame<'a> for T
where
    T: FrameApply<'a> + FrameQr<'a> + 'a,
    T::Space: InnerProductSpace,
    <T::Space as LinearSpace>::F: RlstScalar,
{
    fn apply_dyn(
        &self,
        coefficients: SliceArray<'_, <Self::Space as LinearSpace>::F, 1>,
    ) -> RlstResult<Element<'a, Self::Space>> {
        self.apply(&coefficients)
    }

    fn apply_into_dyn(
        &self,
        alpha: <Self::Space as LinearSpace>::F,
        coefficients: SliceArray<'_, <Self::Space as LinearSpace>::F, 1>,
        beta: <Self::Space as LinearSpace>::F,
        output: &mut Element<Self::Space>,
    ) -> RlstResult<()> {
        self.apply_into(alpha, &coefficients, beta, output)
    }

    fn qr_pivoted_dyn(
        &self,
        options: FrameQrOptions<<<Self::Space as LinearSpace>::F as RlstScalar>::Real>,
    ) -> RlstResult<ErasedQrDecomposition<'a, Self::Space>> {
        let FrameQrDecomposition { q, r, permutation } = self.qr_pivoted(options)?;
        Ok(FrameQrDecomposition {
            q: SpaceFrame::new(q),
            r,
            permutation,
        })
    }
}

/// A frame implementation whose storage backend is erased.
///
/// This is the implementation type returned by
/// [`InnerProductSpace::frame`]. It forwards every
/// operation to the backend the space selected, so overrides such as BLAS
/// application and LAPACK QR are preserved. Backend-specific inherent methods
/// are not: construct the backend directly, or use
/// [`ArrayVectorSpace::array_frame`](crate::operator::interface::array_vector_space::ArrayVectorSpace::array_frame),
/// to reach zero-copy matrix and column views.
pub struct SpaceFrame<'a, Space: LinearSpace + 'a>
where
    <Space as LinearSpace>::F: RlstScalar,
{
    imp: Box<dyn DynFrame<'a, Space = Space> + 'a>,
}

impl<'a, Space: LinearSpace + 'a> SpaceFrame<'a, Space>
where
    <Space as LinearSpace>::F: RlstScalar,
{
    /// Erase a frame implementation without copying its storage.
    pub fn new<T: DynFrame<'a, Space = Space> + 'a>(imp: T) -> Self {
        Self { imp: Box::new(imp) }
    }
}

impl<'a, Space: LinearSpace + 'a> FrameImpl<'a> for SpaceFrame<'a, Space>
where
    <Space as LinearSpace>::F: RlstScalar,
{
    type Space = Space;

    fn space(&self) -> &'a Space {
        self.imp.space()
    }
    fn element(&self, index: usize) -> Option<Element<'a, Space>> {
        self.imp.element(index)
    }
    fn set_element(&mut self, index: usize, elem: &Element<Space>) -> RlstResult<()> {
        self.imp.set_element(index, elem)
    }
    fn len(&self) -> usize {
        self.imp.len()
    }
    fn is_empty(&self) -> bool {
        self.imp.is_empty()
    }
    fn capacity(&self) -> usize {
        self.imp.capacity()
    }
    fn reserve(&mut self, additional: usize) {
        self.imp.reserve(additional);
    }
    fn push(&mut self, elem: &Element<Space>) -> RlstResult<()> {
        self.imp.push(elem)
    }
    fn pop(&mut self) -> Option<Element<'a, Space>> {
        self.imp.pop()
    }
    fn clear(&mut self) {
        self.imp.clear();
    }
}

/// Pack coefficients so that they can cross the object-safe boundary.
///
/// Only the `O(n)` coefficients are copied, never the `O(m*n)` frame data, and
/// packing is what the matrix-backed backend does with strided or lazy inputs
/// anyway.
fn packed<A, Item: RlstScalar>(coefficients: &Array<A, 1>) -> DynArray<Item, 1>
where
    A: Shape<1> + UnsafeRandom1DAccessByValue<Item = Item>,
{
    DynArray::new_from(coefficients)
}

impl<'a, Space: LinearSpace + 'a> FrameApply<'a> for SpaceFrame<'a, Space>
where
    <Space as LinearSpace>::F: RlstScalar,
{
    fn apply<A>(&self, coefficients: &Array<A, 1>) -> RlstResult<Element<'a, Space>>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = <Space as LinearSpace>::F>,
    {
        let coefficients = packed(coefficients);
        let shape = coefficients.shape();
        self.imp
            .apply_dyn(SliceArray::from_shape(coefficients.data().unwrap(), shape))
    }

    fn apply_into<A>(
        &self,
        alpha: <Space as LinearSpace>::F,
        coefficients: &Array<A, 1>,
        beta: <Space as LinearSpace>::F,
        output: &mut Element<Space>,
    ) -> RlstResult<()>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = <Space as LinearSpace>::F>,
        <Space as LinearSpace>::F: num::Zero,
    {
        let coefficients = packed(coefficients);
        let shape = coefficients.shape();
        self.imp.apply_into_dyn(
            alpha,
            SliceArray::from_shape(coefficients.data().unwrap(), shape),
            beta,
            output,
        )
    }
}

impl<'a, Space: InnerProductSpace + 'a> FrameQr<'a> for SpaceFrame<'a, Space>
where
    <Space as LinearSpace>::F: RlstScalar,
{
    fn qr_pivoted(
        &self,
        options: FrameQrOptions<<<Space as LinearSpace>::F as RlstScalar>::Real>,
    ) -> RlstResult<FrameQrDecomposition<Self, <Space as LinearSpace>::F>> {
        self.imp.qr_pivoted_dyn(options)
    }
}
