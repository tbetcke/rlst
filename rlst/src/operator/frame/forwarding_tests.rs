use std::cell::Cell;

use crate::operator::{element::Element, interface::array_vector_space::ArrayVectorSpace};
use crate::{
    Array, ArrayFrameImpl, ElementFrame, FrameApply, FrameImpl, FrameQr, FrameQrDecomposition,
    FrameQrOptions, GenericFrameImpl, InnerProductSpace, LinearSpace, Norm, RlstResult, Shape,
    SliceArray, SpaceFrame, UnsafeRandom1DAccessByValue,
};

// Use a third implementation to ensure dispatch is truly generic. The counters
// distinguish backend overrides from defaults that would produce the same values.
struct ObservedFrameImpl<'a> {
    inner: GenericFrameImpl<'a, ArrayVectorSpace<f64>>,
    is_empty_calls: Cell<usize>,
    apply_calls: Cell<usize>,
    apply_into_calls: Cell<usize>,
    qr_calls: Cell<usize>,
    generation: usize,
}

impl<'a> ObservedFrameImpl<'a> {
    fn new(inner: GenericFrameImpl<'a, ArrayVectorSpace<f64>>, generation: usize) -> Self {
        Self {
            inner,
            is_empty_calls: Cell::new(0),
            apply_calls: Cell::new(0),
            apply_into_calls: Cell::new(0),
            qr_calls: Cell::new(0),
            generation,
        }
    }
}

impl<'a> FrameImpl<'a> for ObservedFrameImpl<'a> {
    type Space = ArrayVectorSpace<f64>;

    fn space(&self) -> &'a Self::Space {
        self.inner.space()
    }
    fn element(&self, index: usize) -> Option<Element<'a, Self::Space>> {
        self.inner.element(index)
    }
    fn set_element(&mut self, index: usize, elem: &Element<Self::Space>) -> RlstResult<()> {
        self.inner.set_element(index, elem)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn is_empty(&self) -> bool {
        self.is_empty_calls.set(self.is_empty_calls.get() + 1);
        self.inner.is_empty()
    }
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }
    fn push(&mut self, elem: &Element<Self::Space>) -> RlstResult<()> {
        self.inner.push(elem)
    }
    fn pop(&mut self) -> Option<Element<'a, Self::Space>> {
        self.inner.pop()
    }
    fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<'a> FrameApply<'a> for ObservedFrameImpl<'a> {
    fn apply<A>(&self, coefficients: &Array<A, 1>) -> RlstResult<Element<'a, Self::Space>>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = f64>,
    {
        self.apply_calls.set(self.apply_calls.get() + 1);
        self.inner.apply(coefficients)
    }

    fn apply_into<A>(
        &self,
        alpha: f64,
        coefficients: &Array<A, 1>,
        beta: f64,
        output: &mut Element<Self::Space>,
    ) -> RlstResult<()>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = f64>,
    {
        self.apply_into_calls.set(self.apply_into_calls.get() + 1);
        self.inner.apply_into(alpha, coefficients, beta, output)
    }
}

impl<'a> FrameQr<'a> for ObservedFrameImpl<'a> {
    fn qr_pivoted(
        &self,
        options: FrameQrOptions<f64>,
    ) -> RlstResult<FrameQrDecomposition<Self, f64>> {
        self.qr_calls.set(self.qr_calls.get() + 1);
        let result = self.inner.qr_pivoted(options)?;
        Ok(FrameQrDecomposition {
            q: Self::new(result.q, self.generation + 1),
            r: result.r,
            permutation: result.permutation,
        })
    }
}

#[test]
fn wrapper_preserves_backend_overrides_through_inherent_and_trait_methods() {
    let space = ArrayVectorSpace::<f64>::new(2);
    let mut frame = ElementFrame::new(ObservedFrameImpl::new(GenericFrameImpl::new(&space), 7));
    assert!(frame.is_empty());
    assert!(FrameImpl::is_empty(&frame));
    assert_eq!(frame.imp().is_empty_calls.get(), 2);
    let mut element = space.zero();
    element.imp_mut()[[0]] = 2.0;
    frame.push(&element).unwrap();
    let coefficients = SliceArray::from_shape(&[3.0], [1]);
    assert_eq!(frame.apply(&coefficients).unwrap().imp()[[0]], 6.0);
    assert_eq!(
        FrameApply::apply(&frame, &coefficients).unwrap().imp()[[0]],
        6.0
    );
    assert_eq!(frame.imp().apply_calls.get(), 2);

    let mut output = space.zero();
    frame
        .apply_into(2.0, &coefficients, 0.0, &mut output)
        .unwrap();
    assert_eq!(output.imp()[[0]], 12.0);
    FrameApply::apply_into(&frame, 1.0, &coefficients, 2.0, &mut output).unwrap();
    assert_eq!(output.imp()[[0]], 30.0);
    assert_eq!(frame.imp().apply_into_calls.get(), 2);
    assert_eq!(frame.imp().apply_calls.get(), 2);

    let qr = frame.qr_pivoted(FrameQrOptions::default()).unwrap();
    let _: &ElementFrame<'_, ObservedFrameImpl<'_>> = &qr.q;
    assert_eq!(qr.q.imp().generation, 8);
    assert!(std::ptr::eq(qr.q.space(), &space));
    assert_eq!(qr.permutation, vec![0]);
    assert!((&qr.q.apply(&qr.r.r().col(0)).unwrap() - &element).norm() < 1e-12);
    let qr = FrameQr::qr_pivoted(&frame, FrameQrOptions::default()).unwrap();
    assert_eq!(qr.q.imp().generation, 8);
    assert_eq!(frame.imp().qr_calls.get(), 2);
    assert!(
        frame
            .qr_pivoted(FrameQrOptions {
                atol: -1.0,
                rtol: 0.0
            })
            .is_err()
    );
    assert_eq!(frame.imp().qr_calls.get(), 3);
}

#[test]
fn implementation_access_and_extraction_preserve_storage() {
    let space = ArrayVectorSpace::<f64>::new(2);
    let mut backend = ArrayFrameImpl::new(&space);
    backend.push(&space.zero()).unwrap();
    let pointer = backend.matrix().data().unwrap().as_ptr();
    let mut frame = ElementFrame::new(backend);
    assert_eq!(frame.matrix().data().unwrap().as_ptr(), pointer);
    frame.imp_mut().column_mut(0).unwrap()[[1]] = 42.0;
    assert_eq!(frame.column(0).unwrap()[[1]], 42.0);
    let mut backend = frame.into_imp();
    assert_eq!(backend.matrix().data().unwrap().as_ptr(), pointer);
    assert_eq!(backend.pop().unwrap().imp()[[1]], 42.0);

    let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
    generic.imp_mut().push(&space.zero()).unwrap();
    assert_eq!(generic.len(), 1);
    let backend = generic.into_imp();
    assert_eq!(backend.iter().count(), 1);
}

#[test]
fn storage_does_not_require_application_or_inner_products() {
    let space = ArrayVectorSpace::<i64>::new(2);
    let mut frame = ElementFrame::new(ArrayFrameImpl::new(&space));
    let mut element = space.zero();
    element.imp_mut()[[1]] = 9;
    frame.push(&element).unwrap();
    assert_eq!(frame.pop().unwrap().imp()[[1]], 9);
    assert!(frame.is_empty());
}

// A space with no array backing, whose frame storage counts the calls that
// reach it. Overriding `frame` is the only way a space selects storage.
struct CountingSpace {
    pushes: Cell<usize>,
}

impl LinearSpace for CountingSpace {
    type F = f64;
    type Impl = [f64; 2];

    fn zero(&self) -> Element<'_, Self> {
        Element::new(self, [0.0; 2])
    }
    fn add(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self> {
        Element::new(self, std::array::from_fn(|i| x.imp()[i] + y.imp()[i]))
    }
    fn sub(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self> {
        Element::new(self, std::array::from_fn(|i| x.imp()[i] - y.imp()[i]))
    }
    fn scalar_mul(&self, scalar: &f64, x: &Element<Self>) -> Element<'_, Self> {
        Element::new(self, x.imp().map(|v| v * *scalar))
    }
    fn neg(&self, x: &Element<Self>) -> Element<'_, Self> {
        self.scalar_mul(&-1.0, x)
    }
    fn sum_inplace(&self, x: &mut Element<Self>, y: &Element<Self>) {
        for i in 0..2 {
            x.imp_mut()[i] += y.imp()[i];
        }
    }
    fn sub_inplace(&self, x: &mut Element<Self>, y: &Element<Self>) {
        for i in 0..2 {
            x.imp_mut()[i] -= y.imp()[i];
        }
    }
    fn scale_inplace(&self, scalar: &f64, x: &mut Element<Self>) {
        for value in x.imp_mut() {
            *value *= *scalar;
        }
    }
    fn copy_from(&self, x: &Element<Self>) -> Element<'_, Self> {
        Element::new(self, *x.imp())
    }
}

impl InnerProductSpace for CountingSpace {
    fn inner_product(&self, x: &Element<Self>, y: &Element<Self>) -> f64 {
        (0..2).map(|i| x.imp()[i] * y.imp()[i]).sum()
    }

    fn frame(&self) -> ElementFrame<'_, SpaceFrame<'_, Self>> {
        ElementFrame::new(SpaceFrame::new(CountingFrameImpl {
            inner: GenericFrameImpl::new(self),
        }))
    }
}

struct CountingFrameImpl<'a> {
    inner: GenericFrameImpl<'a, CountingSpace>,
}

impl<'a> FrameImpl<'a> for CountingFrameImpl<'a> {
    type Space = CountingSpace;

    fn space(&self) -> &'a Self::Space {
        self.inner.space()
    }
    fn element(&self, index: usize) -> Option<Element<'a, Self::Space>> {
        self.inner.element(index)
    }
    fn set_element(&mut self, index: usize, elem: &Element<Self::Space>) -> RlstResult<()> {
        self.inner.set_element(index, elem)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }
    fn push(&mut self, elem: &Element<Self::Space>) -> RlstResult<()> {
        let space = self.inner.space();
        space.pushes.set(space.pushes.get() + 1);
        self.inner.push(elem)
    }
    fn pop(&mut self) -> Option<Element<'a, Self::Space>> {
        self.inner.pop()
    }
    fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<'a> FrameApply<'a> for CountingFrameImpl<'a> {}

impl<'a> FrameQr<'a> for CountingFrameImpl<'a> {
    fn qr_pivoted(
        &self,
        options: FrameQrOptions<f64>,
    ) -> RlstResult<FrameQrDecomposition<Self, f64>> {
        let result = self.inner.qr_pivoted(options)?;
        Ok(FrameQrDecomposition {
            q: Self { inner: result.q },
            r: result.r,
            permutation: result.permutation,
        })
    }
}

// Generic over the space, exactly as an iterative solver is: nothing here can
// name the backend the space selected.
fn push_through_generic_code<S: InnerProductSpace<F = f64>>(space: &S) -> usize {
    let mut frame = space.frame();
    let element = space.zero();
    frame.push(&element).unwrap();
    frame.push(&element).unwrap();
    frame.len()
}

#[test]
fn frame_override_is_reached_from_generic_code() {
    let space = CountingSpace {
        pushes: Cell::new(0),
    };
    assert_eq!(push_through_generic_code(&space), 2);
    // The default implementation would never touch these counters.
    assert_eq!(space.pushes.get(), 2);
}

#[test]
fn default_frame_needs_no_space_implementation() {
    // ArrayVectorSpace overrides `frame`; a space that does not gets element
    // storage without writing anything.
    let space = CountingSpace {
        pushes: Cell::new(0),
    };
    let generic = ElementFrame::new(GenericFrameImpl::new(&space));
    assert!(generic.is_empty());
}
