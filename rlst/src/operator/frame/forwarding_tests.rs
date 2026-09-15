use std::cell::Cell;

use crate::operator::{element::Element, interface::array_vector_space::ArrayVectorSpace};
use crate::{
    Array, ArrayFrameImpl, ElementFrame, FrameApply, FrameImpl, FrameQr, FrameQrDecomposition,
    FrameQrOptions, FrameRefAccess, GenericFrameImpl, LinearSpace, Norm, RlstResult, Shape,
    SliceArray, UnsafeRandom1DAccessByValue,
};

// Use a third implementation to ensure dispatch is truly generic. The counters
// distinguish backend overrides from defaults that would produce the same values.
struct ObservedFrameImpl<'a> {
    inner: GenericFrameImpl<'a, ArrayVectorSpace<f64>>,
    is_empty_calls: Cell<usize>,
    apply_calls: Cell<usize>,
    apply_into_calls: Cell<usize>,
    qr_calls: Cell<usize>,
    ref_calls: Cell<usize>,
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
            ref_calls: Cell::new(0),
            generation,
        }
    }
}

impl<'a> FrameRefAccess<'a> for ObservedFrameImpl<'a> {
    fn element_ref(&self, index: usize) -> Option<&Element<'a, Self::Space>> {
        self.ref_calls.set(self.ref_calls.get() + 1);
        self.inner.element_ref(index)
    }
    fn element_mut<'s>(&'s mut self, index: usize) -> Option<&'s mut Element<'a, Self::Space>> {
        self.ref_calls.set(self.ref_calls.get() + 1);
        self.inner.element_mut(index)
    }
    fn iter<'s>(&'s self) -> impl ExactSizeIterator<Item = &'s Element<'a, Self::Space>>
    where
        'a: 's,
    {
        self.ref_calls.set(self.ref_calls.get() + 1);
        self.inner.iter()
    }
    fn iter_mut<'s>(&'s mut self) -> impl ExactSizeIterator<Item = &'s mut Element<'a, Self::Space>>
    where
        'a: 's,
    {
        self.ref_calls.set(self.ref_calls.get() + 1);
        self.inner.iter_mut()
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

#[test]
fn wrapper_forwards_reference_access_to_the_backend() {
    let space = ArrayVectorSpace::<f64>::new(2);
    let mut inner = GenericFrameImpl::new(&space);
    let mut element = space.zero();
    element.imp_mut()[[0]] = 7.0;
    inner.push(&element).unwrap();
    let mut frame = ElementFrame::new(ObservedFrameImpl::new(inner, 0));

    assert_eq!(frame.element_ref(0).unwrap().imp()[[0]], 7.0);
    assert!(frame.element_ref(1).is_none());
    frame.element_mut(0).unwrap().imp_mut()[[0]] = 8.0;
    assert_eq!(frame.iter().next().unwrap().imp()[[0]], 8.0);
    for elem in frame.iter_mut() {
        elem.imp_mut()[[0]] += 1.0;
    }
    assert_eq!(frame.element_ref(0).unwrap().imp()[[0]], 9.0);

    // Two `element_ref`, one `element_mut`, one `iter`, one `iter_mut`, and the
    // final `element_ref`: every call reached the backend.
    assert_eq!(frame.imp().ref_calls.get(), 6);
}
