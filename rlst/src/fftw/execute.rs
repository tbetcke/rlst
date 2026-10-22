//! Checked execution of plans on buffers.

use std::mem::size_of;
use std::ops::Range;

use num::Complex;

use super::align::AlignmentClass;
use super::buffer::{FftBuffer, FftBufferMut};
use super::error::{FftwError, Side};
use super::layout::Placement;
use super::plan::{C2c, Destroying, InPlace, Plan, Policy, Preserving};
use super::real::FftwReal;

/// Check that the bytes of `input_span` elements at `input` and of `output_span` elements at
/// `output` are disjoint ([`FftwError::Overlap`] otherwise).
///
/// Conservative: two strided ranges that interleave without sharing an element are rejected.
pub(crate) fn check_overlap<E>(
    input: *const E,
    input_span: usize,
    output: *const E,
    output_span: usize,
) -> Result<(), FftwError> {
    let range = |ptr: *const E, span: usize| -> Range<usize> {
        let start = ptr.addr();
        start..start.saturating_add(span.saturating_mul(size_of::<E>()))
    };
    let input = range(input, input_span);
    let output = range(output, output_span);
    if input.start < output.end && output.start < input.end {
        return Err(FftwError::Overlap { input, output });
    }
    Ok(())
}

impl<T: FftwReal, P: Policy> Plan<T, C2c, P> {
    /// The checks of design Section 4.8 that need the buffers only shared: each buffer fits
    /// its side of the layout (an array by shape, strides and storage, a slice by length), and
    /// two arrays of the same rank pair their axes by index (DC15). Returns the pointers that
    /// the side checks validated.
    fn check_layout<I, O>(
        &self,
        input: &I,
        output: &O,
    ) -> Result<(*const Complex<T>, *const Complex<T>), FftwError>
    where
        I: FftBuffer<Complex<T>> + ?Sized,
        O: FftBuffer<Complex<T>> + ?Sized,
    {
        let layout = self.layout();
        let input_ptr = input.check_side(layout, Side::Input)?;
        let output_ptr = output.check_side(layout, Side::Output)?;
        if let (Some(input), Some(output)) = (input.array_geometry(), output.array_geometry())
            && input.shape.len() == output.shape.len()
            && !layout.fits_array_pair(
                (&input.shape, &input.stride),
                (&output.shape, &output.stride),
            )
        {
            return Err(FftwError::LayoutMismatch {
                side: Side::Output,
                expected: format!(
                    "an input and an output array whose axes pair by index as in {}",
                    layout.describe()
                ),
                found: format!(
                    "an input of shape {:?} and strides {:?} and an output of shape {:?} and \
                     strides {:?}",
                    input.shape, input.stride, output.shape, output.stride
                ),
            });
        }
        Ok((input_ptr, output_ptr))
    }

    /// The remaining checks of design Section 4.8, on the pointers that the side checks
    /// validated: the alignment classes are the plan's (unless it is unaligned), and, out of
    /// place, the spans do not overlap.
    fn check_pointers(
        &self,
        input: *const Complex<T>,
        output: *const Complex<T>,
    ) -> Result<(), FftwError> {
        let layout = self.layout();
        if let Some((input_class, output_class)) = self.alignment() {
            for (side, expected, found) in [
                (Side::Input, input_class, AlignmentClass::of_ptr(input)),
                (Side::Output, output_class, AlignmentClass::of_ptr(output)),
            ] {
                if found != expected {
                    return Err(FftwError::AlignmentMismatch {
                        side,
                        expected,
                        found,
                    });
                }
            }
        }
        if P::PLACEMENT == Placement::OutOfPlace {
            check_overlap(input, layout.input_span(), output, layout.output_span())?;
        }
        Ok(())
    }

    /// Execute the plan on checked pointers.
    ///
    /// # Safety
    ///
    /// `input` and `output` are pointers that the side checks (`check_side`, or
    /// `check_side_mut` for a buffer that is written) returned for this plan's layout, and
    /// that `check_pointers` accepted; their buffers are borrowed for the duration of the call
    /// (`output` mutably, and `input` mutably unless the plan is `Preserving`), and for an
    /// in-place plan `input == output`.
    unsafe fn execute_unchecked(&self, input: *mut Complex<T>, output: *mut Complex<T>) {
        // SAFETY: FFTW's four new-array conditions (design, Section 3 item 1) hold:
        // 1. sizes and strides: the plan was made for `self.layout()`, and the side checks
        //    found that its input and output spans fit the buffers behind the very pointers
        //    they returned, so every element FFTW addresses lies inside them. For a slice the
        //    span fits its length. For an array or view the pointer is its first element in
        //    the storage slice that was checked, and the span fits that slice from the first
        //    element on. This check, not the comparison of shapes and strides, is what holds
        //    for a wrong implementation of the safe traits `StridedRawAccess`, `Shape` or
        //    `Stride`, even one that returns another slice on another call, since no pointer
        //    from another call is used. In place the input and output spans are equal, as
        //    the strides are. The span of a view can include elements of other views, the
        //    gaps between its own; FFTW reads and writes only the elements the layout
        //    addresses, never the gaps (design Section 4.8), which F2's sentinel tests check;
        // 2. placement: the plan's type fixes it, and an in-place plan is executed on one
        //    buffer (`input == output`) while an out-of-place plan gets two buffers whose byte
        //    ranges `check_pointers` found disjoint;
        // 3. split storage does not apply to interleaved c2c plans;
        // 4. alignment: unless the plan was made with `FFTW_UNALIGNED`, the alignment class of
        //    each pointer equals the class of the corresponding planning array modulo 64, so
        //    FFTW's alignment (modulo 16 or less) is equal too.
        // `output`, and `input` unless the plan is `Preserving`, come from `check_side_mut`
        // and derive from exclusive borrows, of a slice or of an array's whole storage through
        // `StridedRawAccessMut`, so writing through them is permitted. For a `Preserving`
        // plan, `input` may derive from a shared borrow: FFTW was given `FFTW_PRESERVE_INPUT`,
        // so it never writes through it. Execution needs no lock.
        unsafe { T::execute_dft(self.raw(), input, output) }
    }
}

impl<T: FftwReal> Plan<T, C2c, Preserving> {
    /// Transform `input` into `output`, leaving `input` unchanged.
    ///
    /// The buffers are slices or RLST arrays and views. Before FFTW is called, and in this
    /// order: each buffer must fit its side of the layout ([`FftwError::LayoutMismatch`]); a
    /// slice by its length, an array by its shape and strides and by its storage, which must
    /// hold the layout's span from the array's first element on. When both buffers are arrays
    /// of the same rank, their axes must also pair by index as the layout's dimensions do
    /// (equal shapes, and each dimension's input and output strides on one axis). A slice
    /// and an array are checked each on its own side. Then each buffer's alignment class
    /// must be the plan's, unless the plan is unaligned ([`FftwError::AlignmentMismatch`]),
    /// and the buffers must not overlap ([`FftwError::Overlap`]). On an error nothing is
    /// written.
    ///
    /// On arrays:
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Direction, Layout, Planner};
    ///
    /// let mut input = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// input.fill_from_seed_equally_distributed(0);
    ///
    /// let layout = Layout::along_axis(&input, &output, 0)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// plan.execute(&input, &mut output)?;
    /// let sum: c64 = (0..8).map(|i| input[[i, 2]]).sum();
    /// assert!((output[[0, 2]] - sum).norm() < 1e-12);
    ///
    /// // A plan made for two columns runs on any two adjacent columns: the offsets of a
    /// // view count from its first element. Column 1 starts 128 bytes in, at class 0.
    /// let two = |array| Layout::along_axis(array, array, 0);
    /// let layout = two(&input.r().into_subview([0, 0], [8, 2]))?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// plan.execute(
    ///     &input.r().into_subview([0, 1], [8, 2]),
    ///     &mut output.r_mut().into_subview([0, 1], [8, 2]),
    /// )?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    ///
    /// On slices:
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    ///
    /// let mut input = rlst_dynamic_array!(c64, [4] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(c64, [4] | rlst::FFTW_ALIGNED);
    /// input.data_mut().unwrap().fill(c64::new(1.0, 0.0));
    /// plan.execute(input.data().unwrap(), output.data_mut().unwrap())?;
    /// assert_eq!(output.data().unwrap()[0], c64::new(4.0, 0.0));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    ///
    /// The output is written, so a shared view such as `r()` cannot be passed as the output,
    /// and this does not compile:
    ///
    /// ```compile_fail,E0277
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Direction, Layout, Planner};
    ///
    /// let a = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
    /// let mut b = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis(&a, &b, 0)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// plan.execute(&a, &mut b.r())?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    ///
    /// while the same code with the exclusive view `r_mut()` does:
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Direction, Layout, Planner};
    ///
    /// let a = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
    /// let mut b = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis(&a, &b, 0)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// plan.execute(&a, &mut b.r_mut())?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn execute<I, O>(&self, input: &I, output: &mut O) -> Result<(), FftwError>
    where
        I: FftBuffer<Complex<T>> + ?Sized,
        O: FftBufferMut<Complex<T>> + ?Sized,
    {
        let (input, _) = self.check_layout(input, &*output)?;
        let output = output.check_side_mut(self.layout(), Side::Output)?;
        self.check_pointers(input, output)?;
        // SAFETY: the pointers come from the side checks and passed `check_pointers`;
        // `output` is borrowed mutably and `input` shared, which suffices for a `Preserving`
        // plan.
        unsafe { self.execute_unchecked(input.cast_mut(), output) };
        Ok(())
    }
}

impl<T: FftwReal> Plan<T, C2c, Destroying> {
    /// Transform `input` into `output`; FFTW may overwrite `input`.
    ///
    /// The checks are those of the preserving plan's `execute`: the buffers against the
    /// layout, alignment classes and overlap, in this order, before FFTW is called. On an
    /// error nothing is written. FFTW may overwrite the elements of `input` that the layout
    /// addresses, never the elements of a view's storage between them.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().may_destroy_input().c2c(&layout, Direction::Forward)?;
    ///
    /// let mut input = rlst_dynamic_array!(c64, [4] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(c64, [4] | rlst::FFTW_ALIGNED);
    /// input.data_mut().unwrap().fill(c64::new(1.0, 0.0));
    /// plan.execute(input.data_mut().unwrap(), output.data_mut().unwrap())?;
    /// assert_eq!(output.data().unwrap()[0], c64::new(4.0, 0.0));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn execute<I, O>(&self, input: &mut I, output: &mut O) -> Result<(), FftwError>
    where
        I: FftBufferMut<Complex<T>> + ?Sized,
        O: FftBufferMut<Complex<T>> + ?Sized,
    {
        self.check_layout(&*input, &*output)?;
        let input = input.check_side_mut(self.layout(), Side::Input)?;
        let output = output.check_side_mut(self.layout(), Side::Output)?;
        self.check_pointers(input, output)?;
        // SAFETY: the pointers come from the side checks on the exclusive borrows of both
        // buffers, and passed `check_pointers`.
        unsafe { self.execute_unchecked(input, output) };
        Ok(())
    }
}

impl<T: FftwReal> Plan<T, C2c, InPlace> {
    /// Transform `data` in place.
    ///
    /// Before FFTW is called: the buffer must fit the layout, a slice by its length and an
    /// array by its shape, strides and storage ([`FftwError::LayoutMismatch`]), and its
    /// alignment class must be the plan's, unless the plan is unaligned
    /// ([`FftwError::AlignmentMismatch`]). On an error nothing is written.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 1, os: 1 }], &[], Placement::InPlace)?;
    /// let plan = Planner::<f64>::new().c2c_inplace(&layout, Direction::Forward)?;
    ///
    /// let mut data = rlst_dynamic_array!(c64, [4] | rlst::FFTW_ALIGNED);
    /// data.data_mut().unwrap().fill(c64::new(1.0, 0.0));
    /// plan.execute(data.data_mut().unwrap())?;
    /// assert_eq!(data.data().unwrap()[0], c64::new(4.0, 0.0));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn execute<D>(&self, data: &mut D) -> Result<(), FftwError>
    where
        D: FftBufferMut<Complex<T>> + ?Sized,
    {
        self.check_layout(&*data, &*data)?;
        // In place the input and output spans are equal, so one check covers both sides.
        let data = data.check_side_mut(self.layout(), Side::Input)?;
        self.check_pointers(data, data)?;
        // SAFETY: the pointer comes from the side check on the exclusive borrow of the
        // buffer, passed `check_pointers`, and is passed as both input and output of the
        // in-place plan.
        unsafe { self.execute_unchecked(data, data) };
        Ok(())
    }
}

#[cfg(test)]
mod array_tests;

#[cfg(test)]
mod tests {
    use std::fmt;

    use num::Complex;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::check_overlap;
    use crate::fftw::layout::{Dim, Layout, Placement};
    use crate::fftw::test_support::{
        LayoutKind, PolicyKind, SEED, TestReal, aligned, c2c_tolerance, filled, fixed_sizes,
        layouts_for, random_values, reference_c2c, relative_error, transform_offsets, widen,
    };
    use crate::fftw::{AlignmentClass, Direction, FftwError, Planner, Side};

    /// One cell of the correctness matrix.
    #[derive(Clone, Copy, Debug)]
    struct Cell {
        n: usize,
        kind: LayoutKind,
        direction: Direction,
        seed: u64,
    }

    impl fmt::Display for Cell {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "n = {}, layout {:?}, {:?} (seed {:#x})",
                self.n, self.kind, self.direction, self.seed
            )
        }
    }

    fn cell_seed(n: usize, kind: LayoutKind, direction: Direction) -> u64 {
        let kind = LayoutKind::ALL.iter().position(|&k| k == kind).unwrap() as u64;
        SEED ^ ((n as u64) << 16) ^ (kind << 8) ^ (direction == Direction::Backward) as u64
    }

    fn bits<T: TestReal>(values: &[Complex<T>]) -> Vec<(u64, u64)> {
        values
            .iter()
            .map(|value| (value.re.to_f64().to_bits(), value.im.to_f64().to_bits()))
            .collect()
    }

    /// Transform `input` with a plan of `policy` for `layout`, and return the output buffer.
    /// A preserving plan must leave its input bitwise unchanged.
    fn transform<T: TestReal>(
        layout: &Layout,
        direction: Direction,
        policy: PolicyKind,
        input: &[Complex<T>],
        context: &dyn fmt::Display,
    ) -> Vec<Complex<T>> {
        let zero = Complex::new(T::round_from(0.0), T::round_from(0.0));
        match policy {
            PolicyKind::Preserving => {
                let plan = Planner::<T>::new().c2c(layout, direction).unwrap();
                let source = aligned(input);
                let mut output = filled(zero, layout.output_span());
                plan.execute(&source[..], &mut output[..]).unwrap();
                assert_eq!(
                    bits(&source),
                    bits(input),
                    "{} {context}: a preserving plan changed its input",
                    T::NAME
                );
                output.to_vec()
            }
            PolicyKind::Destroying => {
                let plan = Planner::<T>::new()
                    .may_destroy_input()
                    .c2c(layout, direction)
                    .unwrap();
                let mut source = aligned(input);
                let mut output = filled(zero, layout.output_span());
                plan.execute(&mut source[..], &mut output[..]).unwrap();
                output.to_vec()
            }
            PolicyKind::InPlace => {
                let plan = Planner::<T>::new().c2c_inplace(layout, direction).unwrap();
                let mut data = aligned(input);
                plan.execute(&mut data[..]).unwrap();
                data.to_vec()
            }
        }
    }

    /// The transforms of `output`, widened, in the order of the reference.
    fn gather<T: TestReal>(layout: &Layout, output: &[Complex<T>]) -> Vec<Vec<Complex<f64>>> {
        transform_offsets(layout, false)
            .iter()
            .map(|offsets| widen(&offsets.iter().map(|&o| output[o]).collect::<Vec<_>>()))
            .collect()
    }

    /// Check every policy of a cell against the reference, and, for forward cells, the round
    /// trip.
    fn check_cell<T: TestReal>(cell: Cell, policies: &[PolicyKind]) {
        // The spans do not depend on the placement, as every layout has equal strides.
        let layout = cell.kind.layout(cell.n, Placement::OutOfPlace);
        let input = random_values::<T>(cell.seed, layout.input_span());
        let reference = reference_c2c(&layout, cell.direction, &widen(&input));
        let tolerance = c2c_tolerance::<T>(layout.logical_size());

        for &policy in policies {
            let layout = cell.kind.layout(cell.n, policy.placement());
            let context = format!("{cell}, {policy:?}");
            let output = transform::<T>(&layout, cell.direction, policy, &input, &context);
            for (index, (actual, expected)) in
                gather(&layout, &output).iter().zip(&reference).enumerate()
            {
                let error = relative_error(actual, expected);
                assert!(
                    error <= tolerance,
                    "{} {context}, batch index {index}: relative error {error:e} above {tolerance:e}",
                    T::NAME
                );
            }

            if cell.direction == Direction::Forward {
                check_round_trip::<T>(&layout, policy, &input, &context);
            }
        }
    }

    /// The backward transform after the forward one is N times the input.
    fn check_round_trip<T: TestReal>(
        layout: &Layout,
        policy: PolicyKind,
        input: &[Complex<T>],
        context: &dyn fmt::Display,
    ) {
        let forward = transform::<T>(layout, Direction::Forward, policy, input, context);
        // Out of place, the backward transform reads the forward output, which has the
        // layout's output strides; every layout here has equal strides on both sides.
        let backward = transform::<T>(layout, Direction::Backward, policy, &forward, context);
        let n = layout.logical_size() as f64;
        let scaled: Vec<Vec<Complex<f64>>> = transform_offsets(layout, true)
            .iter()
            .map(|offsets| offsets.iter().map(|&o| widen(&[input[o]])[0] * n).collect())
            .collect();
        let tolerance = c2c_tolerance::<T>(layout.logical_size());
        for (index, (actual, expected)) in gather(layout, &backward).iter().zip(&scaled).enumerate()
        {
            let error = relative_error(actual, expected);
            assert!(
                error <= tolerance,
                "{} {context}, round trip, batch index {index}: relative error {error:e} above \
                 {tolerance:e}",
                T::NAME
            );
        }
    }

    fn check_fixed_cells<T: TestReal>(direction: Direction) {
        for n in fixed_sizes() {
            for &kind in layouts_for(n) {
                let cell = Cell {
                    n,
                    kind,
                    direction,
                    seed: cell_seed(n, kind, direction),
                };
                check_cell::<T>(cell, &PolicyKind::ALL);
            }
        }
    }

    #[test]
    fn test_c2c_fixed_cells_f64_forward() {
        check_fixed_cells::<f64>(Direction::Forward);
    }

    #[test]
    fn test_c2c_fixed_cells_f64_backward() {
        check_fixed_cells::<f64>(Direction::Backward);
    }

    #[test]
    fn test_c2c_fixed_cells_f32_forward() {
        check_fixed_cells::<f32>(Direction::Forward);
    }

    #[test]
    fn test_c2c_fixed_cells_f32_backward() {
        check_fixed_cells::<f32>(Direction::Backward);
    }

    #[test]
    fn test_c2c_sampled_cells() {
        // 32 cells from the product of sizes up to 1080, layouts, directions, precisions and
        // policies, drawn with the fixed seed.
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        for _ in 0..32 {
            let n = rng.random_range(1..=1080_usize);
            let kind = layouts_for(n)[rng.random_range(0..layouts_for(n).len())];
            let direction = if rng.random_bool(0.5) {
                Direction::Forward
            } else {
                Direction::Backward
            };
            let policy = PolicyKind::ALL[rng.random_range(0..3)];
            let cell = Cell {
                n,
                kind,
                direction,
                seed: rng.random(),
            };
            if rng.random_bool(0.5) {
                check_cell::<f64>(cell, &[policy]);
            } else {
                check_cell::<f32>(cell, &[policy]);
            }
        }
    }

    fn check_one_plan_many_buffers<T: TestReal>() {
        // One plan executed on four distinct buffers of class 0 gives, bit for bit, the
        // results of a plan made for each buffer.
        let layout = LayoutKind::Batch3.layout(588, Placement::OutOfPlace);
        let plan = Planner::<T>::new()
            .c2c(&layout, Direction::Forward)
            .unwrap();
        let zero = Complex::new(T::round_from(0.0), T::round_from(0.0));
        for buffer in 0..4 {
            let seed = SEED + buffer;
            let mut input = aligned(&random_values::<T>(seed, layout.input_span()));
            let mut shared = filled(zero, layout.output_span());
            plan.execute(&input[..], &mut shared[..]).unwrap();

            let mut own = filled(zero, layout.output_span());
            let own_plan = Planner::<T>::new()
                .c2c_on(&mut input[..], &mut own[..], &layout, Direction::Forward)
                .unwrap();
            own_plan.execute(&input[..], &mut own[..]).unwrap();
            assert_eq!(
                bits(&shared),
                bits(&own),
                "{} buffer {buffer} (seed {seed:#x}): the shared plan differs from its own plan",
                T::NAME
            );
        }
    }

    #[test]
    fn test_one_plan_many_buffers() {
        check_one_plan_many_buffers::<f64>();
        check_one_plan_many_buffers::<f32>();
    }

    const SENTINEL: Complex<f64> = Complex::new(1234.5, -678.25);

    fn contiguous(n: isize, placement: Placement) -> Layout {
        Layout::new(&[Dim { n, is: 1, os: 1 }], &[], placement).unwrap()
    }

    #[test]
    fn test_short_slice() {
        let layout = contiguous(16, Placement::OutOfPlace);
        let plan = Planner::<f64>::new()
            .c2c(&layout, Direction::Forward)
            .unwrap();
        let input = filled(Complex::new(1.0, 0.0), 16);
        let mut output = filled(SENTINEL, 16);

        let error = plan.execute(&input[..15], &mut output[..]).unwrap_err();
        assert!(
            matches!(
                error,
                FftwError::LayoutMismatch {
                    side: Side::Input,
                    ..
                }
            ),
            "{error}"
        );
        let error = plan.execute(&input[..], &mut output[..15]).unwrap_err();
        assert!(
            matches!(
                error,
                FftwError::LayoutMismatch {
                    side: Side::Output,
                    ..
                }
            ),
            "{error}"
        );
        assert!(output.iter().all(|&value| value == SENTINEL));

        let mut data = filled(SENTINEL, 16);
        let plan = Planner::<f64>::new()
            .c2c_inplace(&contiguous(16, Placement::InPlace), Direction::Forward)
            .unwrap();
        let error = plan.execute(&mut data[..15]).unwrap_err();
        assert!(matches!(error, FftwError::LayoutMismatch { .. }), "{error}");
        assert!(data.iter().all(|&value| value == SENTINEL));
    }

    #[test]
    fn test_alignment_mismatch() {
        let layout = contiguous(16, Placement::OutOfPlace);
        let plan = Planner::<f64>::new()
            .c2c(&layout, Direction::Forward)
            .unwrap();
        let input = filled(Complex::new(1.0, 0.0), 17);
        let mut output = filled(SENTINEL, 17);

        // One element off class 0.
        let error = plan.execute(&input[1..], &mut output[..16]).unwrap_err();
        assert_eq!(
            error,
            FftwError::AlignmentMismatch {
                side: Side::Input,
                expected: AlignmentClass::default(),
                found: AlignmentClass::from_offset(16),
            }
        );
        let error = plan.execute(&input[..16], &mut output[1..]).unwrap_err();
        assert!(
            matches!(
                error,
                FftwError::AlignmentMismatch {
                    side: Side::Output,
                    ..
                }
            ),
            "{error}"
        );
        assert!(output.iter().all(|&value| value == SENTINEL));

        // In single precision, one element off class 0 is class 8.
        let plan = Planner::<f32>::new()
            .c2c_inplace(&contiguous(16, Placement::InPlace), Direction::Forward)
            .unwrap();
        let sentinel = Complex::new(1234.5_f32, -678.25);
        let mut data = filled(sentinel, 17);
        let error = plan.execute(&mut data[1..]).unwrap_err();
        assert_eq!(
            error,
            FftwError::AlignmentMismatch {
                side: Side::Input,
                expected: AlignmentClass::default(),
                found: AlignmentClass::from_offset(8),
            }
        );
        assert!(data.iter().all(|&value| value == sentinel));
    }

    #[test]
    fn test_other_alignment_classes() {
        // A plan for class 16 runs on buffers one element off class 0, and not on class 0.
        let layout = contiguous(16, Placement::OutOfPlace);
        let class = AlignmentClass::from_offset(16);
        let plan = Planner::<f64>::new()
            .alignment(class, class)
            .c2c(&layout, Direction::Forward)
            .unwrap();
        assert_eq!(plan.alignment(), Some((class, class)));
        let input = aligned(&random_values::<f64>(SEED, 17));
        let mut output = filled(SENTINEL, 17);
        plan.execute(&input[1..], &mut output[1..]).unwrap();
        let expected = reference_c2c(&layout, Direction::Forward, &input[1..]);
        let error = relative_error(&output[1..], &expected[0]);
        assert!(
            error <= c2c_tolerance::<f64>(16),
            "relative error {error:e}"
        );
        let error = plan.execute(&input[..16], &mut output[..16]).unwrap_err();
        assert!(
            matches!(error, FftwError::AlignmentMismatch { .. }),
            "{error}"
        );

        // An unaligned plan runs on any class.
        let plan = Planner::<f64>::new()
            .unaligned()
            .c2c(&layout, Direction::Forward)
            .unwrap();
        assert_eq!(plan.alignment(), None);
        plan.execute(&input[1..], &mut output[..16]).unwrap();
        let error = relative_error(&output[..16], &expected[0]);
        assert!(
            error <= c2c_tolerance::<f64>(16),
            "relative error {error:e}"
        );

        // A class that is not a multiple of the complex alignment, and unequal classes in
        // place, are rejected before planning.
        let error = Planner::<f64>::new()
            .alignment(AlignmentClass::from_offset(4), AlignmentClass::default())
            .c2c(&layout, Direction::Forward)
            .unwrap_err();
        assert!(matches!(error, FftwError::InvalidLayout { .. }), "{error}");
        let error = Planner::<f64>::new()
            .alignment(AlignmentClass::default(), class)
            .c2c_inplace(&contiguous(16, Placement::InPlace), Direction::Forward)
            .unwrap_err();
        assert!(matches!(error, FftwError::InvalidLayout { .. }), "{error}");
    }

    #[test]
    fn test_placement_mismatch() {
        let in_place = contiguous(8, Placement::InPlace);
        let out_of_place = contiguous(8, Placement::OutOfPlace);
        let expected = FftwError::PlacementMismatch {
            expected: Placement::OutOfPlace,
            found: Placement::InPlace,
        };
        assert_eq!(
            Planner::<f64>::new()
                .c2c(&in_place, Direction::Forward)
                .unwrap_err(),
            expected
        );
        let mut input = filled(Complex::new(1.0, 0.0), 8);
        let mut output = filled(SENTINEL, 8);
        let error = Planner::<f64>::new()
            .c2c_on(
                &mut input[..],
                &mut output[..],
                &in_place,
                Direction::Forward,
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(output.iter().all(|&value| value == SENTINEL));

        let error = Planner::<f64>::new()
            .c2c_inplace(&out_of_place, Direction::Forward)
            .unwrap_err();
        assert_eq!(
            error,
            FftwError::PlacementMismatch {
                expected: Placement::InPlace,
                found: Placement::OutOfPlace,
            }
        );
    }

    #[test]
    fn test_overlap() {
        let layout = contiguous(16, Placement::OutOfPlace);
        let plan = Planner::<f64>::new()
            .c2c(&layout, Direction::Forward)
            .unwrap();
        let mut buffer = filled(SENTINEL, 32);
        let base = buffer.as_mut_ptr();
        {
            // Input and output overlap by half, which safe borrows rule out. They are built
            // from one allocation only to test the check: the check reads the slices' addresses
            // and lengths and nothing else, and on the error no element is read or written
            // through either slice. The output starts 8 elements, 128 bytes, in, so its class
            // is 0 like the input's and the alignment check passes.
            // SAFETY: both ranges lie inside `buffer`, which outlives the slices.
            let input = unsafe { std::slice::from_raw_parts(base.cast_const(), 16) };
            // SAFETY: as above.
            let output = unsafe { std::slice::from_raw_parts_mut(base.add(8), 16) };
            let error = plan.execute(input, output).unwrap_err();
            assert!(matches!(error, FftwError::Overlap { .. }), "{error}");
        }
        assert!(buffer.iter().all(|&value| value == SENTINEL));

        // The check itself, on raw pointers: adjacent ranges are disjoint.
        let start = base.cast_const();
        assert!(check_overlap(start, 16, start.wrapping_add(16), 16).is_ok());
        assert!(check_overlap(start.wrapping_add(16), 16, start, 16).is_ok());
        assert!(check_overlap(start, 17, start.wrapping_add(16), 16).is_err());
        assert!(check_overlap(start.wrapping_add(15), 16, start, 16).is_err());
    }

    #[test]
    fn test_scratch_too_large() {
        // Four elements 1000 apart span 3001 elements: 2 * 3001 * 16 bytes of scratch.
        let layout = Layout::new(
            &[Dim {
                n: 4,
                is: 1000,
                os: 1000,
            }],
            &[],
            Placement::OutOfPlace,
        )
        .unwrap();
        let planner = Planner::<f64>::new().max_scratch(1024);
        assert_eq!(
            planner.c2c(&layout, Direction::Forward).unwrap_err(),
            FftwError::ScratchTooLarge {
                needed: 2 * 3001 * 16,
                bound: 1024,
            }
        );

        // The same layout plans on the caller's buffers.
        let mut input = aligned(&random_values::<f64>(SEED, 3001));
        let mut output = filled(SENTINEL, 3001);
        let plan = planner
            .c2c_on(&mut input[..], &mut output[..], &layout, Direction::Forward)
            .unwrap();
        plan.execute(&input[..], &mut output[..]).unwrap();
        let expected = reference_c2c(&layout, Direction::Forward, &input);
        let actual: Vec<Complex<f64>> = (0..4).map(|k| output[1000 * k]).collect();
        let error = relative_error(&actual, &expected[0]);
        assert!(error <= c2c_tolerance::<f64>(4), "relative error {error:e}");
        // Elements the layout does not address are not written.
        assert_eq!(output[1], SENTINEL);

        // In place, the scratch is one span.
        let layout = Layout::new(
            &[Dim {
                n: 4,
                is: 1000,
                os: 1000,
            }],
            &[],
            Placement::InPlace,
        )
        .unwrap();
        assert_eq!(
            planner
                .c2c_inplace(&layout, Direction::Forward)
                .unwrap_err(),
            FftwError::ScratchTooLarge {
                needed: 3001 * 16,
                bound: 1024,
            }
        );
    }

    #[test]
    fn test_scratch_size_overflow() {
        // Spans whose bytes overflow `usize` are above every bound, even `usize::MAX`.
        let huge = isize::MAX / 4;
        let layout = Layout::new(
            &[Dim {
                n: 2,
                is: huge,
                os: huge,
            }],
            &[],
            Placement::OutOfPlace,
        )
        .unwrap();
        let error = Planner::<f64>::new()
            .max_scratch(usize::MAX)
            .c2c(&layout, Direction::Forward)
            .unwrap_err();
        assert_eq!(
            error,
            FftwError::ScratchTooLarge {
                needed: usize::MAX,
                bound: usize::MAX,
            }
        );
    }

    #[test]
    fn test_planner_options() {
        let layout = contiguous(32, Placement::OutOfPlace);
        let plan = Planner::<f64>::new()
            .time_limit(10.0)
            .c2c(&layout, Direction::Backward)
            .unwrap();
        assert_eq!(plan.direction(), Direction::Backward);
        assert_eq!(plan.layout(), &layout);
        let debug = format!("{plan:?}");
        assert!(debug.contains("Preserving"), "{debug}");
    }
}
