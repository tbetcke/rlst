//! Checked execution of plans on buffers.

use num::Complex;

use super::buffer::{FftBuffer, FftBufferMut};
use super::error::FftwError;
use super::plan::{C2c, Destroying, InPlace, Plan, Preserving};
use super::real::FftwReal;

impl<T: FftwReal> Plan<T, C2c, Preserving> {
    /// Transform `input` into `output`, leaving `input` unchanged.
    ///
    /// Before FFTW is called, and in this order: the layout's spans must fit the buffers
    /// ([`FftwError::LayoutMismatch`]); each buffer's alignment class must be the plan's,
    /// unless the plan is unaligned ([`FftwError::AlignmentMismatch`]); and the buffers must
    /// not overlap ([`FftwError::Overlap`]). On an error nothing is written.
    ///
    /// ```no_run
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
    pub fn execute<I, O>(&self, input: &I, output: &mut O) -> Result<(), FftwError>
    where
        I: FftBuffer<Complex<T>> + ?Sized,
        O: FftBufferMut<Complex<T>> + ?Sized,
    {
        let _ = (input, output);
        todo!()
    }
}

impl<T: FftwReal> Plan<T, C2c, Destroying> {
    /// Transform `input` into `output`; FFTW may overwrite `input`.
    ///
    /// The checks are those of the preserving plan's `execute`: spans, alignment classes and
    /// overlap, in this order, before FFTW is called. On an error nothing is written.
    ///
    /// ```no_run
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
        let _ = (input, output);
        todo!()
    }
}

impl<T: FftwReal> Plan<T, C2c, InPlace> {
    /// Transform `data` in place.
    ///
    /// Before FFTW is called: the layout's span must fit the buffer
    /// ([`FftwError::LayoutMismatch`]), and its alignment class must be the plan's, unless the
    /// plan is unaligned ([`FftwError::AlignmentMismatch`]). On an error nothing is written.
    ///
    /// ```no_run
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
        let _ = data;
        todo!()
    }
}
