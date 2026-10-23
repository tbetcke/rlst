//! Scale factors of unnormalised transforms.

/// A scale factor for the unnormalised transforms of FFTW, as
/// [`Plan::scale_factor`](super::Plan::scale_factor) computes it from the plan's logical size
/// N.
///
/// FFTW's transforms are unnormalised: a backward transform after a forward one multiplies
/// the data by N. The caller applies a factor with RLST's array arithmetic, to the output of
/// the forward transform, of the backward one, or of both with
/// [`InverseSqrt`](Scale::InverseSqrt). The names avoid Forward and Backward, which belong to
/// [`Direction`](super::Direction).
///
/// A uniform factor does not make FFTW's discrete cosine and sine transforms orthonormal,
/// which needs other weights at the end points; RLST provides no orthonormal variants of them
/// (DC5).
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Direction, Layout, Planner, Scale};
///
/// let mut data = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let mut spectrum = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let mut result = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// data.fill_from_seed_equally_distributed(0);
///
/// let layout = Layout::along_axis(&data, &spectrum, 0)?;
/// let forward = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
/// let backward = Planner::<f64>::new().c2c(&layout, Direction::Backward)?;
/// forward.execute(&data, &mut spectrum)?;
/// backward.execute(&spectrum, &mut result)?;
///
/// // Scaled by 1/N, the round trip gives the data back.
/// let scale = backward.scale_factor(Scale::Inverse);
/// assert_eq!(scale, 1.0 / 8.0);
/// for i in 0..8 {
///     assert!((scale * result[[i]] - data[[i]]).norm() < 1e-12);
/// }
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Scale {
    /// The factor 1: no scaling.
    #[default]
    None,
    /// The factor 1/N.
    Inverse,
    /// The factor 1/√N, which applied after both transforms makes their round trip the
    /// identity.
    InverseSqrt,
}

impl Scale {
    /// The factor for the logical size `n`, in `f64`.
    pub(crate) fn factor(self, n: usize) -> f64 {
        let n = n as f64;
        match self {
            Scale::None => 1.0,
            Scale::Inverse => 1.0 / n,
            Scale::InverseSqrt => 1.0 / n.sqrt(),
        }
    }
}
