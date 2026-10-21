//! Transform directions.

/// The direction of a complex-to-complex transform.
///
/// [`Forward`](Direction::Forward) computes `y[k] = Σ_j x[j] exp(-2πi jk/n)` and
/// [`Backward`](Direction::Backward) the same sum with `+2πi`: FFTW's signs −1 and +1.
/// Neither is normalised, so a backward transform after a forward one multiplies the data by
/// the logical size of the transform.
///
/// ```
/// use rlst::fftw::Direction;
///
/// let directions = [Direction::Forward, Direction::Backward];
/// assert_ne!(directions[0], directions[1]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// The forward transform, with exponent sign −1.
    Forward,
    /// The backward transform, with exponent sign +1.
    Backward,
}
