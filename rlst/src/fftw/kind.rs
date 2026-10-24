//! Transform directions and real-to-real kinds.

use std::ffi::c_int;
use std::fmt;

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

impl Direction {
    /// FFTW's sign of the exponent.
    pub(crate) fn sign(self) -> c_int {
        match self {
            // fftw-sys gives FFTW_FORWARD as i32 and FFTW_BACKWARD as u32.
            Direction::Forward => fftw_sys::FFTW_FORWARD as c_int,
            Direction::Backward => fftw_sys::FFTW_BACKWARD as c_int,
        }
    }
}

/// FFTW's eleven real-to-real transform kinds, as RLST's own type.
///
/// A real-to-real plan takes one kind per transform dimension: `kinds[i]` belongs to
/// `layout.dims()[i]`, and so, for a layout made from arrays, to `axes[i]`. Each kind is
/// unnormalised. Its logical size N ([`R2rKind::logical_size`]) is the factor by which the
/// kind and its inverse multiply the data together (FFTW manual, "Real-to-Real Transform
/// Kinds"):
///
/// | Kind | Transform | Logical size N | Inverse |
/// | --- | --- | --- | --- |
/// | [`R2hc`](R2rKind::R2hc) | real to halfcomplex DFT | n | `Hc2r` |
/// | [`Hc2r`](R2rKind::Hc2r) | halfcomplex to real DFT | n | `R2hc` |
/// | [`Dht`](R2rKind::Dht) | discrete Hartley transform | n | `Dht` |
/// | [`Redft00`](R2rKind::Redft00) | DCT-I | 2 (n − 1) | `Redft00` |
/// | [`Redft10`](R2rKind::Redft10) | DCT-II, "the" DCT | 2 n | `Redft01` |
/// | [`Redft01`](R2rKind::Redft01) | DCT-III, "the" inverse DCT | 2 n | `Redft10` |
/// | [`Redft11`](R2rKind::Redft11) | DCT-IV | 2 n | `Redft11` |
/// | [`Rodft00`](R2rKind::Rodft00) | DST-I | 2 (n + 1) | `Rodft00` |
/// | [`Rodft10`](R2rKind::Rodft10) | DST-II | 2 n | `Rodft01` |
/// | [`Rodft01`](R2rKind::Rodft01) | DST-III | 2 n | `Rodft10` |
/// | [`Rodft11`](R2rKind::Rodft11) | DST-IV | 2 n | `Rodft11` |
///
/// `Redft00` is not defined for n = 1; planning it returns
/// [`FftwError::InvalidKind`](super::FftwError::InvalidKind). In FFTW, `Hc2r` destroys its
/// input by default; RLST's real-to-real plans preserve their input unless
/// [`may_destroy_input`](super::Planner::may_destroy_input) is given, also with `Hc2r` (see
/// [`Planner::r2r`](super::Planner::r2r)).
///
/// ```
/// use rlst::fftw::R2rKind;
///
/// assert_eq!(R2rKind::Redft10.to_string(), "REDFT10");
/// assert_eq!(R2rKind::Redft10.logical_size(8), 16);
/// assert_eq!(R2rKind::Redft00.logical_size(8), 14);
/// assert_eq!(R2rKind::Rodft00.logical_size(8), 18);
/// assert_eq!(R2rKind::Dht.logical_size(8), 8);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum R2rKind {
    /// Real to halfcomplex (`FFTW_R2HC`): the real and imaginary parts of the forward DFT,
    /// `r0, r1, …, r(n/2), i((n+1)/2 − 1), …, i1`.
    R2hc,
    /// Halfcomplex to real (`FFTW_HC2R`): the backward DFT of a halfcomplex array, the
    /// inverse of [`R2hc`](R2rKind::R2hc) times n.
    Hc2r,
    /// The discrete Hartley transform (`FFTW_DHT`), its own inverse times n.
    Dht,
    /// DCT-I (`FFTW_REDFT00`), even around j = 0 and even around j = n − 1; n ≥ 2.
    Redft00,
    /// DCT-III (`FFTW_REDFT01`), even around j = 0 and odd around j = n.
    Redft01,
    /// DCT-II (`FFTW_REDFT10`), even around j = −1/2 and even around j = n − 1/2.
    Redft10,
    /// DCT-IV (`FFTW_REDFT11`), even around j = −1/2 and odd around j = n − 1/2.
    Redft11,
    /// DST-I (`FFTW_RODFT00`), odd around j = −1 and odd around j = n.
    Rodft00,
    /// DST-III (`FFTW_RODFT01`), odd around j = −1 and even around j = n − 1.
    Rodft01,
    /// DST-II (`FFTW_RODFT10`), odd around j = −1/2 and odd around j = n − 1/2.
    Rodft10,
    /// DST-IV (`FFTW_RODFT11`), odd around j = −1/2 and even around j = n − 1/2.
    Rodft11,
}

impl R2rKind {
    /// The eleven kinds, in FFTW's order.
    ///
    /// ```
    /// use rlst::fftw::R2rKind;
    ///
    /// assert_eq!(R2rKind::ALL.len(), 11);
    /// assert_eq!(R2rKind::ALL[0], R2rKind::R2hc);
    /// ```
    pub const ALL: [R2rKind; 11] = [
        R2rKind::R2hc,
        R2rKind::Hc2r,
        R2rKind::Dht,
        R2rKind::Redft00,
        R2rKind::Redft01,
        R2rKind::Redft10,
        R2rKind::Redft11,
        R2rKind::Rodft00,
        R2rKind::Rodft01,
        R2rKind::Rodft10,
        R2rKind::Rodft11,
    ];

    /// The logical size N of the kind for a dimension of extent `n` (design, Section 3 item
    /// 11): n for `R2hc`, `Hc2r` and `Dht`, 2 (n − 1) for `Redft00`, 2 (n + 1) for `Rodft00`
    /// and 2 n for the other kinds. The kind followed by its inverse multiplies the data by
    /// N. `Redft00` is not defined for n = 1, where this gives 0.
    ///
    /// ```
    /// use rlst::fftw::R2rKind;
    ///
    /// assert_eq!(R2rKind::Hc2r.logical_size(5), 5);
    /// assert_eq!(R2rKind::Rodft11.logical_size(5), 10);
    /// assert_eq!(R2rKind::Redft00.logical_size(1), 0);
    /// ```
    pub fn logical_size(self, n: usize) -> usize {
        match self {
            R2rKind::R2hc | R2rKind::Hc2r | R2rKind::Dht => n,
            R2rKind::Redft00 => 2 * n.saturating_sub(1),
            R2rKind::Rodft00 => 2 * (n + 1),
            _ => 2 * n,
        }
    }

    /// FFTW's constant for the kind (DF4: RLST's type maps to fftw-sys's internally).
    pub(crate) fn raw(self) -> fftw_sys::fftw_r2r_kind {
        match self {
            R2rKind::R2hc => fftw_sys::fftw_r2r_kind::FFTW_R2HC,
            R2rKind::Hc2r => fftw_sys::fftw_r2r_kind::FFTW_HC2R,
            R2rKind::Dht => fftw_sys::fftw_r2r_kind::FFTW_DHT,
            R2rKind::Redft00 => fftw_sys::fftw_r2r_kind::FFTW_REDFT00,
            R2rKind::Redft01 => fftw_sys::fftw_r2r_kind::FFTW_REDFT01,
            R2rKind::Redft10 => fftw_sys::fftw_r2r_kind::FFTW_REDFT10,
            R2rKind::Redft11 => fftw_sys::fftw_r2r_kind::FFTW_REDFT11,
            R2rKind::Rodft00 => fftw_sys::fftw_r2r_kind::FFTW_RODFT00,
            R2rKind::Rodft01 => fftw_sys::fftw_r2r_kind::FFTW_RODFT01,
            R2rKind::Rodft10 => fftw_sys::fftw_r2r_kind::FFTW_RODFT10,
            R2rKind::Rodft11 => fftw_sys::fftw_r2r_kind::FFTW_RODFT11,
        }
    }
}

impl fmt::Display for R2rKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            R2rKind::R2hc => "R2HC",
            R2rKind::Hc2r => "HC2R",
            R2rKind::Dht => "DHT",
            R2rKind::Redft00 => "REDFT00",
            R2rKind::Redft01 => "REDFT01",
            R2rKind::Redft10 => "REDFT10",
            R2rKind::Redft11 => "REDFT11",
            R2rKind::Rodft00 => "RODFT00",
            R2rKind::Rodft01 => "RODFT01",
            R2rKind::Rodft10 => "RODFT10",
            R2rKind::Rodft11 => "RODFT11",
        })
    }
}
