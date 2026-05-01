//! LAPACK interface for RLST.

use crate::{
    dense::linalg::lapack::interface::{
        ev::Ev, gees::Gees, geev::Geev, gels::Gels, geqp3::Geqp3, geqrf::Geqrf, gesdd::Gesdd,
        gesvd::Gesvd, getrf::Getrf, getri::Getri, getrs::Getrs, ggev::Ggev, mqr::Mqr, orgqr::Orgqr,
        posv::Posv, potrf::Potrf, sygv::Sygv, sytrf::Sytrf, sytrs::Sytrs, trsm::Trsm,
    },
    traits::rlst_num::RlstScalar,
};

/// Trait for types that implement LAPACK functionality.
pub trait Lapack:
    Ev
    + Gees
    + Geev
    + Gels
    + Geqp3
    + Geqrf
    + Gesdd
    + Gesvd
    + Ggev
    + Getrf
    + Getri
    + Getrs
    + Mqr
    + Orgqr
    + Potrf
    + Posv
    + Sygv
    + Sytrf
    + Sytrs
    + Trsm
    + RlstScalar
{
}

impl<T> Lapack for T where
    T: Ev
        + Gees
        + Geev
        + Gels
        + Geqp3
        + Geqrf
        + Gesdd
        + Gesvd
        + Ggev
        + Getrf
        + Getri
        + Getrs
        + Mqr
        + Orgqr
        + Potrf
        + Posv
        + Sygv
        + Sytrf
        + Sytrs
        + Trsm
        + RlstScalar
{
}
