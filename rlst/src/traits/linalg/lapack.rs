//! LAPACK interface for RLST.

use crate::{
    dense::linalg::lapack::interface::{
        ev::Ev, gees::Gees, geev::Geev, gelqf::Gelqf, gels::Gels, geqp3::Geqp3, geqrf::Geqrf,
        gesdd::Gesdd, gesvd::Gesvd, getrf::Getrf, getri::Getri, getrs::Getrs, gges::Gges,
        ggev::Ggev, mlq::Mlq, mqr::Mqr, orglq::Orglq, orgqr::Orgqr, posv::Posv, potrf::Potrf,
        sygv::Sygv, sytrf::Sytrf, sytrs::Sytrs, trsm::Trsm, trsyl::Trsyl,
    },
    traits::rlst_num::RlstScalar,
};

/// Trait for types that implement LAPACK functionality.
pub trait Lapack:
    Ev
    + Gees
    + Geev
    + Gelqf
    + Gels
    + Geqp3
    + Geqrf
    + Gesdd
    + Gesvd
    + Gges
    + Ggev
    + Getrf
    + Getri
    + Getrs
    + Mlq
    + Mqr
    + Orglq
    + Orgqr
    + Potrf
    + Posv
    + Sygv
    + Sytrf
    + Sytrs
    + Trsm
    + Trsyl
    + RlstScalar
{
}

impl<T> Lapack for T where
    T: Ev
        + Gees
        + Geev
        + Gelqf
        + Gels
        + Geqp3
        + Geqrf
        + Gesdd
        + Gesvd
        + Gges
        + Ggev
        + Getrf
        + Getri
        + Getrs
        + Mlq
        + Mqr
        + Orglq
        + Orgqr
        + Potrf
        + Posv
        + Sygv
        + Sytrf
        + Sytrs
        + Trsm
        + Trsyl
        + RlstScalar
{
}
