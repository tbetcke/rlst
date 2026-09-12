//! Traits for matrix decompositions and solving linear systems.

use num::Zero;

use itertools::Itertools;

use crate::{
    base_types::{RlstResult, UpLo},
    dense::{
        array::DynArray,
        linalg::lapack::{
            eigenvalue_decomposition::EigMode,
            generalized_schur::QzDecomposition,
            ldlt::LdlDecomposition,
            lq::LqDecomposition,
            lu::LuDecomposition,
            pseudo_inverse::PInv,
            qr::{EnablePivoting, QrDecomposition},
            singular_value_decomposition::SvdMode,
            sylvester::SylvesterSolution,
            symmeig::SymmEigMode,
        },
    },
    traits::{base_operations::EvaluateArray, rlst_num::RlstScalar},
};

use super::{base::Gemm, lapack::Lapack};
use crate::dense::linalg::lapack::interface::trsyl::TrsylSign;

/// Compute the matrix inverse.
pub trait Inverse {
    /// The item type of the inverse.
    type Output;

    /// Compute the inverse of a square matrix.
    ///
    /// The implementation works on a copy. A nonsquare input panics; a
    /// singular matrix is reported as a LAPACK error.
    fn inverse(&self) -> RlstResult<Self::Output>;
}

/// Compute the LU decomposition of a matrix.
///
/// The LU decomposition of an `m times n` matrix `A` is defined as `A = P * L * U`, where:
/// - `A` is the original matrix,
/// - `P` is a permutation matrix of dimension `m x m`,
/// - `L` is a lower triangular matrix of dimension `m x k` with unit diagonal,
/// - `U` is an upper triangular matrix of dimension `k x n`.
///
/// Here, `k = min(m, n)`.
pub trait Lu {
    /// Item type of the LU decomposition.
    type Item: Lapack;
    /// Compute the pivoted LU decomposition of a matrix.
    ///
    /// The input is copied. LAPACK failures, including a zero factor pivot,
    /// are returned as errors.
    fn lu(&self) -> RlstResult<LuDecomposition<Self::Item>>;
}

/// Compute a pivoted LDL factorization of a symmetric or Hermitian matrix.
///
/// The matrix must be square. Only the triangle selected by [`UpLo`] is read;
/// the packed factors and pivots are retained for solving indefinite systems.
pub trait Ldl {
    /// Item type of the LDL decomposition.
    type Item: Lapack;
    /// Factor a copied matrix, using the selected triangle.
    ///
    /// # Panics
    /// Panics when the matrix is not square.
    fn ldl(&self, uplo: UpLo) -> RlstResult<LdlDecomposition<Self::Item>>;
}

/// Compute a QR decomposition, optionally with column pivoting.
///
/// For `k = min(m, n)`, the factors satisfy `A P = Q R`, with compact Q of
/// shape `m x k`, R of shape `k x n`, and P of shape `n x n`.
pub trait Qr {
    /// Item type of the QR decomposition.
    type Item: Lapack;

    /// Factor a copy of the matrix with the requested pivoting strategy.
    fn qr(&self, pivoting: EnablePivoting) -> RlstResult<QrDecomposition<Self::Item>>;
}

/// Compute an LQ decomposition without pivoting.
pub trait Lq {
    /// Item type of the LQ decomposition.
    type Item: Lapack;

    /// Compute packed factors satisfying `A = L Q`.
    ///
    /// For an `m` by `n` matrix, the decomposition can produce an `m` by
    /// `min(m, n)` lower-trapezoidal L and either compact or full Q.
    fn lq(&self) -> RlstResult<LqDecomposition<Self::Item>>;
}

/// Compute the symmetric eigenvalue decomposition of a matrix.
pub trait SymmEig {
    /// Item type of the symmetric eigenvalue decomposition.
    type Item: Lapack;

    /// Compute ascending eigenvalues of a square real symmetric or complex
    /// Hermitian matrix, reading its upper triangle.
    fn eigenvaluesh(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Real, 1>> {
        Ok(self.eigh(UpLo::Upper, SymmEigMode::EigenvaluesOnly)?.0)
    }

    /// Compute ascending eigenvalues and, when requested, eigenvectors.
    ///
    /// The matrix must be square and only the triangle selected by `uplo` is
    /// read. The input is copied. LAPACK failures are returned as errors.
    #[allow(clippy::type_complexity)]
    fn eigh(
        &self,
        uplo: UpLo,
        mode: SymmEigMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        Option<DynArray<Self::Item, 2>>,
    )>;
}

/// Generalized symmetric eigenvalue decomposition type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralizedSymmEigType {
    /// Solve `A x = lambda B x`.
    AxEqualLambdaBx = 1,
    /// Solve `A B x = lambda x`.
    ABxEqualLambdaX = 2,
    /// Solve `B A x = lambda x`.
    BAxEqualLambdaX = 3,
}

/// Compute the generalized symmetric eigenvalue decomposition of a matrix.
///
/// Depending on the `itype`, the decomposition solves one of the following problems:
///
/// 1. `A x = lambda B x`
/// 2. `A B x = lambda x`
/// 3. `B A x = lambda x`
///
/// where `lambda` are the generalized eigenvalues and `x` are the corresponding eigenvectors. The
/// matrix `A` is assumed to be symmetric (or Hermitian) and the matrix `B` is assumed to be positive definite.
pub trait GeneralizedSymmEig<B> {
    /// Item type of the generalized symmetric eigenvalue decomposition.
    type Item: Lapack;

    /// Compute the ascending real eigenvalues of a definite generalized
    /// symmetric/Hermitian eigenproblem.
    fn eigenvaluesh_generalized(
        &self,
        b: &B,
        uplo: UpLo,
        itype: GeneralizedSymmEigType,
    ) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Real, 1>> {
        Ok(self
            .eigh_generalized(b, uplo, SymmEigMode::EigenvaluesOnly, itype)?
            .0)
    }

    /// Compute generalized eigenvalues and, when requested, eigenvectors.
    ///
    /// A and B must be equally sized square matrices; A must be
    /// symmetric/Hermitian and B Hermitian positive definite. Only `uplo` is
    /// read from each input. Both inputs are copied. Failure to factor B or to
    /// converge is returned as a LAPACK error.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn eigh_generalized(
        &self,
        b: &B,
        uplo: UpLo,
        mode: SymmEigMode,
        itype: GeneralizedSymmEigType,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        Option<DynArray<Self::Item, 2>>,
    )>;
}

/// Compute the eigenvalue decomposition of a matrix.
pub trait EigenvalueDecomposition {
    /// The item type of the matrix.
    type Item: Lapack;

    /// Return the complex eigenvalues of a square matrix in unspecified order.
    fn eigenvalues(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Complex, 1>>;

    /// Compute the Schur decomposition of the matrix.
    /// Returns a tuple containing:
    /// - A block-upper triangular matrix `T`. The diagonal blocks are 1x1 or 2x2.
    ///
    /// and encode the eigenvalues of the matrix.
    /// - A unitary matrix `Z` such that `A = Z * T * Z^H`, where `Z^H` is the conjugate transpose
    #[allow(clippy::type_complexity)]
    fn schur(&self) -> RlstResult<(DynArray<Self::Item, 2>, DynArray<Self::Item, 2>)>;

    /// Compute the eigenvalues and eigenvectors of the matrix.
    ///
    /// The function returns a tuple `(lam, v, w)` containing:
    /// - A vector `lam` of eigenvalues.
    /// - An optional matrix `v` of right eigenvectors.
    /// - An optional matrix `w` of left eigenvectors.
    #[allow(clippy::type_complexity)]
    fn eig(
        &self,
        mode: EigMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Complex, 1>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
    )>;
}

/// Compute the generalized eigenvalue decomposition of a matrix.
pub trait GeneralizedEigenvalueDecomposition<B> {
    /// The item type of the matrix.
    type Item: Lapack;

    /// Return eigenvalues of the square pencil `(A, B)` in unspecified order.
    ///
    /// This divides homogeneous LAPACK eigenvalues by their beta values; a
    /// singular B can therefore produce infinities or NaNs.
    fn eigenvalues_generalized(
        &self,
        b: &B,
    ) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Complex, 1>> {
        let (lam, _, _) = self.eig_generalized(b, EigMode::EigenvaluesOnly)?;
        Ok(lam)
    }

    /// Compute eigenvalues and selected eigenvectors of `(A, B)`.
    ///
    /// Returns `(lam, vr, vl)`. Columns of `vr` satisfy `A vr = B vr lam`;
    /// columns of `vl` satisfy `vl^H A = lam vl^H B`. A and B must be equally
    /// sized square matrices and are copied. Values and vectors are complex
    /// even for real inputs, and their order is unspecified. Since homogeneous
    /// alpha/beta pairs are divided, singular B can yield infinities or NaNs.
    #[allow(clippy::type_complexity)]
    fn eig_generalized(
        &self,
        b: &B,
        mode: EigMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Complex, 1>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
    )>;
}

/// Compute an unsorted generalized Schur (QZ) decomposition of a matrix pencil.
pub trait GeneralizedSchur<B> {
    /// The scalar type of the matrix pencil.
    type Item: Lapack;

    /// Compute factors satisfying `A = Q S Z^H` and `B = Q T Z^H`.
    ///
    /// A and B must be equally sized square matrices. Generalized eigenvalues
    /// are returned in homogeneous `(alpha, beta)` form by the decomposition
    /// object. Their order is unspecified, and zero `beta` values are retained
    /// rather than divided through.
    fn qz(&self, b: &B) -> RlstResult<QzDecomposition<Self::Item>>;
}

/// Solve an ordinary dense Sylvester equation.
pub trait SylvesterSolve<B, C> {
    /// The scalar type of the coefficient matrices.
    type Item: Lapack + Gemm;

    /// Solve `A X + sign X B = scale C` without automatically unscaling X.
    ///
    /// A and B must be square (of possibly different orders), and C must have
    /// shape `A.nrows()` by `B.nrows()`. The result reports both LAPACK's
    /// overflow-avoiding scale and whether close/common eigenvalues were
    /// perturbed; a perturbed result solves a nearby problem.
    fn solve_sylvester(
        &self,
        b: &B,
        c: &C,
        sign: TrsylSign,
    ) -> RlstResult<SylvesterSolution<Self::Item>>;
}

/// Compute the singular value decomposition of a matrix.
pub trait SingularValueDecomposition {
    /// The item type of the matrix.
    type Item: Lapack + Gemm;

    /// Compute singular values in descending order.
    fn singular_values(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Real, 1>>;

    /// Compute the singular value decomposition of a matrix.
    ///
    /// The function returns a tuple containing:
    /// - A vector of singular values.
    /// - A matrix `U` containing the left singular vectors.
    /// - A matrix `Vh` containing the right singular vectors as rows.
    #[allow(clippy::type_complexity)]
    fn svd(
        &self,
        mode: SvdMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        DynArray<Self::Item, 2>,
        DynArray<Self::Item, 2>,
    )>;

    /// Compute the truncated singular value decomposition of a matrix.
    ///
    /// **Arguments:**
    /// - `max_singular_values`: Maximum number of singular values to compute. If `None`, all
    ///   singular values are computed.
    /// - `tol`: Relative tolerance for truncation. Singular values smaller or equal to `tol *
    ///   s[0]`, where `s[0]` is the largest singular value, will be discarded. Zero singular values
    ///   are always discarded.
    ///
    /// Returns a tuple containing:
    /// - A vector of singular values
    /// - A matrix `U` containing the left singular vectors.
    /// - A matrix `Vh` containing the right singular vectors as rows.
    #[allow(clippy::type_complexity)]
    fn svd_truncated(
        &self,
        max_singular_values: Option<usize>,
        tol: Option<<Self::Item as RlstScalar>::Real>,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        DynArray<Self::Item, 2>,
        DynArray<Self::Item, 2>,
    )> {
        let (s, u, vh) = self.svd(SvdMode::Compact)?;

        let nvalues = std::cmp::min(
            match max_singular_values {
                Some(n) => n,
                None => s.len(),
            },
            s.len(),
        );

        let tol = match tol {
            Some(t) => t,
            None => <<Self::Item as RlstScalar>::Real as Zero>::zero(),
        };

        let count = match s
            .iter_value()
            .take(nvalues)
            .find_position(|&elem| elem <= tol * s[[0]])
        {
            Some((index, _)) => index,
            None => nvalues,
        };

        let s = s.into_subview([0], [count]).eval();
        let u = u.r().into_subview([0, 0], [u.shape()[0], count]).eval();
        let vh = vh.r().into_subview([0, 0], [count, vh.shape()[1]]).eval();

        Ok((s, u, vh))
    }

    /// Compute an SVD-based, optionally truncated pseudo-inverse.
    ///
    /// If an integer for `max_singular_values` is provided no more than that number of
    /// singular values is contained in the pseudo-inverse. The optional parameter `tol` specifies a relative
    /// cut-off tolerance for the smallest singular value contained in the pseudo-inverse. The parameter `max_singular_values`
    /// takes precedence in the sense that if it is specified the number of singular values can never exceed the given value
    /// irrespective of `tol`.
    fn pseudo_inverse(
        &self,
        max_singular_values: Option<usize>,
        tol: Option<<Self::Item as RlstScalar>::Real>,
    ) -> RlstResult<PInv<Self::Item>>
    where
        <Self::Item as RlstScalar>::Real: Into<Self::Item>,
    {
        let (s, u, vh) = self.svd_truncated(max_singular_values, tol)?;

        Ok(PInv::new(
            s.into_type().eval(),
            u.conj().transpose().eval(),
            vh.conj().transpose().eval(),
        ))
    }
}

/// Generic trait for solving square or rectangular linear systems.
pub trait Solve<Rhs> {
    /// The output type of the solver.
    type Output;

    /// Solve `A x = b`, or a rectangular system in the least-squares sense.
    ///
    /// Dense implementations accept vector and matrix right-hand sides whose
    /// first dimension equals the row count of A. Shape mismatches panic;
    /// factorization or rank failures are returned as errors.
    fn solve(&self, rhs: &Rhs) -> RlstResult<Self::Output>;
}

/// Cholesky decomposition for positive definite matrices.
pub trait Cholesky {
    /// Item type of the array.
    type Item;

    /// Compute the Cholesky factor of a square Hermitian positive-definite matrix.
    ///
    /// **Arguments:**
    /// - `uplo`: Selects the triangle to read and the triangular factor to return.
    ///
    /// A nonsquare input panics; a non-positive pivot is returned as an error.
    fn cholesky(&self, uplo: UpLo) -> RlstResult<DynArray<Self::Item, 2>>;
}

/// Cholesky solver for positive definite systems.
pub trait CholeskySolve<Rhs> {
    /// The output type of the Cholesky solver.
    type Output;

    /// Solve a Hermitian positive-definite system by Cholesky factorization.
    ///
    /// `uplo` selects the triangle read from the coefficient matrix. The right
    /// hand side must have a matching first dimension. Shape mismatches panic;
    /// a non-positive pivot is returned as an error.
    fn cholesky_solve(&self, uplo: UpLo, rhs: &Rhs) -> RlstResult<Self::Output>;
}

/// Solve a triangular system of linear equations.
pub trait SolveTriangular<Rhs> {
    /// The output type of the triangular solver.
    type Output;

    /// Solve a triangular system using only the triangle selected by `uplo`.
    ///
    /// The coefficient matrix must be square. Dense implementations accept a
    /// vector or matrix right-hand side with matching first dimension; shape
    /// mismatches panic.
    fn solve_triangular(&self, uplo: UpLo, rhs: &Rhs) -> RlstResult<Self::Output>;
}
