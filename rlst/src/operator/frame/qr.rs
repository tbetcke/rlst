//! Rank-adaptive QR algorithms.

use super::{ArrayFrameImpl, GenericFrameImpl};
use crate::dense::linalg::lapack::qr::{EnablePivoting, QMode};
use crate::operator::element::Element;
use crate::{
    DynArray, FrameImpl, Gemm, InnerProductSpace, Lapack, Norm, Qr, RlstError, RlstResult,
    RlstScalar,
};
use num::{Float, One, Zero};

/// Tolerances for rank-adaptive frame QR.
///
/// Stop when every remaining column residual has norm at most
/// `max(atol, rtol * max_original_column_norm)`. Discarding nonzero residuals
/// makes reconstruction approximate, in addition to floating-point rounding.
#[derive(Clone, Copy, Debug)]
pub struct FrameQrOptions<Real> {
    /// Absolute residual tolerance. Must be finite and nonnegative.
    pub atol: Real,
    /// Relative residual tolerance. Must be finite and nonnegative.
    pub rtol: Real,
}

impl<Real: Float> Default for FrameQrOptions<Real> {
    /// Use zero absolute tolerance and `100 * epsilon` relative tolerance.
    fn default() -> Self {
        Self {
            atol: Real::zero(),
            rtol: Real::from(100).unwrap() * Real::epsilon(),
        }
    }
}

impl<Real: Float> FrameQrOptions<Real> {
    fn threshold(self, scale: Real) -> RlstResult<Real> {
        if !self.atol.is_finite()
            || !self.rtol.is_finite()
            || self.atol < Real::zero()
            || self.rtol < Real::zero()
        {
            return Err(RlstError::GeneralError(
                "Frame QR tolerances must be finite and nonnegative".into(),
            ));
        }
        finite(self.atol.max(self.rtol * scale))
    }
}

/// A factorization satisfying `original[:, permutation] ≈ q * r`.
///
/// Q has `rank()` orthonormal columns and R has shape `[rank(), original.len()]`.
/// If `permutation[j] = k`, column `j` of QR reconstructs original column `k`.
/// The input is unchanged. Signs/phases and pivot ties can differ by backend.
pub struct FrameQrDecomposition<Q, Item> {
    /// Orthonormal frame, using the same storage backend as the input.
    pub q: Q,
    /// Upper-trapezoidal coefficient matrix in permuted column order.
    pub r: DynArray<Item, 2>,
    /// Zero-based permutation of all original columns.
    pub permutation: Vec<usize>,
}

impl<Q, Item> FrameQrDecomposition<Q, Item> {
    /// Number of retained orthonormal columns.
    pub fn rank(&self) -> usize {
        self.r.shape()[0]
    }
}

/// Pivoted QR for frames over numerical inner-product spaces.
pub trait FrameQr<'a>: FrameImpl<'a> + Sized
where
    Self::Space: InnerProductSpace,
    <Self::Space as crate::LinearSpace>::F: RlstScalar,
{
    /// Factor without modifying the input, truncating at the requested tolerance.
    ///
    /// Rejects invalid tolerances and non-finite norms or coefficients. Generic
    /// frames use classical Gram–Schmidt twice; array frames use LAPACK.
    fn qr_pivoted(
        &self,
        options: FrameQrOptions<<<Self::Space as crate::LinearSpace>::F as RlstScalar>::Real>,
    ) -> RlstResult<FrameQrDecomposition<Self, <Self::Space as crate::LinearSpace>::F>>;
}

fn finite<Real: Float>(value: Real) -> RlstResult<Real> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RlstError::OperationFailed(
            "Non-finite value in frame QR".into(),
        ))
    }
}

/// Two classical passes: each coefficient vector is computed before any update.
fn project_twice<'a, S: InnerProductSpace>(
    q: &GenericFrameImpl<'a, S>,
    original: &Element<S>,
) -> RlstResult<(Element<'a, S>, Vec<S::F>)>
where
    S::F: RlstScalar,
{
    let mut residual = q.space.copy_from(original);
    let mut coefficients = vec![S::F::zero(); q.len()];
    for _ in 0..2 {
        let pass: Vec<_> = q
            .elements
            .iter()
            .map(|basis| q.space.inner_product(&residual, basis))
            .collect();
        let mut projection = q.space.zero();
        for ((basis, coefficient), total) in q.elements.iter().zip(pass).zip(&mut coefficients) {
            finite(coefficient.abs())?;
            *total += coefficient;
            finite(total.abs())?;
            let scaled = q.space.scalar_mul(&coefficient, basis);
            q.space.sum_inplace(&mut projection, &scaled);
        }
        q.space.sub_inplace(&mut residual, &projection);
    }
    Ok((residual, coefficients))
}

impl<'a, S: InnerProductSpace> FrameQr<'a> for GenericFrameImpl<'a, S>
where
    S::F: RlstScalar,
{
    fn qr_pivoted(
        &self,
        options: FrameQrOptions<<S::F as RlstScalar>::Real>,
    ) -> RlstResult<FrameQrDecomposition<Self, S::F>> {
        let n = self.len();
        let mut scale = <S::F as RlstScalar>::Real::zero();
        for element in &self.elements {
            scale = Float::max(scale, finite(element.norm())?);
        }
        let threshold = options.threshold(scale)?;
        let mut q = Self::new(self.space);
        let mut permutation: Vec<_> = (0..n).collect();
        let mut r = DynArray::<S::F, 2>::from_shape([n, n]);
        while q.len() < n {
            let k = q.len();
            let mut best = k;
            let mut largest = <S::F as RlstScalar>::Real::zero();
            let mut pivot = None;
            for j in k..n {
                let (residual, coefficients) = project_twice(&q, &self.elements[permutation[j]])?;
                for (i, coefficient) in coefficients.into_iter().enumerate() {
                    r[[i, j]] = coefficient;
                }
                let norm = finite(residual.norm())?;
                if pivot.is_none() || norm > largest {
                    largest = norm;
                    best = j;
                    pivot = Some(residual);
                }
            }
            if largest <= threshold {
                break;
            }
            permutation.swap(k, best);
            for i in 0..k {
                let value = r[[i, k]];
                r[[i, k]] = r[[i, best]];
                r[[i, best]] = value;
            }
            r[[k, k]] = S::F::from_real(largest);
            let mut pivot = pivot.unwrap();
            let inverse = S::F::from_real(<S::F as RlstScalar>::Real::one() / largest);
            finite(inverse.abs())?;
            self.space.scale_inplace(&inverse, &mut pivot);
            finite(pivot.norm())?;
            q.elements.push(pivot);
        }
        let mut compact_r = DynArray::from_shape([q.len(), n]);
        for j in 0..n {
            for i in 0..q.len() {
                compact_r[[i, j]] = r[[i, j]];
            }
        }
        Ok(FrameQrDecomposition {
            q,
            r: compact_r,
            permutation,
        })
    }
}

impl<'a, Item: Lapack + Gemm> FrameQr<'a> for ArrayFrameImpl<'a, Item> {
    fn qr_pivoted(
        &self,
        options: FrameQrOptions<Item::Real>,
    ) -> RlstResult<FrameQrDecomposition<Self, Item>> {
        let [m, n] = self.matrix().shape();
        let mut scale = Item::Real::zero();
        for j in 0..n {
            let mut norm = Item::Real::zero();
            for i in 0..m {
                norm = norm.hypot(finite(self.data[[i, j]].abs())?);
            }
            scale = Float::max(scale, finite(norm)?);
        }
        let threshold = options.threshold(scale)?;
        if m == 0 || n == 0 || scale <= threshold {
            return Ok(FrameQrDecomposition {
                q: Self::new(self.space),
                r: DynArray::from_shape([0, n]),
                permutation: (0..n).collect(),
            });
        }
        let factor = self.matrix().qr(EnablePivoting::Yes)?;
        let full_r = factor.r_mat()?;
        let k = m.min(n);
        let mut rank = 0;
        // Scan each trailing column from bottom to top in O(k*n), finding the
        // smallest leading rank for which every discarded tail is small enough.
        for j in 0..n {
            let mut norm = Item::Real::zero();
            for i in (0..k).rev() {
                norm = finite(norm.hypot(finite(full_r[[i, j]].abs())?))?;
                if norm > threshold {
                    rank = rank.max(i + 1);
                }
            }
        }
        let full_q = factor.q_mat(QMode::Compact)?;
        let mut q_data = DynArray::from_shape([m, rank]);
        for j in 0..rank {
            for i in 0..m {
                finite(full_q[[i, j]].abs())?;
                q_data[[i, j]] = full_q[[i, j]];
            }
        }
        let mut r = DynArray::from_shape([rank, n]);
        for j in 0..n {
            for i in 0..rank {
                r[[i, j]] = full_r[[i, j]];
            }
        }
        let q = Self {
            space: self.space,
            data: q_data,
            len: rank,
        };
        Ok(FrameQrDecomposition {
            q,
            r,
            permutation: factor.perm(),
        })
    }
}
