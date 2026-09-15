//! The Generalized Minimal Residual Method.
//!
//! The Generalized Minimal Residual Method (GMRES) solves operator equations
//! `Ax=b` for a general operator `A` that need not be self-adjoint. It builds an
//! orthonormal Krylov basis with Arnoldi's method and minimizes the residual
//! over the resulting affine subspace.
//!
//! The Krylov basis is stored in an [`ElementFrame`]. By default this is a
//! [`GenericFrameImpl`], which owns one element per basis vector and works for
//! every inner-product space. Pass `ElementFrame::new(ArrayFrameImpl::new(rhs.space()))`
//! to [`GmresIteration::set_storage`] to keep the basis as one dense matrix
//! instead, so that orthogonalization and the solution update become dense
//! matrix-vector products.

use num::{Float, One, Zero};

use crate::{
    DynArray, ElementFrame, FrameApply, GenericFrameImpl, InnerProductSpace, Norm, NormedSpace,
    RlstScalar, SliceArray,
    abstract_operator::OperatorBase,
    operator::{abstract_operator::Operator, element::Element},
};

/// Builder and in-place iteration for the generalized minimal residual method.
///
/// GMRES applies to a general operator mapping an inner-product space to
/// itself. [`Self::run`] mutates the supplied initial guess and returns a
/// residual measure, not a convergence flag.
///
/// Without restarting, the Krylov basis grows by one element per iteration and
/// the work per iteration grows linearly, so long runs should set a restart
/// length with [`Self::set_restart`].
pub struct GmresIteration<
    'a,
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
    Storage: FrameApply<'a, Space = Space> = GenericFrameImpl<'a, Space>,
> {
    operator: &'a Operator<OpImpl>,
    rhs: &'a Element<'a, Space>,
    x: &'a mut Element<'a, Space>,
    max_iter: usize,
    restart: Option<usize>,
    tol: Scalar::Real,
    #[allow(clippy::type_complexity)]
    callable: Option<Box<dyn FnMut(usize, Scalar::Real) + 'a>>,
    #[allow(clippy::type_complexity)]
    element_callable: Option<Box<dyn FnMut(&Element<Space>, &Element<Space>) + 'a>>,
    print_debug: bool,
    storage: ElementFrame<'a, Storage>,
}

impl<
    'a,
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
> GmresIteration<'a, Scalar, Space, OpImpl, GenericFrameImpl<'a, Space>>
where
    Space: NormedSpace<Output = Scalar::Real>,
{
    /// Create a new GMRES iteration.
    ///
    /// This function takes an operator `op`, a right-hand side `rhs`, and an element `x` of the
    /// function space as starting approximation. Typically, `x` can be chosen to be the zero
    /// element of the space. Defaults are tolerance `1E-6`, 1000 iterations, and no restarting.
    ///
    /// The Krylov basis is stored in an element-backed frame. Use
    /// [`Self::set_storage`] to choose a different frame implementation.
    pub fn new(
        op: &'a Operator<OpImpl>,
        rhs: &'a Element<'a, Space>,
        x: &'a mut Element<'a, Space>,
    ) -> Self {
        Self {
            operator: op,
            rhs,
            x,
            max_iter: 1000,
            restart: None,
            tol: num::cast(1E-6).unwrap(),
            callable: None,
            element_callable: None,
            print_debug: false,
            storage: ElementFrame::new(GenericFrameImpl::new(rhs.space())),
        }
    }
}

impl<
    'a,
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
    Storage: FrameApply<'a, Space = Space>,
> GmresIteration<'a, Scalar, Space, OpImpl, Storage>
where
    Space: NormedSpace<Output = Scalar::Real>,
{
    /// Store the Krylov basis in `storage` instead of the default
    /// element-backed frame.
    ///
    /// The frame is scratch space: it is cleared at the start of every restart
    /// cycle, so elements it already holds are discarded while its allocated
    /// capacity is reused. The frame must borrow the space of the right-hand
    /// side; array-backed spaces keep the basis as one dense matrix with
    /// `set_storage(ElementFrame::new(ArrayFrameImpl::new(rhs.space())))`.
    pub fn set_storage<NewStorage: FrameApply<'a, Space = Space>>(
        self,
        storage: ElementFrame<'a, NewStorage>,
    ) -> GmresIteration<'a, Scalar, Space, OpImpl, NewStorage> {
        GmresIteration {
            operator: self.operator,
            rhs: self.rhs,
            x: self.x,
            max_iter: self.max_iter,
            restart: self.restart,
            tol: self.tol,
            callable: self.callable,
            element_callable: self.element_callable,
            print_debug: self.print_debug,
            storage,
        }
    }

    /// Set the tolerance for convergence.
    ///
    /// For a nonzero right-hand side, convergence is measured by the relative
    /// residual `||b - Ax|| / ||b||`. For a zero right-hand side, the absolute
    /// residual `||b - Ax||` is used instead. The default tolerance is 1E-6.
    pub fn set_tol(mut self, tol: Scalar::Real) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum number of iterations.
    ///
    /// The default maximum number of iterations is 1000. The limit counts
    /// Arnoldi steps across all restart cycles.
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the restart length.
    ///
    /// After `restart` Arnoldi steps the Krylov basis is discarded and the
    /// iteration starts again from the current approximation. This bounds
    /// memory and per-iteration work at the cost of slower convergence, and
    /// restarted GMRES can stagnate on problems that full GMRES solves. By
    /// default GMRES does not restart, so the basis may grow to `max_iter`
    /// elements. A restart length of zero is treated as one.
    pub fn set_restart(mut self, restart: usize) -> Self {
        self.restart = Some(restart);
        self
    }

    /// Define a callable.
    ///
    /// A callable is a function `FnMut(usize, Scalar::Real)` that is executed
    /// in each step of the iteration. The first parameter is the number of
    /// Arnoldi steps performed so far and the second parameter is the current
    /// residual, normalized as described in [`Self::set_tol`]. Both values come
    /// from the Givens recursion at no additional cost. The callable runs after
    /// each successful step, before convergence is checked. It is not called for
    /// an already-converged initial guess or a pre-update breakdown.
    ///
    /// Use [`Self::set_element_callable`] to inspect the approximation and the
    /// residual themselves.
    pub fn set_callable(mut self, callable: impl FnMut(usize, Scalar::Real) + 'a) -> Self {
        self.callable = Some(Box::new(callable));
        self
    }

    /// Define a callable that receives the current approximation and residual.
    ///
    /// A callable is a function `FnMut(&Element, &Element)` whose first
    /// parameter is the current approximation `x` and whose second parameter is
    /// the current residual `b - Ax`. It follows the same schedule as
    /// [`Self::set_callable`] and both callables can be set at the same time.
    ///
    /// Unlike conjugate gradients, GMRES does not carry the approximation and
    /// the residual along. They are reconstructed for every call, which costs
    /// one additional triangular solve, one linear combination of the Krylov
    /// basis, and one operator application per iteration. Prefer
    /// [`Self::set_callable`] when only the residual norm is needed.
    pub fn set_element_callable(
        mut self,
        callable: impl FnMut(&Element<Space>, &Element<Space>) + 'a,
    ) -> Self {
        self.element_callable = Some(Box::new(callable));
        self
    }

    /// Enable debug printing
    ///
    /// Print debug information on `stdout`.
    pub fn print_debug(mut self) -> Self {
        self.print_debug = true;
        self
    }

    /// Run the GMRES iteration.
    ///
    /// Returns the final relative residual, or the absolute residual when the
    /// right-hand side is zero. The residual is recomputed from the operator at
    /// the end of every restart cycle, so the returned value is the true
    /// residual and not the estimate from the Givens recursion. The caller
    /// determines convergence by comparing this value with the configured
    /// tolerance; exhausting `max_iter` is not otherwise signaled. An exact
    /// initial solution returns immediately. If GMRES encounters a degenerate
    /// or non-finite Hessenberg column, it terminates without applying an
    /// invalid update and returns the current residual. The callables are
    /// invoked only after successful steps.
    ///
    /// # Panics
    ///
    /// Panics if a frame supplied through [`Self::set_storage`] rejects an
    /// element of the right-hand side's space.
    pub fn run(mut self) -> Scalar::Real {
        fn print_success<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!(
                "GMRES converged in {it_count} iterations with relative residual {rel_res:+E}."
            );
        }

        fn print_fail<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!(
                "GMRES did not converge in {it_count} iterations. Final relative residual is {rel_res:+E}.",
            );
        }

        let zero_real = <Scalar::Real as Zero>::zero();
        let one_real = <Scalar::Real as One>::one();

        let rhs_norm = Norm::norm(self.rhs);

        let mut res = self.rhs.clone();
        self.operator
            .apply(-<Scalar as One>::one(), self.x, One::one(), &mut res);
        let mut residual = Norm::norm(&res);

        // Handle an exact solution before normalizing. In particular, this
        // avoids `0 / 0` for a zero right-hand side and zero initial guess.
        if residual == zero_real {
            if self.print_debug {
                print_success(0, residual);
            }
            return residual;
        }

        let use_absolute_residual = rhs_norm == zero_real;
        let mut reported_residual = report(residual, rhs_norm, use_absolute_residual);

        if reported_residual < self.tol {
            if self.print_debug {
                print_success(0, reported_residual);
            }
            return reported_residual;
        }

        let restart = self.restart.unwrap_or(self.max_iter).max(1);

        // The frame grows geometrically as basis vectors are pushed. Reserving
        // the whole restart length up front would allocate for `max_iter`
        // elements even when the iteration converges after a few steps.
        let frame = &mut self.storage;

        let mut total_iter = 0usize;

        while total_iter < self.max_iter {
            // `res` holds `b - Ax` and `residual` its norm, both for the current `x`.
            let beta = residual;

            // Columns of the Hessenberg matrix after all Givens rotations have
            // been applied. Column `k` holds `k + 1` entries, the last being the pivot.
            let mut hessenberg = Vec::<Vec<Scalar>>::new();
            // Rotated right-hand side of the least-squares problem.
            let mut g = vec![Scalar::from_real(beta), <Scalar as Zero>::zero()];
            let mut cosines = Vec::<Scalar>::new();
            let mut sines = Vec::<Scalar>::new();

            // The normalized residual is the first basis vector. `res` is
            // rebuilt from the operator at the end of the cycle.
            frame.clear();
            let mut basis_vector = res;
            basis_vector *= Scalar::from_real(one_real / beta);
            frame
                .push(&basis_vector)
                .expect("GMRES: the frame rejected a Krylov basis vector.");

            let cycle_length = restart.min(self.max_iter - total_iter);
            let mut breakdown = false;

            // Invariant: the frame holds exactly `hessenberg.len() + 1`
            // basis vectors at the top of each step, and exactly
            // `hessenberg.len()` once the cycle ends.
            for step in 0..cycle_length {
                let mut candidate = self.operator.dot(&basis_vector);
                let mut column = Self::orthogonalize(frame, &mut candidate);
                let candidate_norm = Norm::norm(&candidate);
                column.push(Scalar::from_real(candidate_norm));

                // Apply the previous rotations to the new column.
                for index in 0..step {
                    let (upper, lower) = apply_rotation(
                        cosines[index],
                        sines[index],
                        column[index],
                        column[index + 1],
                    );
                    column[index] = upper;
                    column[index + 1] = lower;
                }

                // Eliminate the subdiagonal entry of the new column.
                let (cosine, sine, pivot) = givens_rotation(column[step], column[step + 1]);
                if pivot == zero_real || !pivot.is_finite() {
                    // A degenerate column carries no usable update. Drop the
                    // basis vector it belongs to so the frame matches the
                    // Hessenberg columns.
                    frame.pop();
                    breakdown = true;
                    break;
                }
                column[step] = Scalar::from_real(pivot);
                column.pop();
                cosines.push(cosine);
                sines.push(sine);
                hessenberg.push(column);

                let (updated, carried) = apply_rotation(cosine, sine, g[step], g[step + 1]);
                g[step] = updated;
                g[step + 1] = carried;
                g.push(<Scalar as Zero>::zero());

                total_iter += 1;

                let estimate = report(g[step + 1].abs(), rhs_norm, use_absolute_residual);

                if let Some(callable) = self.callable.as_mut() {
                    callable(total_iter, estimate);
                }
                if let Some(callable) = self.element_callable.as_mut() {
                    let mut current = (*self.x).clone();
                    frame
                        .apply_into(
                            One::one(),
                            &least_squares_coefficients(&hessenberg, &g),
                            One::one(),
                            &mut current,
                        )
                        .expect("GMRES: the frame rejected the coefficient vector.");
                    let mut current_residual = self.rhs.clone();
                    self.operator.apply(
                        -<Scalar as One>::one(),
                        &current,
                        One::one(),
                        &mut current_residual,
                    );
                    callable(&current, &current_residual);
                }

                if estimate < self.tol {
                    break;
                }

                if candidate_norm == zero_real || !candidate_norm.is_finite() {
                    // The Krylov space is invariant; the current least-squares
                    // solution is the best this cycle can produce.
                    breakdown = true;
                    break;
                }

                // The last step of a cycle needs no further basis vector.
                if step + 1 < cycle_length {
                    candidate *= Scalar::from_real(one_real / candidate_norm);
                    frame
                        .push(&candidate)
                        .expect("GMRES: the frame rejected a Krylov basis vector.");
                    basis_vector = candidate;
                }
            }

            if !hessenberg.is_empty() {
                frame
                    .apply_into(
                        One::one(),
                        &least_squares_coefficients(&hessenberg, &g),
                        One::one(),
                        self.x,
                    )
                    .expect("GMRES: the frame rejected the coefficient vector.");
            }

            // Recompute the true residual rather than trusting the estimate.
            res = self.rhs.clone();
            self.operator
                .apply(-<Scalar as One>::one(), self.x, One::one(), &mut res);
            residual = Norm::norm(&res);
            reported_residual = report(residual, rhs_norm, use_absolute_residual);

            if reported_residual < self.tol {
                if self.print_debug {
                    print_success(total_iter, reported_residual);
                }
                return reported_residual;
            }

            if breakdown || hessenberg.is_empty() || residual == zero_real || !residual.is_finite()
            {
                if self.print_debug {
                    print_fail(total_iter, reported_residual);
                }
                return reported_residual;
            }
        }

        if self.print_debug {
            print_fail(total_iter, reported_residual);
        }
        reported_residual
    }

    /// Orthogonalize `candidate` against the frame and return the projection
    /// coefficients, one per basis vector.
    ///
    /// Two passes of classical Gram-Schmidt (CGS2) are used because they work
    /// on the whole frame at once through [`FrameApply::adjoint_apply`] and
    /// [`FrameApply::apply_into`], so array-backed frames run two dense
    /// matrix-vector products per pass and no basis vector is copied.
    fn orthogonalize(
        frame: &ElementFrame<'a, Storage>,
        candidate: &mut Element<Space>,
    ) -> Vec<Scalar> {
        let mut coefficients = Vec::new();
        for pass in 0..2 {
            let projection = frame
                .adjoint_apply(candidate)
                .expect("GMRES: the frame rejected a Krylov candidate.");
            frame
                .apply_into(
                    -<Scalar as One>::one(),
                    &SliceArray::from_shape(&projection, [projection.len()]),
                    One::one(),
                    candidate,
                )
                .expect("GMRES: the frame rejected the coefficient vector.");
            if pass == 0 {
                coefficients = projection;
            } else {
                for (total, correction) in coefficients.iter_mut().zip(projection) {
                    *total += correction;
                }
            }
        }
        coefficients
    }
}

/// Normalize a residual norm the way [`GmresIteration::set_tol`] describes.
fn report<Real: Float>(residual: Real, rhs_norm: Real, use_absolute_residual: bool) -> Real {
    if use_absolute_residual {
        residual
    } else {
        residual / rhs_norm
    }
}

/// Compute a Givens rotation that eliminates `lower` against `upper`.
///
/// The returned `(cosine, sine, pivot)` satisfy
/// `conj(cosine) * upper + conj(sine) * lower == pivot` and
/// `-sine * upper + cosine * lower == 0`. A zero or non-finite pivot leaves the
/// rotation as the identity and is reported to the caller through `pivot`.
fn givens_rotation<Scalar: RlstScalar>(
    upper: Scalar,
    lower: Scalar,
) -> (Scalar, Scalar, Scalar::Real) {
    let pivot = upper.abs().hypot(lower.abs());
    if pivot == <Scalar::Real as Zero>::zero() || !pivot.is_finite() {
        (<Scalar as One>::one(), <Scalar as Zero>::zero(), pivot)
    } else {
        (upper.div_real(pivot), lower.div_real(pivot), pivot)
    }
}

/// Apply the rotation defined by `cosine` and `sine` to a pair of entries.
fn apply_rotation<Scalar: RlstScalar>(
    cosine: Scalar,
    sine: Scalar,
    upper: Scalar,
    lower: Scalar,
) -> (Scalar, Scalar) {
    (
        cosine.conj() * upper + sine.conj() * lower,
        cosine * lower - sine * upper,
    )
}

/// Solve the rotated least-squares problem by back substitution.
///
/// `hessenberg` holds the rotated Hessenberg columns, each ending in its
/// nonzero pivot, and `g` the rotated right-hand side. The result is the
/// coefficient vector of the correction with respect to the Krylov basis,
/// which has one entry per column.
fn least_squares_coefficients<Scalar: RlstScalar>(
    hessenberg: &[Vec<Scalar>],
    g: &[Scalar],
) -> DynArray<Scalar, 1> {
    let columns = hessenberg.len();
    let mut coefficients = DynArray::<Scalar, 1>::from_shape([columns]);
    for row in (0..columns).rev() {
        let mut value = g[row];
        for column in row + 1..columns {
            value -= hessenberg[column][row] * coefficients[[column]];
        }
        coefficients[[row]] = value / hessenberg[row][row];
    }
    coefficients
}

#[cfg(test)]
mod test {
    use std::cell::{Cell, RefCell};

    use approx::assert_relative_eq;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        ArrayFrameImpl, ElementFrame, InnerProductSpace, LinearSpace, Norm,
        abstract_operator::OperatorBase,
        base_types::c64,
        dense::array::DynArray,
        operator::{
            abstract_operator::Operator, algorithms::gmres::GmresIteration, element::Element,
            space::zero_element,
        },
    };

    /// A space that is not array-backed, to show that GMRES needs nothing of a
    /// space beyond the inner product.
    struct TripleSpace;

    impl LinearSpace for TripleSpace {
        type F = f64;
        type Impl = [f64; 3];

        fn zero(&self) -> Element<'_, Self> {
            Element::new(self, [0.0; 3])
        }
        fn add(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self> {
            Element::new(self, std::array::from_fn(|i| x.imp()[i] + y.imp()[i]))
        }
        fn sub(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self> {
            Element::new(self, std::array::from_fn(|i| x.imp()[i] - y.imp()[i]))
        }
        fn scalar_mul(&self, scalar: &f64, x: &Element<Self>) -> Element<'_, Self> {
            Element::new(self, x.imp().map(|value| value * *scalar))
        }
        fn neg(&self, x: &Element<Self>) -> Element<'_, Self> {
            self.scalar_mul(&-1.0, x)
        }
        fn sum_inplace(&self, x: &mut Element<Self>, y: &Element<Self>) {
            for i in 0..3 {
                x.imp_mut()[i] += y.imp()[i];
            }
        }
        fn sub_inplace(&self, x: &mut Element<Self>, y: &Element<Self>) {
            for i in 0..3 {
                x.imp_mut()[i] -= y.imp()[i];
            }
        }
        fn scale_inplace(&self, scalar: &f64, x: &mut Element<Self>) {
            for value in x.imp_mut() {
                *value *= *scalar;
            }
        }
        fn copy_from(&self, x: &Element<Self>) -> Element<'_, Self> {
            Element::new(self, *x.imp())
        }
    }

    impl InnerProductSpace for TripleSpace {
        fn inner_product(&self, x: &Element<Self>, y: &Element<Self>) -> f64 {
            (0..3).map(|i| x.imp()[i] * y.imp()[i]).sum()
        }
    }

    /// A non-symmetric operator on [`TripleSpace`].
    struct Shift;

    impl OperatorBase for Shift {
        type Domain = TripleSpace;
        type Range = TripleSpace;

        fn domain(&self) -> &Self::Domain {
            &TripleSpace
        }
        fn range(&self) -> &Self::Range {
            &TripleSpace
        }
        fn apply(
            &self,
            alpha: f64,
            x: &Element<Self::Domain>,
            beta: f64,
            y: &mut Element<Self::Range>,
        ) {
            let x = *x.imp();
            let result = [
                3.0 * x[0] + x[1],
                -x[0] + 4.0 * x[1] + 2.0 * x[2],
                x[1] + 5.0 * x[2],
            ];
            for i in 0..3 {
                y.imp_mut()[i] = alpha * result[i] + beta * y.imp()[i];
            }
        }
    }

    /// A well-conditioned non-symmetric matrix with a fixed random perturbation.
    fn non_symmetric_matrix(dim: usize) -> DynArray<f64, 2> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut mat = DynArray::<f64, 2>::from_shape([dim, dim]);
        for row in 0..dim {
            for col in 0..dim {
                mat[[row, col]] = rng.random_range(-0.1..=0.1);
            }
            mat[[row, row]] += 4.0;
        }
        mat
    }

    #[test]
    fn zero_rhs_and_zero_guess_converges_without_iteration() {
        let mut mat = DynArray::<f64, 2>::from_shape([2, 2]);
        mat[[0, 0]] = 1.0;
        mat[[1, 1]] = 1.0;
        let op = Operator::from(&mat);
        let rhs = zero_element(op.range());
        let mut x = zero_element(op.domain());
        let callback_count = Cell::new(0);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_callable(|_, _| callback_count.set(callback_count.get() + 1))
            .run();

        assert_eq!(residual, 0.0);
        assert!(residual.is_finite());
        assert_eq!(callback_count.get(), 0);
    }

    #[test]
    fn zero_rhs_uses_absolute_residual() {
        let mut mat = DynArray::<f64, 2>::from_shape([2, 2]);
        mat[[0, 0]] = 1.0;
        mat[[1, 1]] = 1.0;
        let op = Operator::from(&mat);
        let rhs = zero_element(op.range());
        let mut x = zero_element(op.domain());
        x.imp_mut()[[0]] = 2.0;
        x.imp_mut()[[1]] = -1.0;
        let callback_count = Cell::new(0);
        let final_x_0 = Cell::new(f64::NAN);
        let final_x_1 = Cell::new(f64::NAN);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-12)
            .set_callable(|_, _| callback_count.set(callback_count.get() + 1))
            .set_element_callable(|current, _| {
                final_x_0.set(current.imp()[[0]]);
                final_x_1.set(current.imp()[[1]]);
            })
            .run();

        assert!(residual.is_finite());
        assert!(residual < 1E-12);
        assert_eq!(callback_count.get(), 1);
        assert_relative_eq!(final_x_0.get(), 0.0, epsilon = 1E-12);
        assert_relative_eq!(final_x_1.get(), 0.0, epsilon = 1E-12);
    }

    #[test]
    fn non_symmetric_system_is_solved_exactly() {
        // A skew-symmetric operator. Conjugate gradients cannot solve this.
        let mut mat = DynArray::<f64, 2>::from_shape([2, 2]);
        mat[[0, 1]] = 1.0;
        mat[[1, 0]] = -1.0;
        let op = Operator::from(&mat);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut()[[0]] = 1.0;
        rhs.imp_mut()[[1]] = 2.0;
        let mut x = zero_element(op.domain());
        let final_x_0 = Cell::new(f64::NAN);
        let final_x_1 = Cell::new(f64::NAN);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-12)
            .set_element_callable(|current, _| {
                final_x_0.set(current.imp()[[0]]);
                final_x_1.set(current.imp()[[1]]);
            })
            .run();

        assert!(residual < 1E-12);
        assert_relative_eq!(final_x_0.get(), -2.0, epsilon = 1E-10);
        assert_relative_eq!(final_x_1.get(), 1.0, epsilon = 1E-10);
    }

    #[test]
    fn converges_within_dimension_steps() {
        let dim = 5;
        let mat = non_symmetric_matrix(dim);
        let op = Operator::from(&mat);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut().fill_from_equally_distributed(&mut rng);
        let mut x = zero_element(op.domain());
        let iterations = Cell::new(0);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-13)
            .set_callable(|it, _| iterations.set(it))
            .run();

        assert!(residual < 1E-13);
        assert!(
            iterations.get() <= dim,
            "GMRES needed {} iterations for a {dim}-dimensional problem.",
            iterations.get()
        );
    }

    #[test]
    fn matrix_storage_matches_the_default_element_storage() {
        let dim = 10;
        let tol = 1E-10;
        let mat = non_symmetric_matrix(dim);
        let op = Operator::from(&mat);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut().fill_from_equally_distributed(&mut rng);

        let mut x_default = zero_element(op.domain());
        let default_solution = RefCell::new(Vec::<f64>::new());
        let default_residual = GmresIteration::new(&op, &rhs, &mut x_default)
            .set_tol(tol)
            .set_element_callable(|current, _| {
                let mut solution = default_solution.borrow_mut();
                solution.clear();
                solution.extend(current.imp().data().unwrap().iter().copied());
            })
            .run();

        let mut x_dense = zero_element(op.domain());
        let dense_solution = RefCell::new(Vec::<f64>::new());
        let dense_residual = GmresIteration::new(&op, &rhs, &mut x_dense)
            .set_storage(ElementFrame::new(ArrayFrameImpl::new(rhs.space())))
            .set_tol(tol)
            .set_element_callable(|current, _| {
                let mut solution = dense_solution.borrow_mut();
                solution.clear();
                solution.extend(current.imp().data().unwrap().iter().copied());
            })
            .run();

        assert!(default_residual < tol);
        assert!(dense_residual < tol);
        let default_solution = default_solution.borrow();
        let dense_solution = dense_solution.borrow();
        assert_eq!(dense_solution.len(), dim);
        assert_eq!(default_solution.len(), dim);
        for (dense, default) in dense_solution.iter().zip(default_solution.iter()) {
            assert_relative_eq!(*dense, *default, epsilon = 1E-8);
        }
    }

    #[test]
    fn supplied_storage_is_cleared_before_use() {
        let dim = 4;
        let tol = 1E-10;
        let mat = non_symmetric_matrix(dim);
        let op = Operator::from(&mat);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut()[[0]] = 1.0;
        let mut x = zero_element(op.domain());

        let mut storage = ElementFrame::new(ArrayFrameImpl::new(rhs.space()));
        storage.push(&rhs.space().zero()).unwrap();
        storage.push(&rhs.space().zero()).unwrap();
        assert_eq!(storage.len(), 2);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_storage(storage)
            .set_tol(tol)
            .run();

        assert!(residual < tol);
    }

    #[test]
    fn solves_on_a_space_without_array_storage() {
        let op = Operator::new(Shift);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut()[0] = 1.0;
        rhs.imp_mut()[2] = -2.0;
        let mut x = zero_element(op.domain());
        let solution = Cell::new([f64::NAN; 3]);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-12)
            .set_element_callable(|current, _| solution.set(*current.imp()))
            .run();

        assert!(residual < 1E-12);
        // `run` recomputes the residual from the operator; check the captured
        // approximation against the equations independently of that.
        let mut applied = zero_element(op.range());
        Shift.apply(
            1.0,
            &Element::new(&TripleSpace, solution.get()),
            0.0,
            &mut applied,
        );
        assert!((&applied - &rhs).norm() < 1E-10);
    }

    #[test]
    fn restart_converges() {
        let dim = 10;
        let tol = 1E-8;
        let mat = non_symmetric_matrix(dim);
        let op = Operator::from(&mat);
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut().fill_from_equally_distributed(&mut rng);
        let mut x = zero_element(op.domain());
        let estimates = RefCell::new(Vec::<f64>::new());

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_restart(2)
            .set_tol(tol)
            .set_callable(|_, estimate| estimates.borrow_mut().push(estimate))
            .print_debug()
            .run();

        assert!(residual < tol);
        let estimates = estimates.borrow();
        // Full GMRES converges within `dim` steps; a restart length of two
        // needs more cycles, so the total must exceed the restart length.
        assert!(estimates.len() > 2);
        assert!(estimates.iter().all(|value| value.is_finite()));

        // Restarting with matrix storage reaches the same tolerance.
        let mut x_dense = zero_element(op.domain());
        let dense_residual = GmresIteration::new(&op, &rhs, &mut x_dense)
            .set_storage(ElementFrame::new(ArrayFrameImpl::new(rhs.space())))
            .set_restart(2)
            .set_tol(tol)
            .run();
        assert!(dense_residual < tol);
    }

    #[test]
    fn max_iter_bounds_the_total_number_of_steps() {
        let dim = 10;
        let mat = non_symmetric_matrix(dim);
        let op = Operator::from(&mat);
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut().fill_from_equally_distributed(&mut rng);
        let last_count = Cell::new(0);
        let calls = Cell::new(0);

        for restart in [None, Some(2)] {
            let mut x = zero_element(op.domain());
            let mut gmres = GmresIteration::new(&op, &rhs, &mut x)
                .set_max_iter(5)
                .set_tol(1E-300)
                .set_callable(|count, _| {
                    last_count.set(count);
                    calls.set(calls.get() + 1);
                });
            if let Some(restart) = restart {
                gmres = gmres.set_restart(restart);
            }
            let residual = gmres.run();

            assert!(residual.is_finite());
            assert_eq!(calls.get(), 5, "restart {restart:?}");
            assert_eq!(last_count.get(), 5, "restart {restart:?}");
            calls.set(0);
        }
    }

    #[test]
    fn lucky_breakdown_terminates_cleanly() {
        // The right-hand side is an eigenvector, so the Krylov space is
        // one-dimensional and the first step is already exact.
        let dim = 4;
        let mut mat = DynArray::<f64, 2>::from_shape([dim, dim]);
        for index in 0..dim {
            mat[[index, index]] = (index + 1) as f64;
        }
        let op = Operator::from(&mat);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut()[[2]] = 1.0;
        let mut x = zero_element(op.domain());
        let callback_count = Cell::new(0);
        let final_x_2 = Cell::new(f64::NAN);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-13)
            .set_callable(|_, _| callback_count.set(callback_count.get() + 1))
            .set_element_callable(|current, _| final_x_2.set(current.imp()[[2]]))
            .run();

        assert!(residual.is_finite());
        assert!(residual < 1E-13);
        assert_eq!(callback_count.get(), 1);
        assert_relative_eq!(final_x_2.get(), 1.0 / 3.0, epsilon = 1E-12);
    }

    #[test]
    fn complex_system_is_solved() {
        let mut mat = DynArray::<c64, 2>::from_shape([2, 2]);
        mat[[0, 0]] = c64::new(2.0, 1.0);
        mat[[0, 1]] = c64::new(0.0, 1.0);
        mat[[1, 0]] = c64::new(1.0, 0.0);
        mat[[1, 1]] = c64::new(3.0, -1.0);
        let rhs_0 = c64::new(1.0, -2.0);
        let rhs_1 = c64::new(-1.0, 0.5);
        let op = Operator::from(&mat);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut()[[0]] = rhs_0;
        rhs.imp_mut()[[1]] = rhs_1;
        let mut x = zero_element(op.domain());
        let final_x_0 = Cell::new(c64::new(0.0, 0.0));
        let final_x_1 = Cell::new(c64::new(0.0, 0.0));

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-13)
            .set_element_callable(|current, _| {
                final_x_0.set(current.imp()[[0]]);
                final_x_1.set(current.imp()[[1]]);
            })
            .run();

        assert!(residual < 1E-13);

        // Verify the solution directly against the original system.
        let x_0 = final_x_0.get();
        let x_1 = final_x_1.get();
        let residual_0 = rhs_0 - (mat[[0, 0]] * x_0 + mat[[0, 1]] * x_1);
        let residual_1 = rhs_1 - (mat[[1, 0]] * x_0 + mat[[1, 1]] * x_1);
        assert!(residual_0.norm() < 1E-12);
        assert!(residual_1.norm() < 1E-12);
    }

    #[test]
    fn element_callable_sees_true_residual() {
        let dim = 5;
        let mat = non_symmetric_matrix(dim);
        let op = Operator::from(&mat);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut().fill_from_equally_distributed(&mut rng);
        let rhs_values: Vec<f64> = (0..dim).map(|index| rhs.imp()[[index]]).collect();
        let mut x = zero_element(op.domain());
        let calls = Cell::new(0);
        let worst_mismatch = Cell::new(0.0f64);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-13)
            .set_element_callable(|current, current_residual| {
                calls.set(calls.get() + 1);
                for row in 0..dim {
                    let mut value = rhs_values[row];
                    for col in 0..dim {
                        value -= mat[[row, col]] * current.imp()[[col]];
                    }
                    let mismatch = (value - current_residual.imp()[[row]]).abs();
                    worst_mismatch.set(worst_mismatch.get().max(mismatch));
                }
            })
            .run();

        assert!(residual < 1E-13);
        assert!(calls.get() > 0);
        assert!(worst_mismatch.get() < 1E-12);
    }

    #[test]
    fn singular_operator_does_not_update_solution() {
        let mat = DynArray::<f64, 2>::from_shape([2, 2]);
        let op = Operator::from(&mat);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut()[[0]] = 1.0;
        let mut x = zero_element(op.domain());
        x.imp_mut()[[0]] = 3.0;
        x.imp_mut()[[1]] = 4.0;
        let callback_count = Cell::new(0);

        let residual = GmresIteration::new(&op, &rhs, &mut x)
            .set_callable(|_, _| callback_count.set(callback_count.get() + 1))
            .set_element_callable(|_, _| callback_count.set(callback_count.get() + 1))
            .run();

        assert_eq!(residual, 1.0);
        assert_eq!(callback_count.get(), 0);
    }
}
