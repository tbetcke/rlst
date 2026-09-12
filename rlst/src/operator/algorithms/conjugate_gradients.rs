//! The Conjugate Gradient Method.
//!
//! The Conjugate Gradient Method (CG) solves operator equations `Ax=b`, where
//! `A` is a symmetric positive definite operator.

use std::ops::Mul;

use num::{Float, One, Zero};

use crate::{
    Inner, InnerProductSpace, Norm, NormedSpace, RlstScalar, Sqrt,
    abstract_operator::OperatorBase,
    operator::{abstract_operator::Operator, element::Element},
};

/// Iteration for CG
pub struct CgIteration<
    'a,
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
> {
    operator: &'a Operator<OpImpl>,
    rhs: &'a Element<'a, Space>,
    x: &'a mut Element<'a, Space>,
    max_iter: usize,
    tol: Scalar::Real,
    #[allow(clippy::type_complexity)]
    callable: Option<Box<dyn FnMut(&Element<Space>, &Element<Space>) + 'a>>,
    print_debug: bool,
}

impl<
    'a,
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
> CgIteration<'a, Scalar, Space, OpImpl>
where
    Space: NormedSpace<Output = Scalar::Real>,
{
    /// Create a new CG iteration.
    ///
    /// This function takes an operator `op`, a right-hand side `rhs`, and an element `x` of the
    /// function space as starting approximation. Typically, `x` can be chosen to be the zero
    /// element of the space.
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
            tol: num::cast(1E-6).unwrap(),
            callable: None,
            print_debug: false,
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
    /// The default maximum number of iterations is 1000.
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Define a callable.
    ///
    /// A callable is a function `FnMut(&Element, &Element)` that is
    /// executed in each step of the iteration. The first parameter is
    /// the current approximation `x` and the second parameter is the
    /// current residual `b - Ax`. This callable can be used to obtain
    /// information about each step of the iteration.
    pub fn set_callable(
        mut self,
        callable: impl FnMut(&Element<Space>, &Element<Space>) + 'a,
    ) -> Self {
        self.callable = Some(Box::new(callable));
        self
    }

    /// Enable debug printing
    ///
    /// Print debug information on `stdout`.
    pub fn print_debug(mut self) -> Self {
        self.print_debug = true;
        self
    }

    /// Run the CG iteration.
    ///
    /// Returns the final relative residual, or the absolute residual when the
    /// right-hand side is zero. An exact initial solution returns immediately.
    /// If CG encounters a zero or non-finite search-direction denominator, it
    /// terminates without applying an invalid update and returns the current
    /// residual.
    pub fn run(mut self) -> Scalar::Real {
        fn print_success<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!("CG converged in {it_count} iterations with relative residual {rel_res:+E}.");
        }

        fn print_fail<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!(
                "CG did not converge in {it_count} iterations. Final relative residual is {rel_res:+E}.",
            );
        }

        let mut res = self.rhs.clone();
        self.operator
            .apply(-<Scalar as One>::one(), self.x, One::one(), &mut res);

        let mut p = res.clone();

        // This syntax is only necessary because the type inference becomes confused for some reason.
        // If I write `let rhs_norm = self.rhs.norm()` the compiler thinks that `self.rhs` is a space and
        // not an element.
        let mut res_inner = res.inner(&res);
        let rhs_norm = self.rhs.norm();
        let mut residual = Sqrt::sqrt(res_inner.abs());

        // Handle an exact solution before normalizing. In particular, this
        // avoids `0 / 0` for a zero right-hand side and zero initial guess.
        if residual == Scalar::Real::zero() {
            if self.print_debug {
                print_success(0, residual);
            }
            return residual;
        }

        let use_absolute_residual = rhs_norm == Scalar::Real::zero();
        let mut reported_residual = if use_absolute_residual {
            residual
        } else {
            residual / rhs_norm
        };

        if reported_residual < self.tol {
            if self.print_debug {
                print_success(0, reported_residual);
            }
            return reported_residual;
        }

        for it_count in 0..self.max_iter {
            let a_p = self.operator.dot(&p);
            let p_conj_inner = a_p.inner(&p);
            let denominator_magnitude = p_conj_inner.abs();
            if denominator_magnitude == Scalar::Real::zero() || !denominator_magnitude.is_finite() {
                if self.print_debug {
                    print_fail(it_count, reported_residual);
                }
                return reported_residual;
            }

            let alpha = res_inner / p_conj_inner;
            *self.x += (&p).mul(alpha);
            res -= (&a_p).mul(alpha);

            if let Some(callable) = self.callable.as_mut() {
                callable(self.x, &res);
            }
            let res_inner_previous = res_inner;
            res_inner = res.inner(&res);
            residual = Sqrt::sqrt(res_inner.abs());
            reported_residual = if use_absolute_residual {
                residual
            } else {
                residual / rhs_norm
            };
            if reported_residual < self.tol {
                if self.print_debug {
                    print_success(it_count, reported_residual);
                }
                return reported_residual;
            }
            let beta = res_inner / res_inner_previous;
            p *= beta;
            p += &res;
        }

        if self.print_debug {
            print_fail(self.max_iter, reported_residual);
        }
        reported_residual
    }
}

#[cfg(test)]
mod test {
    use std::cell::Cell;

    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        Norm,
        abstract_operator::OperatorBase,
        dense::array::DynArray,
        operator::{
            abstract_operator::Operator, algorithms::conjugate_gradients::CgIteration,
            space::zero_element,
        },
    };

    #[test]
    fn zero_rhs_and_zero_guess_converges_without_iteration() {
        let mut mat = DynArray::<f64, 2>::from_shape([2, 2]);
        mat[[0, 0]] = 1.0;
        mat[[1, 1]] = 1.0;
        let op = Operator::from(&mat);
        let rhs = zero_element(op.range());
        let mut x = zero_element(op.domain());
        let callback_count = Cell::new(0);

        let residual = CgIteration::new(&op, &rhs, &mut x)
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

        let residual = CgIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-12)
            .set_callable(|x, _| {
                callback_count.set(callback_count.get() + 1);
                final_x_0.set(x.imp()[[0]]);
                final_x_1.set(x.imp()[[1]]);
            })
            .run();

        assert!(residual.is_finite());
        assert!(residual < 1E-12);
        assert_eq!(callback_count.get(), 1);
        assert_relative_eq!(final_x_0.get(), 0.0, epsilon = 1E-12);
        assert_relative_eq!(final_x_1.get(), 0.0, epsilon = 1E-12);
    }

    #[test]
    fn zero_denominator_does_not_update_solution() {
        let mat = DynArray::<f64, 2>::from_shape([2, 2]);
        let op = Operator::from(&mat);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut()[[0]] = 1.0;
        let mut x = zero_element(op.domain());
        x.imp_mut()[[0]] = 3.0;
        x.imp_mut()[[1]] = 4.0;
        let callback_count = Cell::new(0);

        let residual = CgIteration::new(&op, &rhs, &mut x)
            .set_callable(|_, _| callback_count.set(callback_count.get() + 1))
            .run();

        assert_eq!(residual, 1.0);
        assert_eq!(callback_count.get(), 0);
    }

    #[test]
    fn non_diagonal_spd_solution_has_small_true_residual() {
        let mut mat = DynArray::<f64, 2>::from_shape([2, 2]);
        mat[[0, 0]] = 4.0;
        mat[[0, 1]] = 1.0;
        mat[[1, 0]] = 1.0;
        mat[[1, 1]] = 3.0;
        let op = Operator::from(&mat);
        let mut rhs = zero_element(op.range());
        rhs.imp_mut()[[0]] = 1.0;
        rhs.imp_mut()[[1]] = 2.0;
        let mut x = zero_element(op.domain());
        let final_x_0 = Cell::new(f64::NAN);
        let final_x_1 = Cell::new(f64::NAN);

        let reported_residual = CgIteration::new(&op, &rhs, &mut x)
            .set_tol(1E-12)
            .set_callable(|x, _| {
                final_x_0.set(x.imp()[[0]]);
                final_x_1.set(x.imp()[[1]]);
            })
            .run();

        let x_0 = final_x_0.get();
        let x_1 = final_x_1.get();
        let residual_0 = 1.0 - (4.0 * x_0 + x_1);
        let residual_1 = 2.0 - (x_0 + 3.0 * x_1);
        let true_relative_residual =
            (residual_0 * residual_0 + residual_1 * residual_1).sqrt() / 5.0_f64.sqrt();
        assert!(reported_residual < 1E-12);
        assert!(true_relative_residual < 1E-12);
        assert_relative_eq!(x_0, 1.0 / 11.0, epsilon = 1E-12);
        assert_relative_eq!(x_1, 7.0 / 11.0, epsilon = 1E-12);
    }

    #[test]
    fn test_cg() {
        let dim = 10;
        let tol = 1E-5;

        let mut residuals = Vec::<f64>::new();

        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let mut mat = DynArray::<f64, 2>::from_shape([dim, dim]);

        for index in 0..dim {
            mat[[index, index]] = rng.random_range(1.0..=2.0);
        }

        let op = Operator::from(&mat);

        let mut rhs = zero_element(op.range());
        rhs.imp_mut().fill_from_equally_distributed(&mut rng);
        let mut x = zero_element(op.domain());

        let cg = (CgIteration::new(&op, &rhs, &mut x))
            .set_callable(|_, res| {
                let res_norm = res.norm();
                residuals.push(res_norm);
            })
            .set_tol(tol)
            .print_debug();

        let res = cg.run();
        assert!(res < tol);
    }
}
