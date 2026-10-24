//! Errors of the FFTW interface.

use std::fmt;
use std::ops::Range;

use super::align::AlignmentClass;
use super::layout::Placement;
use super::options::Rigor;

/// The side of a transform that an error refers to.
///
/// ```
/// use rlst::fftw::Side;
///
/// assert_eq!(Side::Input.to_string(), "input");
/// assert_eq!(Side::Output.to_string(), "output");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The input buffer of a transform.
    Input,
    /// The output buffer of a transform.
    Output,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Side::Input => "input",
            Side::Output => "output",
        })
    }
}

/// An error of the FFTW interface.
///
/// RLST validates layouts, buffers and options before it calls FFTW, and returns one of these
/// errors instead of calling FFTW with arguments it would not accept. Each variant carries the
/// values involved, and its message names what was expected, what was found and, where there
/// is one, the remedy.
///
/// ```
/// use rlst::fftw::{Dim, FftwError, Layout, Placement};
///
/// let error = Layout::new(&[Dim { n: 0, is: 1, os: 1 }], &[], Placement::OutOfPlace)
///     .unwrap_err();
/// assert!(matches!(error, FftwError::InvalidLayout { .. }));
/// println!("{error}");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FftwError {
    /// The layout cannot describe a transform.
    #[error("invalid layout, {dimension}: {reason}")]
    InvalidLayout {
        /// The dimension concerned, for example `transform dimension 0`, or `layout` when the
        /// error concerns the layout as a whole.
        dimension: String,
        /// Why the layout is rejected.
        reason: String,
    },

    /// A buffer does not fit the plan's layout.
    #[error("the {side} buffer does not fit the plan's layout: expected {expected}, found {found}")]
    LayoutMismatch {
        /// The side of the buffer.
        side: Side,
        /// What the layout needs.
        expected: String,
        /// What the buffer provides.
        found: String,
    },

    /// A buffer's alignment class differs from the class the plan was made for.
    #[error(
        "the {side} buffer has alignment class {found}, but the plan was made for class \
         {expected}; allocate buffers with `rlst_dynamic_array!(…, shape | rlst::FFTW_ALIGNED)` \
         or `AlignedDynArray`, plan for this class with `Planner::alignment`, or plan with \
         `Planner::unaligned`"
    )]
    AlignmentMismatch {
        /// The side of the buffer.
        side: Side,
        /// The class of the plan.
        expected: AlignmentClass,
        /// The class of the buffer.
        found: AlignmentClass,
    },

    /// The input and output of an out-of-place transform overlap in memory.
    #[error(
        "the input bytes {:#x}..{:#x} and the output bytes {:#x}..{:#x} of an out-of-place \
         transform overlap; use separate buffers, or an in-place plan",
        .input.start, .input.end, .output.start, .output.end
    )]
    Overlap {
        /// The address range of the input.
        input: Range<usize>,
        /// The address range of the output.
        output: Range<usize>,
    },

    /// The layout's placement does not match the planner method.
    #[error(
        "the layout is {found}, but this planner method needs an {expected} layout; use the \
         `*_inplace` planner methods for in-place layouts and the others for out-of-place ones"
    )]
    PlacementMismatch {
        /// The placement the planner method needs.
        expected: Placement,
        /// The placement of the layout.
        found: Placement,
    },

    /// A real-to-real kind is not defined for the layout, for example REDFT00 with n = 1.
    #[error("invalid real-to-real kind: {reason}")]
    InvalidKind {
        /// Why the kind is rejected.
        reason: String,
    },

    /// FFTW has no input-preserving algorithm for this transform.
    #[error(
        "FFTW cannot preserve the input of a complex-to-real transform over {rank} dimensions; \
         plan without `preserve_input()`, which lets the plan overwrite its input"
    )]
    PreserveInputUnsupported {
        /// The number of transform dimensions.
        rank: usize,
    },

    /// Planning on scratch would need more memory than the planner's bound.
    #[error(
        "planning needs {needed} bytes of scratch, above the bound of {bound} bytes; raise the \
         bound with `Planner::max_scratch`, or plan on your own buffers with the `*_on` \
         planner methods"
    )]
    ScratchTooLarge {
        /// The bytes of scratch the layout needs.
        needed: usize,
        /// The planner's bound, from `Planner::max_scratch`.
        bound: usize,
    },

    /// The FFTW library linked at run time is older than RLST supports.
    #[error(
        "the linked FFTW is {found}, older than the minimum {minimum}; install a newer FFTW and \
         set PKG_CONFIG_PATH to its lib/pkgconfig directory"
    )]
    VersionTooOld {
        /// FFTW's version string.
        found: String,
        /// The minimum version.
        minimum: &'static str,
    },

    /// FFTW returned no plan.
    ///
    /// FFTW gives no reason. When RLST knows a remedy, `hint` names it, and the message
    /// shows it: for a preserving real-to-real plan with an HC2R kind,
    /// [`may_destroy_input`](super::Planner::may_destroy_input) (DC34).
    #[error(
        "FFTW could not create a plan with rigor {rigor} and flags {flags:#x} for {layout}; \
         FFTW gives no reason{}",
        .hint.map(|hint| format!("; {hint}")).unwrap_or_default()
    )]
    PlanCreationFailed {
        /// The planning rigor.
        rigor: Rigor,
        /// The flags passed to FFTW.
        flags: u32,
        /// A description of the layout.
        layout: String,
        /// A remedy, when RLST knows one.
        hint: Option<&'static str>,
    },

    /// FFTW did not accept the wisdom given to it.
    #[error("FFTW did not accept the {precision} wisdom; FFTW gives no reason")]
    WisdomImportFailed {
        /// The precision of the wisdom, `f32` or `f64`.
        precision: &'static str,
    },
}
