//! Traits for I/O Operations.

/// Associates a Rust scalar with its Matrix Market field and parser.
///
/// Real values occupy one data field, while complex values occupy two fields
/// containing the real and imaginary parts, respectively.
pub trait MmIdentifier: Sized {
    /// Matrix Market field identifier.
    const MMTYPE: &'static str;

    /// Number of fields used for one value in a Matrix Market data record.
    const MM_NFIELDS: usize;

    /// Parse one value from its Matrix Market fields.
    fn from_mm_fields(fields: &[&str]) -> Result<Self, String>;
}

macro_rules! impl_real_mm_identifier {
    ($type:ty) => {
        impl MmIdentifier for $type {
            const MMTYPE: &'static str = "real";
            const MM_NFIELDS: usize = 1;

            fn from_mm_fields(fields: &[&str]) -> Result<Self, String> {
                if fields.len() != Self::MM_NFIELDS {
                    return Err(format!("expected one real field, found {}", fields.len()));
                }
                fields[0]
                    .parse::<Self>()
                    .map_err(|_| format!("invalid real value {:?}", fields[0]))
            }
        }
    };
}

impl_real_mm_identifier!(f32);
impl_real_mm_identifier!(f64);

macro_rules! impl_complex_mm_identifier {
    ($type:ty, $real:ty) => {
        impl MmIdentifier for $type {
            const MMTYPE: &'static str = "complex";
            const MM_NFIELDS: usize = 2;

            fn from_mm_fields(fields: &[&str]) -> Result<Self, String> {
                if fields.len() != Self::MM_NFIELDS {
                    return Err(format!(
                        "expected real and imaginary fields, found {}",
                        fields.len()
                    ));
                }
                let real = fields[0]
                    .parse::<$real>()
                    .map_err(|_| format!("invalid real part {:?}", fields[0]))?;
                let imaginary = fields[1]
                    .parse::<$real>()
                    .map_err(|_| format!("invalid imaginary part {:?}", fields[1]))?;
                Ok(<$type>::new(real, imaginary))
            }
        }
    };
}

impl_complex_mm_identifier!(crate::base_types::c32, f32);
impl_complex_mm_identifier!(crate::base_types::c64, f64);
