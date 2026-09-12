//! Matrix Market reader and writer.
//!
//! [`read_array_mm`] and [`read_coordinate_mm`] support general real and
//! complex matrices. Array records are interpreted in the column-major order
//! mandated by the Matrix Market format. Malformed headers, dimensions,
//! records, counts, and coordinate indices are returned as [`RlstError`]s.

use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead};
use std::ops::AddAssign;
use std::path::Path;

use num::Num;

use crate::dense::array::DynArray;
use crate::sparse::csr_mat::CsrMatrix;
use crate::{Array, FromAij, MmIdentifier, RlstError, RlstResult, UnsafeRandom1DAccessByValue};

/// Definition of Matrix format types.
#[derive(PartialEq, Clone, Copy)]
pub enum MatrixFormat {
    /// Coordinate format.
    Coordinate,
    /// Dense array format.
    Array,
}

/// Definition of the data type.
#[derive(PartialEq, Clone, Copy)]
pub enum DataType {
    /// Real entries.
    Real,
    /// Integer entries (currently not supported).
    Integer,
    /// Complex entries.
    Complex,
}

/// Matrix Market symmetry type.
#[derive(PartialEq, Clone, Copy)]
pub enum SymmetryType {
    /// General matrices without symmetry.
    General,
    /// Symmetric matrices (currently not supported).
    Symmetric,
    /// Skew-symmetric matrices (currently not supported).
    Skew,
    /// Hermitian matrices (currently not supported).
    Hermitian,
}

/// Parsed Matrix Market header information used internally by the readers.
///
/// Its fields are intentionally private; callers select a typed reader rather than
/// constructing or inspecting this parser state.
pub struct MatrixMarketInfo {
    format: MatrixFormat,
    data_type: DataType,
    symmetry: SymmetryType,
}

/// Write stored matrix entries in general Matrix Market coordinate format.
///
/// Coordinates emitted by the matrix's AIJ iterator are converted from Rust's
/// zero-based indices to Matrix Market's one-based indices. Existing files are
/// truncated. Returns [`RlstError::IoError`] if the file cannot be created or written.
pub fn write_coordinate_mm<
    T: MmIdentifier + Display,
    Mat: crate::traits::iterators::AijIteratorByValue<Item = T> + crate::traits::Shape<2>,
>(
    mat: &Mat,
    fname: &str,
) -> RlstResult<()> {
    let mut output = File::create(fname)
        .map_err(|error| RlstError::IoError(format!("could not create {fname}: {error}")))?;
    writeln!(
        output,
        "%%MatrixMarket matrix coordinate {} general",
        T::MMTYPE
    )
    .and_then(|_| writeln!(output, "%"))
    .and_then(|_| {
        writeln!(
            output,
            "{} {} {}",
            mat.shape()[0],
            mat.shape()[1],
            mat.len()
        )
    })
    .map_err(|error| RlstError::IoError(format!("could not write {fname}: {error}")))?;
    for ([row, col], data) in mat.iter_aij_value() {
        writeln!(output, "{} {} {} ", 1 + row, 1 + col, data)
            .map_err(|error| RlstError::IoError(format!("could not write {fname}: {error}")))?;
    }
    Ok(())
}

/// Write a dense matrix in general Matrix Market array format.
///
/// Values are emitted in Matrix Market column-major record order. Existing files
/// are truncated. Returns [`RlstError::IoError`] if the file cannot be created or written.
pub fn write_array_mm<
    T: MmIdentifier + Display,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = T> + crate::traits::Shape<2>,
>(
    mat: &Array<ArrayImpl, 2>,
    fname: &str,
) -> RlstResult<()> {
    let mut output = File::create(fname)
        .map_err(|error| RlstError::IoError(format!("could not create {fname}: {error}")))?;
    writeln!(output, "%%MatrixMarket matrix array {} general", T::MMTYPE)
        .and_then(|_| writeln!(output, "%"))
        .and_then(|_| writeln!(output, "{} {}", mat.shape()[0], mat.shape()[1]))
        .map_err(|error| RlstError::IoError(format!("could not write {fname}: {error}")))?;
    for value in mat.iter_value() {
        writeln!(output, "{value}")
            .map_err(|error| RlstError::IoError(format!("could not write {fname}: {error}")))?;
    }
    Ok(())
}

/// Read a general dense Matrix Market array.
///
/// The header field must match `T` (`real` for `f32`/`f64`, `complex` for
/// `c32`/`c64`). Complex records contain separate real and imaginary fields.
/// Records are loaded in Matrix Market column-major order. The reader returns
/// [`RlstError::IoError`] for unsupported format, field, or symmetry declarations,
/// malformed dimensions or values, and missing or excess records.
pub fn read_array_mm<T: Copy + Default + MmIdentifier>(fname: &str) -> RlstResult<DynArray<T, 2>> {
    let mut reader = open_file(fname)
        .map_err(|error| RlstError::IoError(format!("could not open {fname}: {error}")))?;
    let info = parse_header(&mut reader)?;
    validate_info::<T>(&info, MatrixFormat::Array)?;
    let dimensions = next_record(&mut reader, "array dimensions")?;
    let fields: Vec<_> = dimensions.split_whitespace().collect();
    if fields.len() != 2 {
        return io_error(format!(
            "array dimension line must contain 2 fields: {dimensions}"
        ));
    }
    let nrows = parse_usize(fields[0], "row dimension")?;
    let ncols = parse_usize(fields[1], "column dimension")?;
    let nelems = nrows * ncols;
    let mut matrix = DynArray::<T, 2>::from_shape([nrows, ncols]);
    parse_array(
        &mut reader,
        matrix.data_mut().ok_or_else(|| {
            RlstError::IoError("dense Matrix Market destination has no mutable storage".to_string())
        })?,
        nelems,
    )?;
    Ok(matrix)
}

/// Read a general sparse Matrix Market coordinate matrix.
///
/// Record counts and one-based coordinate bounds are validated before the CSR
/// matrix is constructed. Complex records contain separate real and imaginary
/// fields. Duplicate coordinates are summed and explicit zeros removed by CSR
/// assembly. The reader returns [`RlstError::IoError`] for unsupported headers,
/// malformed dimensions or records, count mismatches, and coordinates outside
/// the declared shape.
pub fn read_coordinate_mm<T: Default + Copy + Num + AddAssign + MmIdentifier>(
    fname: &str,
) -> RlstResult<CsrMatrix<T>> {
    let mut reader = open_file(fname)
        .map_err(|error| RlstError::IoError(format!("could not open {fname}: {error}")))?;
    let info = parse_header(&mut reader)?;
    validate_info::<T>(&info, MatrixFormat::Coordinate)?;
    let dimensions = next_record(&mut reader, "coordinate dimensions")?;
    let fields: Vec<_> = dimensions.split_whitespace().collect();
    if fields.len() != 3 {
        return io_error(format!(
            "coordinate dimension line must contain 3 fields: {dimensions}"
        ));
    }
    let nrows = parse_usize(fields[0], "row dimension")?;
    let ncols = parse_usize(fields[1], "column dimension")?;
    let nelems = parse_usize(fields[2], "entry count")?;
    let (rows, cols, data) = parse_coordinate(&mut reader, nrows, ncols, nelems)?;
    Ok(CsrMatrix::from_aij([nrows, ncols], &rows, &cols, &data))
}

fn io_error<T>(message: String) -> RlstResult<T> {
    Err(RlstError::IoError(message))
}

fn parse_usize(field: &str, description: &str) -> RlstResult<usize> {
    field
        .parse()
        .map_err(|_| RlstError::IoError(format!("invalid {description} {field:?}")))
}

fn open_file<P: AsRef<Path>>(fname: P) -> io::Result<io::Lines<io::BufReader<File>>> {
    Ok(io::BufReader::new(File::open(fname)?).lines())
}

fn read_line(
    reader: &mut io::Lines<io::BufReader<File>>,
    context: &str,
) -> RlstResult<Option<String>> {
    reader
        .next()
        .transpose()
        .map_err(|error| RlstError::IoError(format!("failed while reading {context}: {error}")))
}

fn next_record(reader: &mut io::Lines<io::BufReader<File>>, context: &str) -> RlstResult<String> {
    while let Some(line) = read_line(reader, context)? {
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('%') {
            return Ok(trimmed.to_string());
        }
    }
    io_error(format!("missing {context}"))
}

fn parse_header(reader: &mut io::Lines<io::BufReader<File>>) -> RlstResult<MatrixMarketInfo> {
    let line = read_line(reader, "Matrix Market header")?
        .ok_or_else(|| RlstError::IoError("missing Matrix Market header".to_string()))?;
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() != 5
        || fields[0] != "%%MatrixMarket"
        || !fields[1].eq_ignore_ascii_case("matrix")
    {
        return io_error(format!("invalid Matrix Market header: {line}"));
    }
    let format = if fields[2].eq_ignore_ascii_case("coordinate") {
        MatrixFormat::Coordinate
    } else if fields[2].eq_ignore_ascii_case("array") {
        MatrixFormat::Array
    } else {
        return io_error(format!("unsupported Matrix Market format {:?}", fields[2]));
    };
    let data_type = if fields[3].eq_ignore_ascii_case("real") {
        DataType::Real
    } else if fields[3].eq_ignore_ascii_case("complex") {
        DataType::Complex
    } else if fields[3].eq_ignore_ascii_case("integer") {
        DataType::Integer
    } else {
        return io_error(format!("unsupported Matrix Market field {:?}", fields[3]));
    };
    let symmetry = if fields[4].eq_ignore_ascii_case("general") {
        SymmetryType::General
    } else if fields[4].eq_ignore_ascii_case("symmetric") {
        SymmetryType::Symmetric
    } else if fields[4].eq_ignore_ascii_case("skew-symmetric") {
        SymmetryType::Skew
    } else if fields[4].eq_ignore_ascii_case("hermitian") {
        SymmetryType::Hermitian
    } else {
        return io_error(format!(
            "unsupported Matrix Market symmetry {:?}",
            fields[4]
        ));
    };
    Ok(MatrixMarketInfo {
        format,
        data_type,
        symmetry,
    })
}

fn validate_info<T: MmIdentifier>(
    info: &MatrixMarketInfo,
    expected: MatrixFormat,
) -> RlstResult<()> {
    if info.format != expected {
        return io_error(
            "Matrix Market storage format does not match the requested reader".to_string(),
        );
    }
    if info.symmetry != SymmetryType::General {
        return io_error("only general Matrix Market matrices are supported".to_string());
    }
    let declared = match info.data_type {
        DataType::Real => "real",
        DataType::Complex => "complex",
        DataType::Integer => "integer",
    };
    if declared != T::MMTYPE {
        return io_error(format!(
            "Matrix Market field {declared:?} is incompatible with requested scalar field {:?}",
            T::MMTYPE
        ));
    }
    Ok(())
}

fn ensure_no_records(reader: &mut io::Lines<io::BufReader<File>>) -> RlstResult<()> {
    while let Some(line) = read_line(reader, "trailing Matrix Market data")? {
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('%') {
            return io_error(format!("unexpected excess data record: {trimmed}"));
        }
    }
    Ok(())
}

fn parse_array<T: MmIdentifier>(
    reader: &mut io::Lines<io::BufReader<File>>,
    buffer: &mut [T],
    nelems: usize,
) -> RlstResult<()> {
    for (index, destination) in buffer.iter_mut().take(nelems).enumerate() {
        let line = next_record(reader, &format!("array record {}", index + 1))?;
        let fields: Vec<_> = line.split_whitespace().collect();
        *destination = T::from_mm_fields(&fields).map_err(|error| {
            RlstError::IoError(format!(
                "invalid array record {} ({line}): {error}",
                index + 1
            ))
        })?;
    }
    ensure_no_records(reader)
}

fn parse_coordinate<T: MmIdentifier>(
    reader: &mut io::Lines<io::BufReader<File>>,
    nrows: usize,
    ncols: usize,
    nelems: usize,
) -> RlstResult<(Vec<usize>, Vec<usize>, Vec<T>)> {
    let mut rows = Vec::with_capacity(nelems);
    let mut cols = Vec::with_capacity(nelems);
    let mut data = Vec::with_capacity(nelems);
    for index in 0..nelems {
        let line = next_record(reader, &format!("coordinate record {}", index + 1))?;
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 2 + T::MM_NFIELDS {
            return io_error(format!(
                "coordinate record {} has {} fields, expected {}: {line}",
                index + 1,
                fields.len(),
                2 + T::MM_NFIELDS
            ));
        }
        let row = parse_usize(fields[0], "coordinate row")?;
        let col = parse_usize(fields[1], "coordinate column")?;
        if row == 0 || row > nrows || col == 0 || col > ncols {
            return io_error(format!(
                "coordinate ({row}, {col}) is outside one-based matrix bounds {nrows}x{ncols}"
            ));
        }
        let value = T::from_mm_fields(&fields[2..]).map_err(|error| {
            RlstError::IoError(format!(
                "invalid coordinate record {} ({line}): {error}",
                index + 1
            ))
        })?;
        rows.push(row - 1);
        cols.push(col - 1);
        data.push(value);
    }
    ensure_no_records(reader)?;
    Ok((rows, cols, data))
}

#[cfg(test)]
mod tests {
    use super::{read_array_mm, read_coordinate_mm};
    use crate::c64;
    use crate::traits::iterators::AijIteratorByValue;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

    fn fixture(contents: &str) -> PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("rlst-mm-{}-{id}.mtx", std::process::id()));
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn reads_rectangular_real_array_in_column_major_order() {
        let path = fixture("%%MatrixMarket matrix array real general\n2 3\n1\n2\n3\n4\n5\n6\n");
        let matrix = read_array_mm::<f64>(path.to_str().unwrap()).unwrap();
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 0]], 2.0);
        assert_eq!(matrix[[0, 1]], 3.0);
        assert_eq!(matrix[[1, 2]], 6.0);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_complex_array_and_coordinate_records() {
        let array_path =
            fixture("%%MatrixMarket matrix array complex general\n1 2\n1.5 -2\n3 4.5\n");
        let matrix = read_array_mm::<c64>(array_path.to_str().unwrap()).unwrap();
        assert_eq!(matrix[[0, 0]], c64::new(1.5, -2.0));
        assert_eq!(matrix[[0, 1]], c64::new(3.0, 4.5));

        let coordinate_path = fixture(
            "%%MatrixMarket matrix coordinate complex general\n2 3 2\n2 3 1 -1\n1 1 2.5 0.5\n",
        );
        let matrix = read_coordinate_mm::<c64>(coordinate_path.to_str().unwrap()).unwrap();
        let mut entries: Vec<_> = matrix.iter_aij_value().collect();
        entries.sort_by_key(|entry| entry.0);
        assert_eq!(
            entries,
            vec![([0, 0], c64::new(2.5, 0.5)), ([1, 2], c64::new(1.0, -1.0))]
        );
        fs::remove_file(array_path).unwrap();
        fs::remove_file(coordinate_path).unwrap();
    }

    #[test]
    fn rejects_missing_malformed_truncated_and_excess_data() {
        let missing =
            std::env::temp_dir().join(format!("rlst-mm-missing-{}.mtx", std::process::id()));
        assert!(read_array_mm::<f64>(missing.to_str().unwrap()).is_err());

        let cases = [
            "",
            "not a header\n",
            "%%MatrixMarket matrix array real general\n2 x\n",
            "%%MatrixMarket matrix array real general\n1 2\n1\n",
            "%%MatrixMarket matrix array real general\n1 1\n1\n2\n",
            "%%MatrixMarket matrix array complex general\n1 1\n1\n",
            "%%MatrixMarket matrix array real general\n1 1\nnot-a-number\n",
        ];
        for contents in cases {
            let path = fixture(contents);
            assert!(
                read_array_mm::<f64>(path.to_str().unwrap()).is_err(),
                "accepted {contents:?}"
            );
            fs::remove_file(path).unwrap();
        }

        let malformed_complex = fixture("%%MatrixMarket matrix array complex general\n1 1\n1\n");
        assert!(read_array_mm::<c64>(malformed_complex.to_str().unwrap()).is_err());
        fs::remove_file(malformed_complex).unwrap();

        let truncated_coordinate =
            fixture("%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 2\n");
        assert!(read_coordinate_mm::<f64>(truncated_coordinate.to_str().unwrap()).is_err());
        fs::remove_file(truncated_coordinate).unwrap();
    }

    #[test]
    fn reads_real_coordinate_records() {
        let path =
            fixture("%%MatrixMarket matrix coordinate real general\n2 3 2\n2 3 -1.5\n1 1 2.5\n");
        let matrix = read_coordinate_mm::<f64>(path.to_str().unwrap()).unwrap();
        assert_eq!(
            matrix.iter_aij_value().collect::<Vec<_>>(),
            vec![([0, 0], 2.5), ([1, 2], -1.5)]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_incompatible_fields_and_invalid_coordinates() {
        let array_path = fixture("%%MatrixMarket matrix array complex general\n1 1\n1 2\n");
        assert!(read_array_mm::<f64>(array_path.to_str().unwrap()).is_err());
        fs::remove_file(array_path).unwrap();

        for record in ["0 1 2", "1 0 2", "3 1 2", "1 4 2"] {
            let path = fixture(&format!(
                "%%MatrixMarket matrix coordinate real general\n2 3 1\n{record}\n"
            ));
            assert!(read_coordinate_mm::<f64>(path.to_str().unwrap()).is_err());
            fs::remove_file(path).unwrap();
        }

        let path = fixture("%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 2\n2 2 3\n");
        assert!(read_coordinate_mm::<f64>(path.to_str().unwrap()).is_err());
        fs::remove_file(path).unwrap();
    }
}
