use std::ops::AddAssign;

use rlst::sparse::csr_mat::CsrMatrix;
use rlst::{Shape, c64};

use super::{UmfpackError, UmfpackResult};

pub(super) fn complex_parts(values: &[c64]) -> (Vec<f64>, Vec<f64>) {
    values.iter().map(|value| (value.re, value.im)).unzip()
}

#[derive(Debug)]
pub(super) struct CscMatrix<Item> {
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) column_pointers: Vec<i64>,
    pub(super) row_indices: Vec<i64>,
    pub(super) values: Vec<Item>,
}

impl<Item> CscMatrix<Item>
where
    Item: AddAssign + Copy + Default,
{
    pub(super) fn from_csr(matrix: &CsrMatrix<Item>) -> UmfpackResult<Self> {
        let [rows, columns] = matrix.shape();
        if rows == 0 || columns == 0 {
            return Err(UmfpackError::EmptyMatrix);
        }
        if rows != columns {
            return Err(UmfpackError::NonSquareMatrix { rows, columns });
        }
        let indptr = matrix.indptr().data().ok_or(UmfpackError::InvalidCsr(
            "index pointers are not contiguous",
        ))?;
        let indices = matrix.indices().data().ok_or(UmfpackError::InvalidCsr(
            "column indices are not contiguous",
        ))?;
        let data = matrix
            .data()
            .data()
            .ok_or(UmfpackError::InvalidCsr("values are not contiguous"))?;

        validate_csr(rows, columns, indptr, indices, data)?;

        let mut counts = vec![0usize; columns];
        for &column in indices {
            counts[column] = counts[column]
                .checked_add(1)
                .ok_or(UmfpackError::IndexOutOfRange(column))?;
        }

        let mut pointers = vec![0usize; columns + 1];
        for column in 0..columns {
            pointers[column + 1] = pointers[column]
                .checked_add(counts[column])
                .ok_or(UmfpackError::IndexOutOfRange(counts[column]))?;
        }
        let mut offsets = pointers[..columns].to_vec();
        let mut raw_rows = vec![0usize; indices.len()];
        let mut raw_values = vec![Item::default(); data.len()];

        for row in 0..rows {
            for entry in indptr[row]..indptr[row + 1] {
                let column = indices[entry];
                let offset = offsets[column];
                raw_rows[offset] = row;
                raw_values[offset] = data[entry];
                offsets[column] += 1;
            }
        }

        let mut column_pointers = Vec::with_capacity(columns + 1);
        let mut row_indices = Vec::with_capacity(indices.len());
        let mut values = Vec::with_capacity(data.len());
        column_pointers.push(0);
        for column in 0..columns {
            let mut entry = pointers[column];
            while entry < pointers[column + 1] {
                let row = raw_rows[entry];
                let mut value = raw_values[entry];
                entry += 1;
                while entry < pointers[column + 1] && raw_rows[entry] == row {
                    value += raw_values[entry];
                    entry += 1;
                }
                row_indices.push(to_i64(row)?);
                values.push(value);
            }
            column_pointers.push(to_i64(values.len())?);
        }

        Ok(Self {
            rows,
            columns,
            column_pointers,
            row_indices,
            values,
        })
    }

    pub(super) fn rows_i64(&self) -> UmfpackResult<i64> {
        to_i64(self.rows)
    }

    pub(super) fn columns_i64(&self) -> UmfpackResult<i64> {
        to_i64(self.columns)
    }
}

#[derive(Debug)]
pub(super) struct ComplexCscMatrix {
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) column_pointers: Vec<i64>,
    pub(super) row_indices: Vec<i64>,
    pub(super) real_values: Vec<f64>,
    pub(super) imaginary_values: Vec<f64>,
}

impl ComplexCscMatrix {
    pub(super) fn from_csr(matrix: &CsrMatrix<c64>) -> UmfpackResult<Self> {
        let csc = CscMatrix::from_csr(matrix)?;
        let (real_values, imaginary_values) = complex_parts(&csc.values);

        Ok(Self {
            rows: csc.rows,
            columns: csc.columns,
            column_pointers: csc.column_pointers,
            row_indices: csc.row_indices,
            real_values,
            imaginary_values,
        })
    }

    pub(super) fn rows_i64(&self) -> UmfpackResult<i64> {
        to_i64(self.rows)
    }

    pub(super) fn columns_i64(&self) -> UmfpackResult<i64> {
        to_i64(self.columns)
    }
}

fn validate_csr<Item>(
    rows: usize,
    columns: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[Item],
) -> UmfpackResult<()> {
    if indptr.len() != rows + 1 {
        return Err(UmfpackError::InvalidCsr(
            "index pointer length does not match the row count",
        ));
    }
    if indptr.first() != Some(&0) || indptr.last() != Some(&indices.len()) {
        return Err(UmfpackError::InvalidCsr(
            "index pointers do not span the column-index array",
        ));
    }
    if indices.len() != data.len() {
        return Err(UmfpackError::InvalidCsr(
            "column-index and value lengths differ",
        ));
    }
    if indptr.windows(2).any(|window| window[0] > window[1]) {
        return Err(UmfpackError::InvalidCsr(
            "index pointers are not monotonically increasing",
        ));
    }
    if indices.iter().any(|&column| column >= columns) {
        return Err(UmfpackError::InvalidCsr(
            "a column index is outside the matrix shape",
        ));
    }
    Ok(())
}

fn to_i64(value: usize) -> UmfpackResult<i64> {
    i64::try_from(value).map_err(|_| UmfpackError::IndexOutOfRange(value))
}
