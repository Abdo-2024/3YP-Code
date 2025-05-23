// Copyright 2010-2025 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ortools/lp_data/sparse.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "ortools/lp_data/lp_types.h"
#include "ortools/lp_data/permutation.h"
#include "ortools/lp_data/sparse_column.h"
#include "ortools/util/return_macros.h"

namespace operations_research {
namespace glop {

namespace {

using ::util::Reverse;

template <typename Matrix>
EntryIndex ComputeNumEntries(const Matrix& matrix) {
  EntryIndex num_entries(0);
  const ColIndex num_cols(matrix.num_cols());
  for (ColIndex col(0); col < num_cols; ++col) {
    num_entries += matrix.column(col).num_entries();
  }
  return num_entries;
}

// Computes the 1-norm of the matrix.
// The 1-norm |A| is defined as max_j sum_i |a_ij| or
// max_col sum_row |a(row,col)|.
template <typename Matrix>
Fractional ComputeOneNormTemplate(const Matrix& matrix) {
  Fractional norm(0.0);
  const ColIndex num_cols(matrix.num_cols());
  for (ColIndex col(0); col < num_cols; ++col) {
    Fractional column_norm(0);
    for (const SparseColumn::Entry e : matrix.column(col)) {
      // Compute sum_i |a_ij|.
      column_norm += fabs(e.coefficient());
    }
    // Compute max_j sum_i |a_ij|
    norm = std::max(norm, column_norm);
  }
  return norm;
}

// Computes the oo-norm (infinity-norm) of the matrix.
// The oo-norm |A| is defined as max_i sum_j |a_ij| or
// max_row sum_col |a(row,col)|.
template <typename Matrix>
Fractional ComputeInfinityNormTemplate(const Matrix& matrix) {
  DenseColumn row_sum(matrix.num_rows(), 0.0);
  const ColIndex num_cols(matrix.num_cols());
  for (ColIndex col(0); col < num_cols; ++col) {
    for (const SparseColumn::Entry e : matrix.column(col)) {
      // Compute sum_j |a_ij|.
      row_sum[e.row()] += fabs(e.coefficient());
    }
  }

  // Compute max_i sum_j |a_ij|
  Fractional norm = 0.0;
  const RowIndex num_rows(matrix.num_rows());
  for (RowIndex row(0); row < num_rows; ++row) {
    norm = std::max(norm, row_sum[row]);
  }
  return norm;
}

}  // namespace

// --------------------------------------------------------
// SparseMatrix
// --------------------------------------------------------
SparseMatrix::SparseMatrix() : columns_(), num_rows_(0) {}

#if (!defined(_MSC_VER) || (_MSC_VER >= 1800))
SparseMatrix::SparseMatrix(
    std::initializer_list<std::initializer_list<Fractional>> init_list) {
  ColIndex num_cols(0);
  num_rows_ = RowIndex(init_list.size());
  RowIndex row(0);
  for (std::initializer_list<Fractional> init_row : init_list) {
    num_cols = std::max(num_cols, ColIndex(init_row.size()));
    columns_.resize(num_cols, SparseColumn());
    ColIndex col(0);
    for (Fractional value : init_row) {
      if (value != 0.0) {
        columns_[col].SetCoefficient(row, value);
      }
      ++col;
    }
    ++row;
  }
}
#endif

void SparseMatrix::Clear() {
  columns_.clear();
  num_rows_ = RowIndex(0);
}

bool SparseMatrix::IsEmpty() const {
  return columns_.empty() || num_rows_ == 0;
}

void SparseMatrix::CleanUp() {
  const ColIndex num_cols(columns_.size());
  for (ColIndex col(0); col < num_cols; ++col) {
    columns_[col].CleanUp();
  }
}

bool SparseMatrix::CheckNoDuplicates() const {
  DenseBooleanColumn boolean_column;
  const ColIndex num_cols(columns_.size());
  for (ColIndex col(0); col < num_cols; ++col) {
    if (!columns_[col].CheckNoDuplicates(&boolean_column)) return false;
  }
  return true;
}

bool SparseMatrix::IsCleanedUp() const {
  const ColIndex num_cols(columns_.size());
  for (ColIndex col(0); col < num_cols; ++col) {
    if (!columns_[col].IsCleanedUp()) return false;
  }
  return true;
}

void SparseMatrix::SetNumRows(RowIndex num_rows) { num_rows_ = num_rows; }

ColIndex SparseMatrix::AppendEmptyColumn() {
  const ColIndex result = columns_.size();
  columns_.push_back(SparseColumn());
  return result;
}

void SparseMatrix::AppendUnitVector(RowIndex row, Fractional value) {
  DCHECK_LT(row, num_rows_);
  SparseColumn new_col;
  new_col.SetCoefficient(row, value);
  columns_.push_back(std::move(new_col));
}

void SparseMatrix::Swap(SparseMatrix* matrix) {
  // We do not need to swap the different mutable scratchpads we use.
  columns_.swap(matrix->columns_);
  std::swap(num_rows_, matrix->num_rows_);
}

void SparseMatrix::PopulateFromZero(RowIndex num_rows, ColIndex num_cols) {
  columns_.resize(num_cols, SparseColumn());
  for (ColIndex col(0); col < num_cols; ++col) {
    columns_[col].Clear();
  }
  num_rows_ = num_rows;
}

void SparseMatrix::PopulateFromIdentity(ColIndex num_cols) {
  PopulateFromZero(ColToRowIndex(num_cols), num_cols);
  for (ColIndex col(0); col < num_cols; ++col) {
    const RowIndex row = ColToRowIndex(col);
    columns_[col].SetCoefficient(row, Fractional(1.0));
  }
}

template <typename Matrix>
void SparseMatrix::PopulateFromTranspose(const Matrix& input) {
  Reset(RowToColIndex(input.num_rows()), ColToRowIndex(input.num_cols()));

  // We do a first pass on the input matrix to resize the new columns properly.
  StrictITIVector<RowIndex, EntryIndex> row_degree(input.num_rows(),
                                                   EntryIndex(0));
  for (ColIndex col(0); col < input.num_cols(); ++col) {
    for (const SparseColumn::Entry e : input.column(col)) {
      ++row_degree[e.row()];
    }
  }
  for (RowIndex row(0); row < input.num_rows(); ++row) {
    columns_[RowToColIndex(row)].Reserve(row_degree[row]);
  }

  for (ColIndex col(0); col < input.num_cols(); ++col) {
    const RowIndex transposed_row = ColToRowIndex(col);
    for (const SparseColumn::Entry e : input.column(col)) {
      const ColIndex transposed_col = RowToColIndex(e.row());
      columns_[transposed_col].SetCoefficient(transposed_row, e.coefficient());
    }
  }
  DCHECK(IsCleanedUp());
}

void SparseMatrix::PopulateFromSparseMatrix(const SparseMatrix& matrix) {
  Reset(ColIndex(0), matrix.num_rows_);
  columns_ = matrix.columns_;
}

template <typename Matrix>
void SparseMatrix::PopulateFromPermutedMatrix(
    const Matrix& a, const RowPermutation& row_perm,
    const ColumnPermutation& inverse_col_perm) {
  const ColIndex num_cols = a.num_cols();
  Reset(num_cols, a.num_rows());
  for (ColIndex col(0); col < num_cols; ++col) {
    for (const auto e : a.column(inverse_col_perm[col])) {
      columns_[col].SetCoefficient(row_perm[e.row()], e.coefficient());
    }
  }
  DCHECK(CheckNoDuplicates());
}

void SparseMatrix::PopulateFromLinearCombination(Fractional alpha,
                                                 const SparseMatrix& a,
                                                 Fractional beta,
                                                 const SparseMatrix& b) {
  DCHECK_EQ(a.num_cols(), b.num_cols());
  DCHECK_EQ(a.num_rows(), b.num_rows());

  const ColIndex num_cols = a.num_cols();
  Reset(num_cols, a.num_rows());

  const RowIndex num_rows = a.num_rows();
  RandomAccessSparseColumn dense_column(num_rows);
  for (ColIndex col(0); col < num_cols; ++col) {
    for (const SparseColumn::Entry e : a.columns_[col]) {
      dense_column.AddToCoefficient(e.row(), alpha * e.coefficient());
    }
    for (const SparseColumn::Entry e : b.columns_[col]) {
      dense_column.AddToCoefficient(e.row(), beta * e.coefficient());
    }
    dense_column.PopulateSparseColumn(&columns_[col]);
    columns_[col].CleanUp();
    dense_column.Clear();
  }
}

void SparseMatrix::PopulateFromProduct(const SparseMatrix& a,
                                       const SparseMatrix& b) {
  const ColIndex num_cols = b.num_cols();
  const RowIndex num_rows = a.num_rows();
  Reset(num_cols, num_rows);

  RandomAccessSparseColumn tmp_column(num_rows);
  for (ColIndex col_b(0); col_b < num_cols; ++col_b) {
    for (const SparseColumn::Entry eb : b.columns_[col_b]) {
      if (eb.coefficient() == 0.0) {
        continue;
      }
      const ColIndex col_a = RowToColIndex(eb.row());
      for (const SparseColumn::Entry ea : a.columns_[col_a]) {
        const Fractional value = ea.coefficient() * eb.coefficient();
        tmp_column.AddToCoefficient(ea.row(), value);
      }
    }

    // Populate column col_b.
    tmp_column.PopulateSparseColumn(&columns_[col_b]);
    columns_[col_b].CleanUp();
    tmp_column.Clear();
  }
}

void SparseMatrix::DeleteColumns(const DenseBooleanRow& columns_to_delete) {
  if (columns_to_delete.empty()) return;
  ColIndex new_index(0);
  const ColIndex num_cols = columns_.size();
  for (ColIndex col(0); col < num_cols; ++col) {
    if (col >= columns_to_delete.size() || !columns_to_delete[col]) {
      columns_[col].Swap(&(columns_[new_index]));
      ++new_index;
    }
  }
  columns_.resize(new_index);
}

void SparseMatrix::DeleteRows(RowIndex new_num_rows,
                              const RowPermutation& permutation) {
  DCHECK_EQ(num_rows_, permutation.size());
  for (RowIndex row(0); row < num_rows_; ++row) {
    DCHECK_LT(permutation[row], new_num_rows);
  }
  const ColIndex end = num_cols();
  for (ColIndex col(0); col < end; ++col) {
    columns_[col].ApplyPartialRowPermutation(permutation);
  }
  SetNumRows(new_num_rows);
}

bool SparseMatrix::AppendRowsFromSparseMatrix(const SparseMatrix& matrix) {
  const ColIndex end = num_cols();
  if (end != matrix.num_cols()) {
    return false;
  }
  const RowIndex offset = num_rows();
  for (ColIndex col(0); col < end; ++col) {
    const SparseColumn& source_column = matrix.columns_[col];
    columns_[col].AppendEntriesWithOffset(source_column, offset);
  }
  SetNumRows(offset + matrix.num_rows());
  return true;
}

void SparseMatrix::ApplyRowPermutation(const RowPermutation& row_perm) {
  const ColIndex num_cols(columns_.size());
  for (ColIndex col(0); col < num_cols; ++col) {
    columns_[col].ApplyRowPermutation(row_perm);
  }
}

Fractional SparseMatrix::LookUpValue(RowIndex row, ColIndex col) const {
  return columns_[col].LookUpCoefficient(row);
}

bool SparseMatrix::Equals(const SparseMatrix& a, Fractional tolerance) const {
  if (num_cols() != a.num_cols() || num_rows() != a.num_rows()) {
    return false;
  }

  RandomAccessSparseColumn dense_column(num_rows());
  RandomAccessSparseColumn dense_column_a(num_rows());
  const ColIndex num_cols = a.num_cols();
  for (ColIndex col(0); col < num_cols; ++col) {
    // Store all entries of current matrix in a dense column.
    for (const SparseColumn::Entry e : columns_[col]) {
      dense_column.AddToCoefficient(e.row(), e.coefficient());
    }

    // Check all entries of a are those stored in the dense column.
    for (const SparseColumn::Entry e : a.columns_[col]) {
      if (fabs(e.coefficient() - dense_column.GetCoefficient(e.row())) >
          tolerance) {
        return false;
      }
    }

    // Store all entries of matrix a in a dense column.
    for (const SparseColumn::Entry e : a.columns_[col]) {
      dense_column_a.AddToCoefficient(e.row(), e.coefficient());
    }

    // Check all entries are those stored in the dense column a.
    for (const SparseColumn::Entry e : columns_[col]) {
      if (fabs(e.coefficient() - dense_column_a.GetCoefficient(e.row())) >
          tolerance) {
        return false;
      }
    }

    dense_column.Clear();
    dense_column_a.Clear();
  }

  return true;
}

void SparseMatrix::ComputeMinAndMaxMagnitudes(Fractional* min_magnitude,
                                              Fractional* max_magnitude) const {
  RETURN_IF_NULL(min_magnitude);
  RETURN_IF_NULL(max_magnitude);
  *min_magnitude = kInfinity;
  *max_magnitude = 0.0;
  for (ColIndex col(0); col < num_cols(); ++col) {
    for (const SparseColumn::Entry e : columns_[col]) {
      const Fractional magnitude = fabs(e.coefficient());
      if (magnitude != 0.0) {
        *min_magnitude = std::min(*min_magnitude, magnitude);
        *max_magnitude = std::max(*max_magnitude, magnitude);
      }
    }
  }
  if (*max_magnitude == 0.0) {
    *min_magnitude = 0.0;
  }
}

EntryIndex SparseMatrix::num_entries() const {
  return ComputeNumEntries(*this);
}
Fractional SparseMatrix::ComputeOneNorm() const {
  return ComputeOneNormTemplate(*this);
}
Fractional SparseMatrix::ComputeInfinityNorm() const {
  return ComputeInfinityNormTemplate(*this);
}

std::string SparseMatrix::Dump() const {
  std::string result;
  const ColIndex num_cols(columns_.size());

  for (RowIndex row(0); row < num_rows_; ++row) {
    result.append("{ ");
    for (ColIndex col(0); col < num_cols; ++col) {
      absl::StrAppendFormat(&result, "%g ", ToDouble(LookUpValue(row, col)));
    }
    result.append("}\n");
  }
  return result;
}

void SparseMatrix::Reset(ColIndex num_cols, RowIndex num_rows) {
  Clear();
  columns_.resize(num_cols, SparseColumn());
  num_rows_ = num_rows;
}

EntryIndex MatrixView::num_entries() const { return ComputeNumEntries(*this); }
Fractional MatrixView::ComputeOneNorm() const {
  return ComputeOneNormTemplate(*this);
}
Fractional MatrixView::ComputeInfinityNorm() const {
  return ComputeInfinityNormTemplate(*this);
}

// Instantiate needed templates.
template void SparseMatrix::PopulateFromTranspose<SparseMatrix>(
    const SparseMatrix& input);
template void SparseMatrix::PopulateFromPermutedMatrix<SparseMatrix>(
    const SparseMatrix& a, const RowPermutation& row_perm,
    const ColumnPermutation& inverse_col_perm);
template void SparseMatrix::PopulateFromPermutedMatrix<CompactSparseMatrixView>(
    const CompactSparseMatrixView& a, const RowPermutation& row_perm,
    const ColumnPermutation& inverse_col_perm);

void CompactSparseMatrix::PopulateFromMatrixView(const MatrixView& input) {
  num_cols_ = input.num_cols();
  num_rows_ = input.num_rows();
  const EntryIndex num_entries = input.num_entries();
  starts_.assign(num_cols_ + 1, EntryIndex(0));
  coefficients_.assign(num_entries, 0.0);
  rows_.assign(num_entries, RowIndex(0));
  EntryIndex index(0);
  for (ColIndex col(0); col < input.num_cols(); ++col) {
    starts_[col] = index;
    for (const SparseColumn::Entry e : input.column(col)) {
      coefficients_[index] = e.coefficient();
      rows_[index] = e.row();
      ++index;
    }
  }
  starts_[input.num_cols()] = index;
}

void CompactSparseMatrix::PopulateFromSparseMatrixAndAddSlacks(
    const SparseMatrix& input) {
  const int input_num_cols = input.num_cols().value();
  num_cols_ = input_num_cols + RowToColIndex(input.num_rows());
  num_rows_ = input.num_rows();
  const EntryIndex num_entries =
      input.num_entries() + EntryIndex(num_rows_.value());
  starts_.assign(num_cols_ + 1, EntryIndex(0));
  coefficients_.resize(num_entries, 0.0);
  rows_.resize(num_entries, RowIndex(0));
  EntryIndex index(0);
  for (ColIndex col(0); col < input_num_cols; ++col) {
    starts_[col] = index;
    for (const SparseColumn::Entry e : input.column(col)) {
      coefficients_[index] = e.coefficient();
      rows_[index] = e.row();
      ++index;
    }
  }
  for (RowIndex row(0); row < num_rows_; ++row) {
    starts_[input_num_cols + RowToColIndex(row)] = index;
    coefficients_[index] = 1.0;
    rows_[index] = row;
    ++index;
  }
  DCHECK_EQ(index, num_entries);
  starts_[num_cols_] = index;
}

void CompactSparseMatrix::PopulateFromTranspose(
    const CompactSparseMatrix& input) {
  num_cols_ = RowToColIndex(input.num_rows());
  num_rows_ = ColToRowIndex(input.num_cols());

  // Fill the starts_ vector by computing the number of entries of each rows and
  // then doing a cumulative sum. After this step starts_[col + 1] will be the
  // actual start of the column col when we are done.
  const ColIndex start_size = num_cols_ + 2;
  starts_.assign(start_size, EntryIndex(0));
  for (const RowIndex row : input.rows_) {
    ++starts_[RowToColIndex(row) + 2];
  }
  for (ColIndex col(2); col < start_size; ++col) {
    starts_[col] += starts_[col - 1];
  }
  coefficients_.resize(starts_.back(), 0.0);
  rows_.resize(starts_.back(), kInvalidRow);
  starts_.pop_back();

  // Use starts_ to fill the matrix. Note that starts_ is modified so that at
  // the end it has its final values.
  const auto entry_rows = rows_.view();
  const auto input_entry_rows = input.rows_.view();
  const auto entry_coefficients = coefficients_.view();
  const auto input_entry_coefficients = input.coefficients_.view();
  const auto num_cols = input.num_cols();
  const auto starts = starts_.view();
  for (ColIndex col(0); col < num_cols; ++col) {
    const RowIndex transposed_row = ColToRowIndex(col);
    for (const EntryIndex i : input.Column(col)) {
      const ColIndex transposed_col = RowToColIndex(input_entry_rows[i]);
      const EntryIndex index = starts[transposed_col + 1]++;
      entry_coefficients[index] = input_entry_coefficients[i];
      entry_rows[index] = transposed_row;
    }
  }

  DCHECK_EQ(starts_.front(), 0);
  DCHECK_EQ(starts_.back(), rows_.size());
}

void TriangularMatrix::PopulateFromTranspose(const TriangularMatrix& input) {
  CompactSparseMatrix::PopulateFromTranspose(input);

  // This takes care of the triangular special case.
  diagonal_coefficients_ = input.diagonal_coefficients_;
  all_diagonal_coefficients_are_one_ = input.all_diagonal_coefficients_are_one_;

  // The elimination structure of the transpose is not the same.
  pruned_ends_.resize(num_cols_, EntryIndex(0));
  for (ColIndex col(0); col < num_cols_; ++col) {
    pruned_ends_[col] = starts_[col + 1];
  }

  // Compute first_non_identity_column_. Note that this is not necessarily the
  // same as input.first_non_identity_column_ for an upper triangular matrix.
  first_non_identity_column_ = 0;
  const ColIndex end = diagonal_coefficients_.size();
  while (first_non_identity_column_ < end &&
         ColumnNumEntries(first_non_identity_column_) == 0 &&
         diagonal_coefficients_[first_non_identity_column_] == 1.0) {
    ++first_non_identity_column_;
  }
}

void CompactSparseMatrix::Reset(RowIndex num_rows) {
  num_rows_ = num_rows;
  num_cols_ = 0;
  rows_.clear();
  coefficients_.clear();
  starts_.clear();
  starts_.push_back(EntryIndex(0));
}

void TriangularMatrix::Reset(RowIndex num_rows, ColIndex col_capacity) {
  CompactSparseMatrix::Reset(num_rows);
  first_non_identity_column_ = 0;
  all_diagonal_coefficients_are_one_ = true;

  pruned_ends_.resize(col_capacity);
  diagonal_coefficients_.resize(col_capacity);
  starts_.resize(col_capacity + 1);
  // Non-zero entries in the first column always have an offset of 0.
  starts_[ColIndex(0)] = 0;
}

void CompactSparseMatrix::AddEntryToCurrentColumn(RowIndex row,
                                                  Fractional coeff) {
  rows_.push_back(row);
  coefficients_.push_back(coeff);
}

void CompactSparseMatrix::CloseCurrentColumn() {
  starts_.push_back(rows_.size());
  ++num_cols_;
}

ColIndex CompactSparseMatrix::AddDenseColumn(const DenseColumn& dense_column) {
  return AddDenseColumnPrefix(dense_column.const_view(), RowIndex(0));
}

ColIndex CompactSparseMatrix::AddDenseColumnPrefix(
    DenseColumn::ConstView dense_column, RowIndex start) {
  const RowIndex num_rows(dense_column.size());
  for (RowIndex row(start); row < num_rows; ++row) {
    if (dense_column[row] != 0.0) {
      rows_.push_back(row);
      coefficients_.push_back(dense_column[row]);
    }
  }
  starts_.push_back(rows_.size());
  ++num_cols_;
  return num_cols_ - 1;
}

ColIndex CompactSparseMatrix::AddDenseColumnWithNonZeros(
    const DenseColumn& dense_column, absl::Span<const RowIndex> non_zeros) {
  if (non_zeros.empty()) return AddDenseColumn(dense_column);
  for (const RowIndex row : non_zeros) {
    const Fractional value = dense_column[row];
    if (value != 0.0) {
      rows_.push_back(row);
      coefficients_.push_back(value);
    }
  }
  starts_.push_back(rows_.size());
  ++num_cols_;
  return num_cols_ - 1;
}

ColIndex CompactSparseMatrix::AddAndClearColumnWithNonZeros(
    DenseColumn* column, std::vector<RowIndex>* non_zeros) {
  for (const RowIndex row : *non_zeros) {
    const Fractional value = (*column)[row];
    if (value != 0.0) {
      rows_.push_back(row);
      coefficients_.push_back(value);
      (*column)[row] = 0.0;
    }
  }
  non_zeros->clear();
  starts_.push_back(rows_.size());
  ++num_cols_;
  return num_cols_ - 1;
}

void CompactSparseMatrix::Swap(CompactSparseMatrix* other) {
  std::swap(num_rows_, other->num_rows_);
  std::swap(num_cols_, other->num_cols_);
  coefficients_.swap(other->coefficients_);
  rows_.swap(other->rows_);
  starts_.swap(other->starts_);
}

void TriangularMatrix::Swap(TriangularMatrix* other) {
  CompactSparseMatrix::Swap(other);
  diagonal_coefficients_.swap(other->diagonal_coefficients_);
  std::swap(first_non_identity_column_, other->first_non_identity_column_);
  std::swap(all_diagonal_coefficients_are_one_,
            other->all_diagonal_coefficients_are_one_);
}

EntryIndex CompactSparseMatrixView::num_entries() const {
  return ComputeNumEntries(*this);
}
Fractional CompactSparseMatrixView::ComputeOneNorm() const {
  return ComputeOneNormTemplate(*this);
}
Fractional CompactSparseMatrixView::ComputeInfinityNorm() const {
  return ComputeInfinityNormTemplate(*this);
}

// Internal function used to finish adding one column to a triangular matrix.
// This sets the diagonal coefficient to the given value, and prepares the
// matrix for the next column addition.
void TriangularMatrix::CloseCurrentColumn(Fractional diagonal_value) {
  DCHECK_NE(diagonal_value, 0.0);
  // The vectors diagonal_coefficients, pruned_ends, and starts_ should have all
  // been preallocated by a call to SetTotalNumberOfColumns().
  DCHECK_LT(num_cols_, diagonal_coefficients_.size());
  diagonal_coefficients_[num_cols_] = diagonal_value;

  // TODO(user): This is currently not used by all matrices. It will be good
  // to fill it only when needed.
  DCHECK_LT(num_cols_, pruned_ends_.size());
  const EntryIndex num_entries = coefficients_.size();
  pruned_ends_[num_cols_] = num_entries;
  ++num_cols_;
  DCHECK_LT(num_cols_, starts_.size());
  starts_[num_cols_] = num_entries;
  if (first_non_identity_column_ == num_cols_ - 1 && diagonal_value == 1.0 &&
      num_entries == 0) {
    first_non_identity_column_ = num_cols_;
  }
  all_diagonal_coefficients_are_one_ =
      all_diagonal_coefficients_are_one_ && (diagonal_value == 1.0);
}

void TriangularMatrix::AddDiagonalOnlyColumn(Fractional diagonal_value) {
  CloseCurrentColumn(diagonal_value);
}

void TriangularMatrix::AddTriangularColumn(const ColumnView& column,
                                           RowIndex diagonal_row) {
  Fractional diagonal_value = 0.0;
  for (const SparseColumn::Entry e : column) {
    if (e.row() == diagonal_row) {
      diagonal_value = e.coefficient();
    } else {
      DCHECK_NE(0.0, e.coefficient());
      rows_.push_back(e.row());
      coefficients_.push_back(e.coefficient());
    }
  }
  CloseCurrentColumn(diagonal_value);
}

void TriangularMatrix::AddAndNormalizeTriangularColumn(
    const SparseColumn& column, RowIndex diagonal_row,
    Fractional diagonal_coefficient) {
  // TODO(user): use division by a constant using multiplication.
  for (const SparseColumn::Entry e : column) {
    if (e.row() != diagonal_row) {
      if (e.coefficient() != 0.0) {
        rows_.push_back(e.row());
        coefficients_.push_back(e.coefficient() / diagonal_coefficient);
      }
    } else {
      DCHECK_EQ(e.coefficient(), diagonal_coefficient);
    }
  }
  CloseCurrentColumn(1.0);
}

void TriangularMatrix::AddTriangularColumnWithGivenDiagonalEntry(
    const SparseColumn& column, RowIndex diagonal_row,
    Fractional diagonal_value) {
  for (SparseColumn::Entry e : column) {
    DCHECK_NE(e.row(), diagonal_row);
    rows_.push_back(e.row());
    coefficients_.push_back(e.coefficient());
  }
  CloseCurrentColumn(diagonal_value);
}

void TriangularMatrix::PopulateFromTriangularSparseMatrix(
    const SparseMatrix& input) {
  Reset(input.num_rows(), input.num_cols());
  for (ColIndex col(0); col < input.num_cols(); ++col) {
    AddTriangularColumn(ColumnView(input.column(col)), ColToRowIndex(col));
  }
  DCHECK(IsLowerTriangular() || IsUpperTriangular());
}

bool TriangularMatrix::IsLowerTriangular() const {
  for (ColIndex col(0); col < num_cols_; ++col) {
    if (diagonal_coefficients_[col] == 0.0) return false;
    for (EntryIndex i : Column(col)) {
      if (rows_[i] <= ColToRowIndex(col)) return false;
    }
  }
  return true;
}

bool TriangularMatrix::IsUpperTriangular() const {
  for (ColIndex col(0); col < num_cols_; ++col) {
    if (diagonal_coefficients_[col] == 0.0) return false;
    for (EntryIndex i : Column(col)) {
      if (rows_[i] >= ColToRowIndex(col)) return false;
    }
  }
  return true;
}

void TriangularMatrix::ApplyRowPermutationToNonDiagonalEntries(
    const RowPermutation& row_perm) {
  EntryIndex num_entries = rows_.size();
  for (EntryIndex i(0); i < num_entries; ++i) {
    rows_[i] = row_perm[rows_[i]];
  }
}

void TriangularMatrix::CopyColumnToSparseColumn(ColIndex col,
                                                SparseColumn* output) const {
  output->Clear();
  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  for (const EntryIndex i : Column(col)) {
    output->SetCoefficient(entry_rows[i], entry_coefficients[i]);
  }
  output->SetCoefficient(ColToRowIndex(col), diagonal_coefficients_[col]);
  output->CleanUp();
}

void TriangularMatrix::CopyToSparseMatrix(SparseMatrix* output) const {
  output->PopulateFromZero(num_rows_, num_cols_);
  for (ColIndex col(0); col < num_cols_; ++col) {
    CopyColumnToSparseColumn(col, output->mutable_column(col));
  }
}

void TriangularMatrix::LowerSolve(DenseColumn* rhs) const {
  LowerSolveStartingAt(ColIndex(0), rhs);
}

void TriangularMatrix::LowerSolveStartingAt(ColIndex start,
                                            DenseColumn* rhs) const {
  RETURN_IF_NULL(rhs);
  if (all_diagonal_coefficients_are_one_) {
    LowerSolveStartingAtInternal<true>(start, rhs->view());
  } else {
    LowerSolveStartingAtInternal<false>(start, rhs->view());
  }
}

template <bool diagonal_of_ones>
void TriangularMatrix::LowerSolveStartingAtInternal(
    ColIndex start, DenseColumn::View rhs) const {
  const ColIndex begin = std::max(start, first_non_identity_column_);
  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  const auto diagonal_coefficients = diagonal_coefficients_.view();
  const ColIndex end = diagonal_coefficients.size();
  for (ColIndex col(begin); col < end; ++col) {
    const Fractional value = rhs[ColToRowIndex(col)];
    if (value == 0.0) continue;
    const Fractional coeff =
        diagonal_of_ones ? value : value / diagonal_coefficients[col];
    if (!diagonal_of_ones) {
      rhs[ColToRowIndex(col)] = coeff;
    }
    for (const EntryIndex i : Column(col)) {
      rhs[entry_rows[i]] -= coeff * entry_coefficients[i];
    }
  }
}

void TriangularMatrix::UpperSolve(DenseColumn* rhs) const {
  RETURN_IF_NULL(rhs);
  if (all_diagonal_coefficients_are_one_) {
    UpperSolveInternal<true>(rhs->view());
  } else {
    UpperSolveInternal<false>(rhs->view());
  }
}

template <bool diagonal_of_ones>
void TriangularMatrix::UpperSolveInternal(DenseColumn::View rhs) const {
  const ColIndex end = first_non_identity_column_;
  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  const auto diagonal_coefficients = diagonal_coefficients_.view();
  const auto starts = starts_.view();
  for (ColIndex col(diagonal_coefficients.size() - 1); col >= end; --col) {
    const Fractional value = rhs[ColToRowIndex(col)];
    if (value == 0.0) continue;
    const Fractional coeff =
        diagonal_of_ones ? value : value / diagonal_coefficients[col];
    if (!diagonal_of_ones) {
      rhs[ColToRowIndex(col)] = coeff;
    }

    // It is faster to iterate this way (instead of i : Column(col)) because of
    // cache locality. Note that the floating-point computations are exactly the
    // same in both cases.
    const EntryIndex i_end = starts[col];
    for (EntryIndex i(starts[col + 1] - 1); i >= i_end; --i) {
      rhs[entry_rows[i]] -= coeff * entry_coefficients[i];
    }
  }
}

void TriangularMatrix::TransposeUpperSolve(DenseColumn* rhs) const {
  RETURN_IF_NULL(rhs);
  if (all_diagonal_coefficients_are_one_) {
    TransposeUpperSolveInternal<true>(rhs->view());
  } else {
    TransposeUpperSolveInternal<false>(rhs->view());
  }
}

template <bool diagonal_of_ones>
void TriangularMatrix::TransposeUpperSolveInternal(
    DenseColumn::View rhs) const {
  const ColIndex end = num_cols_;
  const auto starts = starts_.view();
  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  const auto diagonal_coefficients = diagonal_coefficients_.view();

  EntryIndex i = starts_[first_non_identity_column_];
  for (ColIndex col(first_non_identity_column_); col < end; ++col) {
    Fractional sum = rhs[ColToRowIndex(col)];

    // Note that this is a bit faster than the simpler
    //     for (const EntryIndex i : Column(col)) {
    // EntryIndex i is explicitly not modified in outer iterations, since
    // the last entry in column col is stored contiguously just before the
    // first entry in column col+1.
    const EntryIndex i_end = starts[col + 1];
    const EntryIndex shifted_end = i_end - 3;
    for (; i < shifted_end; i += 4) {
      sum -= entry_coefficients[i] * rhs[entry_rows[i]] +
             entry_coefficients[i + 1] * rhs[entry_rows[i + 1]] +
             entry_coefficients[i + 2] * rhs[entry_rows[i + 2]] +
             entry_coefficients[i + 3] * rhs[entry_rows[i + 3]];
    }
    if (i < i_end) {
      sum -= entry_coefficients[i] * rhs[entry_rows[i]];
      if (i + 1 < i_end) {
        sum -= entry_coefficients[i + 1] * rhs[entry_rows[i + 1]];
        if (i + 2 < i_end) {
          sum -= entry_coefficients[i + 2] * rhs[entry_rows[i + 2]];
        }
      }
      i = i_end;
    }

    rhs[ColToRowIndex(col)] =
        diagonal_of_ones ? sum : sum / diagonal_coefficients[col];
  }
}

void TriangularMatrix::TransposeLowerSolve(DenseColumn* rhs) const {
  RETURN_IF_NULL(rhs);
  if (all_diagonal_coefficients_are_one_) {
    TransposeLowerSolveInternal<true>(rhs->view());
  } else {
    TransposeLowerSolveInternal<false>(rhs->view());
  }
}

template <bool diagonal_of_ones>
void TriangularMatrix::TransposeLowerSolveInternal(
    DenseColumn::View rhs) const {
  const ColIndex end = first_non_identity_column_;

  // We optimize a bit the solve by skipping the last 0.0 positions.
  ColIndex col = num_cols_ - 1;
  while (col >= end && rhs[ColToRowIndex(col)] == 0.0) {
    --col;
  }

  const auto starts = starts_.view();
  const auto diagonal_coeffs = diagonal_coefficients_.view();
  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  EntryIndex i = starts[col + 1] - 1;
  for (; col >= end; --col) {
    Fractional sum = rhs[ColToRowIndex(col)];

    // Note that this is a bit faster than the simpler
    //     for (const EntryIndex i : Column(col)) {
    // mainly because we iterate in a good direction for the cache.
    // EntryIndex i is explicitly not modified in outer iterations, since
    // the last entry in column col is stored contiguously just before the
    // first entry in column col+1.
    const EntryIndex i_end = starts[col];
    const EntryIndex shifted_end = i_end + 3;
    for (; i >= shifted_end; i -= 4) {
      sum -= entry_coefficients[i] * rhs[entry_rows[i]] +
             entry_coefficients[i - 1] * rhs[entry_rows[i - 1]] +
             entry_coefficients[i - 2] * rhs[entry_rows[i - 2]] +
             entry_coefficients[i - 3] * rhs[entry_rows[i - 3]];
    }
    if (i >= i_end) {
      sum -= entry_coefficients[i] * rhs[entry_rows[i]];
      if (i >= i_end + 1) {
        sum -= entry_coefficients[i - 1] * rhs[entry_rows[i - 1]];
        if (i >= i_end + 2) {
          sum -= entry_coefficients[i - 2] * rhs[entry_rows[i - 2]];
        }
      }
      i = i_end - 1;
    }

    rhs[ColToRowIndex(col)] =
        diagonal_of_ones ? sum : sum / diagonal_coeffs[col];
  }
}

void TriangularMatrix::HyperSparseSolve(DenseColumn* rhs,
                                        RowIndexVector* non_zero_rows) const {
  RETURN_IF_NULL(rhs);
  if (all_diagonal_coefficients_are_one_) {
    HyperSparseSolveInternal<true>(rhs->view(), non_zero_rows);
  } else {
    HyperSparseSolveInternal<false>(rhs->view(), non_zero_rows);
  }
}

template <bool diagonal_of_ones>
void TriangularMatrix::HyperSparseSolveInternal(
    DenseColumn::View rhs, RowIndexVector* non_zero_rows) const {
  int new_size = 0;
  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  for (const RowIndex row : *non_zero_rows) {
    if (rhs[row] == 0.0) continue;
    const ColIndex row_as_col = RowToColIndex(row);
    const Fractional coeff =
        diagonal_of_ones ? rhs[row]
                         : rhs[row] / diagonal_coefficients_[row_as_col];
    rhs[row] = coeff;
    for (const EntryIndex i : Column(row_as_col)) {
      rhs[entry_rows[i]] -= coeff * entry_coefficients[i];
    }
    (*non_zero_rows)[new_size] = row;
    ++new_size;
  }
  non_zero_rows->resize(new_size);
}

void TriangularMatrix::HyperSparseSolveWithReversedNonZeros(
    DenseColumn* rhs, RowIndexVector* non_zero_rows) const {
  RETURN_IF_NULL(rhs);
  if (all_diagonal_coefficients_are_one_) {
    HyperSparseSolveWithReversedNonZerosInternal<true>(rhs->view(),
                                                       non_zero_rows);
  } else {
    HyperSparseSolveWithReversedNonZerosInternal<false>(rhs->view(),
                                                        non_zero_rows);
  }
}

template <bool diagonal_of_ones>
void TriangularMatrix::HyperSparseSolveWithReversedNonZerosInternal(
    DenseColumn::View rhs, RowIndexVector* non_zero_rows) const {
  int new_start = non_zero_rows->size();
  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  for (const RowIndex row : Reverse(*non_zero_rows)) {
    if (rhs[row] == 0.0) continue;
    const ColIndex row_as_col = RowToColIndex(row);
    const Fractional coeff =
        diagonal_of_ones ? rhs[row]
                         : rhs[row] / diagonal_coefficients_[row_as_col];
    rhs[row] = coeff;
    for (const EntryIndex i : Column(row_as_col)) {
      rhs[entry_rows[i]] -= coeff * entry_coefficients[i];
    }
    --new_start;
    (*non_zero_rows)[new_start] = row;
  }
  non_zero_rows->erase(non_zero_rows->begin(),
                       non_zero_rows->begin() + new_start);
}

void TriangularMatrix::TransposeHyperSparseSolve(
    DenseColumn* rhs, RowIndexVector* non_zero_rows) const {
  RETURN_IF_NULL(rhs);
  if (all_diagonal_coefficients_are_one_) {
    TransposeHyperSparseSolveInternal<true>(rhs->view(), non_zero_rows);
  } else {
    TransposeHyperSparseSolveInternal<false>(rhs->view(), non_zero_rows);
  }
}

template <bool diagonal_of_ones>
void TriangularMatrix::TransposeHyperSparseSolveInternal(
    DenseColumn::View rhs, RowIndexVector* non_zero_rows) const {
  int new_size = 0;

  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  for (const RowIndex row : *non_zero_rows) {
    Fractional sum = rhs[row];
    const ColIndex row_as_col = RowToColIndex(row);

    // Note that we do the loop in exactly the same way as
    // in TransposeUpperSolveInternal().
    EntryIndex i = starts_[row_as_col];
    const EntryIndex i_end = starts_[row_as_col + 1];
    const EntryIndex shifted_end = i_end - 3;
    for (; i < shifted_end; i += 4) {
      sum -= entry_coefficients[i] * rhs[entry_rows[i]] +
             entry_coefficients[i + 1] * rhs[entry_rows[i + 1]] +
             entry_coefficients[i + 2] * rhs[entry_rows[i + 2]] +
             entry_coefficients[i + 3] * rhs[entry_rows[i + 3]];
    }
    if (i < i_end) {
      sum -= entry_coefficients[i] * rhs[entry_rows[i]];
      if (i + 1 < i_end) {
        sum -= entry_coefficients[i + 1] * rhs[entry_rows[i + 1]];
        if (i + 2 < i_end) {
          sum -= entry_coefficients[i + 2] * rhs[entry_rows[i + 2]];
        }
      }
    }

    rhs[row] =
        diagonal_of_ones ? sum : sum / diagonal_coefficients_[row_as_col];
    if (sum != 0.0) {
      (*non_zero_rows)[new_size] = row;
      ++new_size;
    }
  }
  non_zero_rows->resize(new_size);
}

void TriangularMatrix::TransposeHyperSparseSolveWithReversedNonZeros(
    DenseColumn* rhs, RowIndexVector* non_zero_rows) const {
  RETURN_IF_NULL(rhs);
  if (all_diagonal_coefficients_are_one_) {
    TransposeHyperSparseSolveWithReversedNonZerosInternal<true>(rhs->view(),
                                                                non_zero_rows);
  } else {
    TransposeHyperSparseSolveWithReversedNonZerosInternal<false>(rhs->view(),
                                                                 non_zero_rows);
  }
}

template <bool diagonal_of_ones>
void TriangularMatrix::TransposeHyperSparseSolveWithReversedNonZerosInternal(
    DenseColumn::View rhs, RowIndexVector* non_zero_rows) const {
  int new_start = non_zero_rows->size();
  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  for (const RowIndex row : Reverse(*non_zero_rows)) {
    Fractional sum = rhs[row];
    const ColIndex row_as_col = RowToColIndex(row);

    // We do the loop this way so that the floating point operations are exactly
    // the same as the ones performed by TransposeLowerSolveInternal().
    EntryIndex i = starts_[row_as_col + 1] - 1;
    const EntryIndex i_end = starts_[row_as_col];
    const EntryIndex shifted_end = i_end + 3;
    for (; i >= shifted_end; i -= 4) {
      sum -= entry_coefficients[i] * rhs[entry_rows[i]] +
             entry_coefficients[i - 1] * rhs[entry_rows[i - 1]] +
             entry_coefficients[i - 2] * rhs[entry_rows[i - 2]] +
             entry_coefficients[i - 3] * rhs[entry_rows[i - 3]];
    }
    if (i >= i_end) {
      sum -= entry_coefficients[i] * rhs[entry_rows[i]];
      if (i >= i_end + 1) {
        sum -= entry_coefficients[i - 1] * rhs[entry_rows[i - 1]];
        if (i >= i_end + 2) {
          sum -= entry_coefficients[i - 2] * rhs[entry_rows[i - 2]];
        }
      }
    }

    rhs[row] =
        diagonal_of_ones ? sum : sum / diagonal_coefficients_[row_as_col];
    if (sum != 0.0) {
      --new_start;
      (*non_zero_rows)[new_start] = row;
    }
  }
  non_zero_rows->erase(non_zero_rows->begin(),
                       non_zero_rows->begin() + new_start);
}

void TriangularMatrix::PermutedLowerSolve(
    const SparseColumn& rhs, const RowPermutation& row_perm,
    const RowMapping& partial_inverse_row_perm, SparseColumn* lower,
    SparseColumn* upper) const {
  DCHECK(all_diagonal_coefficients_are_one_);
  RETURN_IF_NULL(lower);
  RETURN_IF_NULL(upper);

  initially_all_zero_scratchpad_.resize(num_rows_, 0.0);
  for (const SparseColumn::Entry e : rhs) {
    initially_all_zero_scratchpad_[e.row()] = e.coefficient();
  }

  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  const RowIndex end_row(partial_inverse_row_perm.size());
  for (RowIndex row(ColToRowIndex(first_non_identity_column_)); row < end_row;
       ++row) {
    const RowIndex permuted_row = partial_inverse_row_perm[row];
    const Fractional pivot = initially_all_zero_scratchpad_[permuted_row];
    if (pivot == 0.0) continue;

    for (EntryIndex i : Column(RowToColIndex(row))) {
      initially_all_zero_scratchpad_[entry_rows[i]] -=
          entry_coefficients[i] * pivot;
    }
  }

  lower->Clear();
  const RowIndex num_rows = num_rows_;
  for (RowIndex row(0); row < num_rows; ++row) {
    if (initially_all_zero_scratchpad_[row] != 0.0) {
      if (row_perm[row] < 0) {
        lower->SetCoefficient(row, initially_all_zero_scratchpad_[row]);
      } else {
        upper->SetCoefficient(row, initially_all_zero_scratchpad_[row]);
      }
      initially_all_zero_scratchpad_[row] = 0.0;
    }
  }
  DCHECK(lower->CheckNoDuplicates());
}

void TriangularMatrix::PermutedLowerSparseSolve(const ColumnView& rhs,
                                                const RowPermutation& row_perm,
                                                SparseColumn* lower_column,
                                                SparseColumn* upper_column) {
  DCHECK(all_diagonal_coefficients_are_one_);
  RETURN_IF_NULL(lower_column);
  RETURN_IF_NULL(upper_column);

  // Compute the set of rows that will be non zero in the result (lower_column,
  // upper_column).
  PermutedComputeRowsToConsider(rhs, row_perm, &lower_column_rows_,
                                &upper_column_rows_);

  // Copy rhs into initially_all_zero_scratchpad_.
  initially_all_zero_scratchpad_.resize(num_rows_, 0.0);
  for (const auto e : rhs) {
    initially_all_zero_scratchpad_[e.row()] = e.coefficient();
  }

  // We clear lower_column first in case upper_column and lower_column point to
  // the same underlying SparseColumn.
  num_fp_operations_ = 0;
  lower_column->Clear();

  // rows_to_consider_ contains the row to process in reverse order. Note in
  // particular that each "permuted_row" will never be touched again and so its
  // value is final. We copy the result in (lower_column, upper_column) and
  // clear initially_all_zero_scratchpad_ at the same time.
  upper_column->Reserve(upper_column->num_entries() +
                        EntryIndex(upper_column_rows_.size()));
  for (const RowIndex permuted_row : Reverse(upper_column_rows_)) {
    const Fractional pivot = initially_all_zero_scratchpad_[permuted_row];
    if (pivot == 0.0) continue;
    // Note that permuted_row will not appear in the loop below so we
    // already know the value of the solution at this position.
    initially_all_zero_scratchpad_[permuted_row] = 0.0;
    const ColIndex row_as_col = RowToColIndex(row_perm[permuted_row]);
    DCHECK_GE(row_as_col, 0);
    upper_column->SetCoefficient(permuted_row, pivot);
    DCHECK_EQ(diagonal_coefficients_[row_as_col], 1.0);
    num_fp_operations_ += 1 + ColumnNumEntries(row_as_col).value();
    for (const auto e : column(row_as_col)) {
      initially_all_zero_scratchpad_[e.row()] -= e.coefficient() * pivot;
    }
  }

  // TODO(user): The size of lower is exact, so we could be slighly faster here.
  lower_column->Reserve(EntryIndex(lower_column_rows_.size()));
  for (const RowIndex permuted_row : lower_column_rows_) {
    const Fractional pivot = initially_all_zero_scratchpad_[permuted_row];
    initially_all_zero_scratchpad_[permuted_row] = 0.0;
    lower_column->SetCoefficient(permuted_row, pivot);
  }
  DCHECK(lower_column->CheckNoDuplicates());
  DCHECK(upper_column->CheckNoDuplicates());
}

// The goal is to find which rows of the working column we will need to look
// at in PermutedLowerSparseSolve() when solving P^{-1}.L.P.x = rhs, 'P' being a
// row permutation, 'L' a lower triangular matrix and 'this' being 'P^{-1}.L'.
// Note that the columns of L that are identity columns (this is the case for
// the ones corresponding to a kNonPivotal in P) can be skipped since they will
// leave the working column unchanged.
//
// Let G denote the graph G = (V,E) of the column-to-row adjacency of A:
// - 'V' is the set of nodes, one node i corresponds to a both a row
//    and a column (the matrix is square).
// - 'E' is the set of arcs. There is an arc from node i to node j iff the
//    coefficient of i-th column, j-th row of A = P^{-1}.L.P is non zero.
//
// Let S denote the set of nodes i such that rhs_i != 0.
// Let R denote the set of all accessible nodes from S in G.
// x_k is possibly non-zero iff k is in R, i.e. if k is not in R then x_k = 0
// for sure, and there is no need to look a the row k during the solve.
//
// So, to solve P^{-1}.L.P.x = rhs, only rows corresponding to P.R have to be
// considered (ignoring the one that map to identity column of L). A topological
// sort of P.R is used to decide in which order one should iterate on them. This
// will be given by upper_column_rows_ and it will be populated in reverse
// order.
void TriangularMatrix::PermutedComputeRowsToConsider(
    const ColumnView& rhs, const RowPermutation& row_perm,
    RowIndexVector* lower_column_rows, RowIndexVector* upper_column_rows) {
  stored_.Resize(num_rows_);
  marked_.resize(num_rows_, false);
  lower_column_rows->clear();
  upper_column_rows->clear();
  nodes_to_explore_.clear();

  for (SparseColumn::Entry e : rhs) {
    const ColIndex col = RowToColIndex(row_perm[e.row()]);
    if (col < 0) {
      stored_.Set(e.row());
      lower_column_rows->push_back(e.row());
    } else {
      nodes_to_explore_.push_back(e.row());
    }
  }

  // Topological sort based on Depth-First-Search.
  // A few notes:
  // - By construction, if the matrix can be permuted into a lower triangular
  //   form, there is no cycle. This code does nothing to test for cycles, but
  //   there is a DCHECK() to detect them during debugging.
  // - This version uses sentinels (kInvalidRow) on nodes_to_explore_ to know
  //   when a node has been explored (i.e. when the recursive dfs goes back in
  //   the call stack). This is faster than an alternate implementation that
  //   uses another Boolean array to detect when we go back in the
  //   depth-first search.
  const auto entry_rows = rows_.view();
  while (!nodes_to_explore_.empty()) {
    const RowIndex row = nodes_to_explore_.back();

    // If the depth-first search from the current node is finished (i.e. there
    // is a sentinel on the stack), we store the node (which is just before on
    // the stack). This will store the nodes in reverse topological order.
    if (row < 0) {
      nodes_to_explore_.pop_back();
      const RowIndex explored_row = nodes_to_explore_.back();
      nodes_to_explore_.pop_back();
      DCHECK(!stored_[explored_row]);
      stored_.Set(explored_row);
      upper_column_rows->push_back(explored_row);

      // Unmark and prune the nodes that are already unmarked. See the header
      // comment on marked_ for the algorithm description.
      //
      // Complexity note: The only difference with the "normal" DFS doing no
      // pruning is this extra loop here and the marked_[entry_row] = true in
      // the loop later in this function. On an already pruned graph, this is
      // probably between 1 and 2 times slower than the "normal" DFS.
      const ColIndex col = RowToColIndex(row_perm[explored_row]);
      EntryIndex i = starts_[col];
      EntryIndex end = pruned_ends_[col];
      while (i < end) {
        const RowIndex entry_row = entry_rows[i];
        if (!marked_[entry_row]) {
          --end;

          // Note that we could keep the pruned row in a separate vector and
          // not touch the triangular matrix. But the current solution seems
          // better cache-wise and memory-wise.
          std::swap(rows_[i], rows_[end]);
          std::swap(coefficients_[i], coefficients_[end]);
        } else {
          marked_[entry_row] = false;
          ++i;
        }
      }
      pruned_ends_[col] = end;
      continue;
    }

    // If the node is already stored, skip.
    if (stored_[row]) {
      nodes_to_explore_.pop_back();
      continue;
    }

    // Expand only if we are not on a kNonPivotal row.
    // Otherwise we can store the node right away.
    const ColIndex col = RowToColIndex(row_perm[row]);
    if (col < 0) {
      stored_.Set(row);
      lower_column_rows->push_back(row);
      nodes_to_explore_.pop_back();
      continue;
    }

    // Go one level forward in the depth-first search, and store the 'adjacent'
    // node on nodes_to_explore_ for further processing.
    nodes_to_explore_.push_back(kInvalidRow);
    const EntryIndex end = pruned_ends_[col];
    for (EntryIndex i = starts_[col]; i < end; ++i) {
      const RowIndex entry_row = entry_rows[i];
      if (!stored_[entry_row]) {
        nodes_to_explore_.push_back(entry_row);
      }
      marked_[entry_row] = true;
    }

    // The graph contains cycles? this is not supposed to happen.
    DCHECK_LE(nodes_to_explore_.size(), 2 * num_rows_.value() + rows_.size());
  }

  // Clear stored_.
  for (const RowIndex row : *lower_column_rows) {
    stored_.ClearBucket(row);
  }
  for (const RowIndex row : *upper_column_rows) {
    stored_.ClearBucket(row);
  }
}

void TriangularMatrix::ComputeRowsToConsiderWithDfs(
    RowIndexVector* non_zero_rows) const {
  if (non_zero_rows->empty()) return;

  // We don't start the DFS if the initial number of non-zeros is under the
  // sparsity_threshold. During the DFS, we abort it if the number of floating
  // points operations get larger than the num_ops_threshold.
  //
  // In both cases, we make sure to clear non_zero_rows so that the solving part
  // will use the non-hypersparse version of the code.
  //
  // TODO(user): Investigate the best thresholds.
  const int sparsity_threshold =
      static_cast<int>(0.025 * static_cast<double>(num_rows_.value()));
  const int num_ops_threshold =
      static_cast<int>(0.05 * static_cast<double>(num_rows_.value()));
  int num_ops = non_zero_rows->size();
  if (num_ops > sparsity_threshold) {
    non_zero_rows->clear();
    return;
  }

  // Initialize using the non-zero positions of the input.
  stored_.Resize(num_rows_);
  nodes_to_explore_.clear();
  nodes_to_explore_.swap(*non_zero_rows);

  // Topological sort based on Depth-First-Search.
  // Same remarks as the version implemented in PermutedComputeRowsToConsider().
  const auto entry_rows = rows_.view();
  while (!nodes_to_explore_.empty()) {
    const RowIndex row = nodes_to_explore_.back();

    // If the depth-first search from the current node is finished, we store the
    // node. This will store the node in reverse topological order.
    if (row < 0) {
      nodes_to_explore_.pop_back();
      const RowIndex explored_row = -row - 1;
      stored_.Set(explored_row);
      non_zero_rows->push_back(explored_row);
      continue;
    }

    // If the node is already stored, skip.
    if (stored_[row]) {
      nodes_to_explore_.pop_back();
      continue;
    }

    // Go one level forward in the depth-first search, and store the 'adjacent'
    // node on nodes_to_explore_ for further processing.
    //
    // We reverse the sign of nodes_to_explore_.back() to detect when the
    // DFS will be back on this node.
    nodes_to_explore_.back() = -row - 1;
    for (const EntryIndex i : Column(RowToColIndex(row))) {
      ++num_ops;
      const RowIndex entry_row = entry_rows[i];
      if (!stored_[entry_row]) {
        nodes_to_explore_.push_back(entry_row);
      }
    }

    // Abort if the number of operations is not negligible compared to the
    // number of rows. Note that this test also prevents the code from cycling
    // in case the matrix is actually not triangular.
    if (num_ops > num_ops_threshold) break;
  }

  // Clear stored_.
  for (const RowIndex row : *non_zero_rows) {
    stored_.ClearBucket(row);
  }

  // If we aborted, clear the result.
  if (num_ops > num_ops_threshold) non_zero_rows->clear();
}

void TriangularMatrix::ComputeRowsToConsiderInSortedOrder(
    RowIndexVector* non_zero_rows) const {
  if (non_zero_rows->empty()) return;

  // TODO(user): Investigate the best thresholds.
  const int sparsity_threshold =
      static_cast<int>(0.025 * static_cast<double>(num_rows_.value()));
  const int num_ops_threshold =
      static_cast<int>(0.05 * static_cast<double>(num_rows_.value()));
  int num_ops = non_zero_rows->size();
  if (num_ops > sparsity_threshold) {
    non_zero_rows->clear();
    return;
  }

  stored_.Resize(num_rows_);
  Bitset64<RowIndex>::View stored = stored_.view();
  for (const RowIndex row : *non_zero_rows) {
    stored.Set(row);
  }

  const auto matrix_view = view();
  const auto entry_rows = rows_.view();
  for (int i = 0; i < non_zero_rows->size(); ++i) {
    const RowIndex row = (*non_zero_rows)[i];
    for (const EntryIndex index : matrix_view.Column(RowToColIndex(row))) {
      ++num_ops;
      const RowIndex entry_row = entry_rows[index];
      if (!stored[entry_row]) {
        non_zero_rows->push_back(entry_row);
        stored.Set(entry_row);
      }
    }
    if (num_ops > num_ops_threshold) break;
  }

  if (num_ops > num_ops_threshold) {
    stored_.ClearAll();
    non_zero_rows->clear();
  } else {
    std::sort(non_zero_rows->begin(), non_zero_rows->end());
    for (const RowIndex row : *non_zero_rows) {
      stored_.ClearBucket(row);
    }
  }
}

// A known upper bound for the infinity norm of T^{-1} is the
// infinity norm of y where T'*y = x with:
// - x the all 1s vector.
// - Each entry in T' is the absolute value of the same entry in T.
Fractional TriangularMatrix::ComputeInverseInfinityNormUpperBound() const {
  if (first_non_identity_column_ == num_cols_) {
    // Identity matrix
    return 1.0;
  }

  const bool is_upper = IsUpperTriangular();
  DenseColumn row_norm_estimate(num_rows_, 1.0);
  const int num_cols = num_cols_.value();

  const auto entry_rows = rows_.view();
  const auto entry_coefficients = coefficients_.view();
  for (int i = 0; i < num_cols; ++i) {
    const ColIndex col(is_upper ? num_cols - 1 - i : i);
    DCHECK_NE(diagonal_coefficients_[col], 0.0);
    const Fractional coeff = row_norm_estimate[ColToRowIndex(col)] /
                             std::abs(diagonal_coefficients_[col]);

    row_norm_estimate[ColToRowIndex(col)] = coeff;
    for (const EntryIndex i : Column(col)) {
      row_norm_estimate[entry_rows[i]] +=
          coeff * std::abs(entry_coefficients[i]);
    }
  }

  return *std::max_element(row_norm_estimate.begin(), row_norm_estimate.end());
}

Fractional TriangularMatrix::ComputeInverseInfinityNorm() const {
  const bool is_upper = IsUpperTriangular();

  DenseColumn row_sum(num_rows_, 0.0);
  DenseColumn right_hand_side;
  for (ColIndex col(0); col < num_cols_; ++col) {
    right_hand_side.assign(num_rows_, 0);
    right_hand_side[ColToRowIndex(col)] = 1.0;

    // Get the col-th column of the matrix inverse.
    if (is_upper) {
      UpperSolve(&right_hand_side);
    } else {
      LowerSolve(&right_hand_side);
    }

    // Compute sum_j |inverse_ij|.
    for (RowIndex row(0); row < num_rows_; ++row) {
      row_sum[row] += std::abs(right_hand_side[row]);
    }
  }
  // Compute max_i sum_j |inverse_ij|.
  Fractional norm = 0.0;
  for (RowIndex row(0); row < num_rows_; ++row) {
    norm = std::max(norm, row_sum[row]);
  }

  return norm;
}
}  // namespace glop
}  // namespace operations_research
