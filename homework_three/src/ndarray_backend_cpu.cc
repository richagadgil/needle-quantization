#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
typedef int8_t quant_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
const size_t QUANT_SIZE = sizeof(quant_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};


struct AlignedArray_quant {
  AlignedArray_quant(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * QUANT_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray_quant() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  quant_t* ptr;
  size_t size;
};


void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}


void FillQuant(AlignedArray_quant* out, quant_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}


void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION

  size_t num_dims = shape.size();
  std::vector<size_t> indices(num_dims, 0);

  size_t cnt = 0;
  size_t in_index;
  size_t total_size = 1;
  int dim = num_dims - 1;

  while (dim > -1) {
  
    in_index = offset;
    for (size_t i = 0; i < indices.size(); i++) {
      std::cout << i << " " << strides[i] << " " << indices[i] << "\n";
      in_index += (strides[i]*indices[i]);
    }

    //std::cout << num_dims << ": " << indices[0] << " " << indices[1] << " " << indices[2] << "\n";
    // std::cout << indices[0] << " " << indices[1] << " " << cnt << " " << in_index << "\n";
    out->ptr[cnt] = a.ptr[in_index];
    cnt++;

    dim = num_dims - 1;

    while (indices[dim]+1 == shape[dim]) {
      indices[dim] = 0;
      dim--;
    }

    if (dim > -1) {
      // std::cout << dim << " increment \n";
      // std::cout << indices[dim] << "\n";
      indices[dim]++;
      // std::cout << indices[dim] << "\n";
    }

  }

  /// END SOLUTION
}


void CompactQuant(const AlignedArray_quant& a, AlignedArray_quant* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION

  size_t num_dims = shape.size();
  std::vector<size_t> indices(num_dims, 0);

  size_t cnt = 0;
  size_t in_index;
  size_t total_size = 1;
  int dim = num_dims - 1;

  while (dim > -1) {
    in_index = offset;
    for (size_t i = 0; i < indices.size(); i++) {
      in_index += (strides[i]*indices[i]);
    }

    out->ptr[cnt] = a.ptr[in_index];
    cnt++;

    dim = num_dims - 1;

    while (indices[dim]+1 == shape[dim]) {
      indices[dim] = 0;
      dim--;
    }

    if (dim > -1) {
      indices[dim]++;
    }
	}
}


void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t num_dims = shape.size();
  std::vector<size_t> indices(num_dims, 0);

  size_t cnt = 0;
  size_t in_index;
  size_t total_size = 1;
  int dim = num_dims - 1;

  while (dim > -1) {
  
    in_index = offset;
    for (size_t i = 0; i < indices.size(); i++) {
      in_index += (strides[i]*indices[i]);
    }

    out->ptr[in_index] = a.ptr[cnt];
    cnt++;

    dim = num_dims - 1;

    while (indices[dim]+1 == shape[dim]) {
      indices[dim] = 0;
      dim--;
    }

    if (dim > -1) {
      indices[dim]++;
    }

  }

  /// END SOLUTION
}


void EwiseSetitemQuant(const AlignedArray_quant& a, AlignedArray_quant* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t num_dims = shape.size();
  std::vector<size_t> indices(num_dims, 0);

  size_t cnt = 0;
  size_t in_index;
  size_t total_size = 1;
  int dim = num_dims - 1;

  while (dim > -1) {
    in_index = offset;
    for (size_t i = 0; i < indices.size(); i++) {
      in_index += (strides[i]*indices[i]);
    }

    out->ptr[in_index] = a.ptr[cnt];
    cnt++;

    dim = num_dims - 1;

    while (indices[dim]+1 == shape[dim]) {
      indices[dim] = 0;
      dim--;
    }

    if (dim > -1) {
      indices[dim]++;
    }
  }
}


void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  size_t num_dims = shape.size();
  std::vector<size_t> indices(num_dims, 0);

  size_t cnt = 0;
  size_t in_index;
  size_t total_size = 1;
  int dim = num_dims - 1;

  while (dim > -1) {
  
    in_index = offset;
    for (size_t i = 0; i < indices.size(); i++) {
      in_index += (strides[i]*indices[i]);
    }

    out->ptr[in_index] = val;
    cnt++;

    dim = num_dims - 1;

    while (indices[dim]+1 == shape[dim]) {
      indices[dim] = 0;
      dim--;
    }

    if (dim > -1) {
      indices[dim]++;
    }

  }
  /// END SOLUTION
}


void ScalarSetitemQuant(const size_t size, scalar_t val, AlignedArray_quant* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  size_t num_dims = shape.size();
  std::vector<size_t> indices(num_dims, 0);

  size_t cnt = 0;
  size_t in_index;
  size_t total_size = 1;
  int dim = num_dims - 1;

  while (dim > -1) {
    in_index = offset;
    for (size_t i = 0; i < indices.size(); i++) {
      in_index += (strides[i]*indices[i]);
    }

    out->ptr[in_index] = val;
    cnt++;

    dim = num_dims - 1;

    while (indices[dim]+1 == shape[dim]) {
      indices[dim] = 0;
      dim--;
    }

    if (dim > -1) {
      indices[dim]++;
    }

  }
}


void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}


void EwiseAddQuant(const AlignedArray_quant& a, const AlignedArray_quant& b, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}


void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


void ScalarAddQuant(const AlignedArray_quant& a, scalar_t val, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}


void EwiseMulQuant(const AlignedArray_quant& a, const AlignedArray_quant& b, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}


void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}


void ScalarMulQuant(const AlignedArray_quant& a, scalar_t val, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}


void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}


void EwiseDivQuant(const AlignedArray_quant& a, const AlignedArray_quant& b, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}


void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}


void ScalarDivQuant(const AlignedArray_quant& a, scalar_t val, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}


void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}


void ScalarPowerQuant(const AlignedArray_quant& a, scalar_t val, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}


void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}


void EwiseMaximumQuant(const AlignedArray_quant& a, const AlignedArray_quant& b, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max<quant_t>(a.ptr[i], b.ptr[i]);
  }
}


void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void ScalarMaximumQuant(const AlignedArray_quant& a, scalar_t val, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max<quant_t>(a.ptr[i], val);
  }
}


void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == b.ptr[i]);
  }
}


void EwiseEqQuant(const AlignedArray_quant& a, const AlignedArray_quant& b, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == b.ptr[i]);
  }
}


void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == val);
  }
}


void ScalarEqQuant(const AlignedArray_quant& a, scalar_t val, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == val);
  }
}


void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= b.ptr[i]);
  }
}


void EwiseGeQuant(const AlignedArray_quant& a, const AlignedArray_quant& b, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= b.ptr[i]);
  }
}


void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= val);
  }
}

void ScalarGeQuant(const AlignedArray_quant& a, scalar_t val, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= val);
  }
}


void EwiseLog(const AlignedArray& a,  AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}


void EwiseLogQuant(const AlignedArray_quant& a,  AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}


void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}


void EwiseExpQuant(const AlignedArray_quant& a, AlignedArray_quant* out) {
	for (size_t i = 0; i < a.size; i++) {
		out->ptr[i] = std::exp(a.ptr[i]);
	}
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}


void EwiseTanhQuant(const AlignedArray_quant& a, AlignedArray_quant* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      out->ptr[i * p + j] = 0;  // Assuming row-major ordering.
    }
  }

  // Naive matrix multiplication.
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      for (size_t k = 0; k < n; ++k) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}


void MatmulQuant(const AlignedArray_quant& a, const AlignedArray_quant& b, AlignedArray_quant* out,
                 uint32_t m, uint32_t n, uint32_t p) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      out->ptr[i * p + j] = 0;  // Assuming row-major ordering.
    }
  }

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      for (size_t k = 0; k < n; ++k) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < TILE; ++i) {
    for (uint32_t j = 0; j < TILE; ++j) {
      for (uint32_t k = 0; k < TILE; ++k) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}


inline void AlignedDotQuant(const int8_t* __restrict__ a,
                            const int8_t* __restrict__ b,
                            int8_t* __restrict__ out) {

  a = (const int8_t*)__builtin_assume_aligned(a, TILE * QUANT_SIZE);
  b = (const int8_t*)__builtin_assume_aligned(b, TILE * QUANT_SIZE);

  for (uint32_t i = 0; i < TILE; ++i) {
    for (uint32_t j = 0; j < TILE; ++j) {
      for (uint32_t k = 0; k < TILE; ++k) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
}


void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  const size_t m_tiles = m / TILE;
  const size_t n_tiles = n / TILE;
  const size_t p_tiles = p / TILE;

  // For each tile-row from matrix a
  for (size_t i = 0; i < m_tiles; ++i) {
      // For each tile-column from matrix b
      for (size_t j = 0; j < p_tiles; ++j) {
          // Initialize the result tile out[i][j] to zero
          for (size_t x = 0; x < TILE; ++x) {
              for (size_t y = 0; y < TILE; ++y) {
                  out->ptr[(i * p_tiles + j) * TILE * TILE + x * TILE + y] = 0.0f;
              }
          }
          // For each kth tile from matrix a and matrix b
          for (size_t k = 0; k < n_tiles; ++k) {
              const float* a_tile = &a.ptr[(i * n_tiles + k) * TILE * TILE];
              const float* b_tile = &b.ptr[(k * p_tiles + j) * TILE * TILE];
              float* out_tile = &out->ptr[(i * p_tiles + j) * TILE * TILE];

              // Multiply a_tile and b_tile, and add the result to out_tile
              AlignedDot(a_tile, b_tile, out_tile);
          }
      }
  }
  /// END SOLUTION
}


void MatmulTiledQuant(const AlignedArray_quant& a, const AlignedArray_quant& b,
                      AlignedArray_quant* out, uint32_t m, uint32_t n, uint32_t p) {
  const size_t m_tiles = m / TILE;
  const size_t n_tiles = n / TILE;
  const size_t p_tiles = p / TILE;

  for (size_t i = 0; i < m_tiles; ++i) {
      for (size_t j = 0; j < p_tiles; ++j) {
          for (size_t x = 0; x < TILE; ++x) {
              for (size_t y = 0; y < TILE; ++y) {
                  out->ptr[(i * p_tiles + j) * TILE * TILE + x * TILE + y] = 0;  // TODO: check data type
              }
          }
          for (size_t k = 0; k < n_tiles; ++k) {
              const int8_t* a_tile = &a.ptr[(i * n_tiles + k) * TILE * TILE];
              const int8_t* b_tile = &b.ptr[(k * p_tiles + j) * TILE * TILE];
              int8_t* out_tile = &out->ptr[(i * p_tiles + j) * TILE * TILE];

              AlignedDotQuant(a_tile, b_tile, out_tile);
          }
			}
	}
}


void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < a.size / reduce_size ; i++) {
    float max = 0;

    for (size_t j = i * reduce_size; j < ((i+1) * reduce_size); j++) {
      if (a.ptr[j] >= max) {
        max = a.ptr[j];
      }
    }
    out->ptr[i] = max;
  }
  /// END SOLUTION
}


void ReduceMaxQuant(const AlignedArray_quant& a, AlignedArray_quant* out, size_t reduce_size) {
  for (size_t i = 0; i < a.size / reduce_size ; i++) {
    int8_t max = 0;

    for (size_t j = i * reduce_size; j < ((i+1) * reduce_size); j++) {
      if (a.ptr[j] >= max) {
        max = a.ptr[j];
      }
    }
    out->ptr[i] = max;
  }
}


void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < a.size / reduce_size ; i++) {
    float sum = 0;

    for (size_t j = i * reduce_size; j < ((i+1) * reduce_size); j++) {
      sum += a.ptr[j];
    }
    out->ptr[i] = sum;
  }
  /// END SOLUTION
}


void ReduceSumQuant(const AlignedArray_quant& a, AlignedArray_quant* out, size_t reduce_size) {
  for (size_t i = 0; i < a.size / reduce_size ; i++) {
    int8_t sum = 0;

    for (size_t j = i * reduce_size; j < ((i+1) * reduce_size); j++) {
      sum += a.ptr[j];
    }
    out->ptr[i] = sum;
  }
}


}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  py::class_<AlignedArray_quant>(m, "ArrayQuant")
    .def(py::init<size_t>(), py::return_value_policy::take_ownership)
    .def("ptr", &AlignedArray_quant::ptr_as_int)
    .def_readonly("size", &AlignedArray_quant::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });
  // TODO: check if int8 to_numpy is necessary

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });
  // TODO: check if int8 from_numpy is necessary

  m.def("fill", Fill);
  m.def("fill_quant", FillQuant);
  m.def("compact", Compact);
  m.def("compact_quant", CompactQuant);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("ewise_setitem_quant", EwiseSetitemQuant);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("scalar_setitem_quant", ScalarSetitemQuant);
  m.def("ewise_add", EwiseAdd);
  m.def("ewise_add_quant", EwiseAddQuant);
  m.def("scalar_add", ScalarAdd);
  m.def("scalar_add_quant", ScalarAddQuant);

  m.def("ewise_mul", EwiseMul);
  m.def("ewise_mul_quant", EwiseMulQuant);
  m.def("scalar_mul", ScalarMul);
  m.def("scalar_mul_quant", ScalarMulQuant);
  m.def("ewise_div", EwiseDiv);
  m.def("ewise_div_quant", EwiseDivQuant);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_div_quant", ScalarDivQuant);
  m.def("scalar_power", ScalarPower);
  m.def("scalar_power_quant", ScalarPowerQuant);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("ewise_maximum_quant", EwiseMaximumQuant);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("scalar_maximum_quant", ScalarMaximumQuant);
  m.def("ewise_eq", EwiseEq);
  m.def("ewise_eq_quant", EwiseEqQuant);
  m.def("scalar_eq", ScalarEq);
  m.def("scalar_eq_quant", ScalarEqQuant);
  m.def("ewise_ge", EwiseGe);
  m.def("ewise_ge_quant", EwiseGeQuant);
  m.def("scalar_ge", ScalarGe);
  m.def("scalar_ge_quant", ScalarGeQuant);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_log_quant", EwiseLogQuant);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_exp_quant", EwiseExpQuant);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_quant", MatmulQuant);
  m.def("matmul_tiled", MatmulTiled);
  m.def("matmul_tiled_quant", MatmulTiledQuant);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_max_quant", ReduceMaxQuant);
  m.def("reduce_sum", ReduceSum);
  m.def("reduce_sum_quant", ReduceSumQuant);
}
