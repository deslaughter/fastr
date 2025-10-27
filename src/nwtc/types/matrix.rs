use crate::nwtc::vector::Vector3;

/// A 3x3 matrix for 3D linear algebra operations.
///
/// This matrix is stored in row-major order and provides common operations
/// for 3D transformations, rotations, and linear algebra calculations.
///
/// # Examples
///
/// ```
/// use fastr::nwtc::Matrix3;
///
/// // Create an identity matrix
/// let identity = Matrix3::identity();
///
/// // Create a custom matrix
/// let matrix = Matrix3::new([
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0]
/// ]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3 {
    /// The matrix data stored in row-major order as a 3x3 array
    pub data: [[f64; 3]; 3],
}

impl Matrix3 {
    /// Creates a new Matrix3 with the given data array.
    ///
    /// # Arguments
    ///
    /// * `data` - A 3x3 array of f64 values in row-major order
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Matrix3;
    ///
    /// let matrix = Matrix3::new([
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0]
    /// ]);
    /// ```
    pub fn new(data: [[f64; 3]; 3]) -> Self {
        Self { data }
    }

    /// Creates a 3x3 identity matrix.
    ///
    /// The identity matrix has 1.0 on the diagonal and 0.0 elsewhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Matrix3;
    ///
    /// let identity = Matrix3::identity();
    /// assert_eq!(identity.get(0, 0), 1.0);
    /// assert_eq!(identity.get(1, 1), 1.0);
    /// assert_eq!(identity.get(2, 2), 1.0);
    /// ```
    pub fn identity() -> Self {
        Self {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Gets the value at the specified row and column.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index (0-2)
    /// * `col` - The column index (0-2)
    ///
    /// # Returns
    ///
    /// The f64 value at the specified position
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Matrix3;
    ///
    /// let matrix = Matrix3::identity();
    /// assert_eq!(matrix.get(0, 0), 1.0);
    /// assert_eq!(matrix.get(0, 1), 0.0);
    /// ```
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    /// Gets the specified column as an array.
    ///
    /// # Arguments
    ///
    /// * `col` - The column index (0-2)
    ///
    /// # Returns
    ///
    /// An array of 3 f64 values representing the column
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Matrix3;
    ///
    /// let matrix = Matrix3::identity();
    /// let first_column = matrix.column(0);
    /// assert_eq!(first_column, [1.0, 0.0, 0.0]);
    /// ```
    pub fn column(&self, col: usize) -> [f64; 3] {
        [self.data[0][col], self.data[1][col], self.data[2][col]]
    }

    /// Gets the specified row as an array.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index (0-2)
    ///
    /// # Returns
    ///
    /// An array of 3 f64 values representing the row
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Matrix3;
    ///
    /// let matrix = Matrix3::identity();
    /// let first_row = matrix.row(0);
    /// assert_eq!(first_row, [1.0, 0.0, 0.0]);
    /// ```
    pub fn row(&self, row: usize) -> [f64; 3] {
        self.data[row]
    }

    /// Creates a matrix from an array of row vectors.
    ///
    /// # Arguments
    ///
    /// * `r` - A slice of 3 Vector3 objects representing the rows
    ///
    /// # Returns
    ///
    /// A new Matrix3 where each row corresponds to the given vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::{Matrix3, Vector3};
    ///
    /// let rows = [
    ///     Vector3::new(1.0, 2.0, 3.0),
    ///     Vector3::new(4.0, 5.0, 6.0),
    ///     Vector3::new(7.0, 8.0, 9.0)
    /// ];
    /// let matrix = Matrix3::from_rows(&rows);
    /// ```
    pub fn from_rows(r: &[Vector3]) -> Self {
        assert_eq!(r.len(), 3, "Expected exactly 3 row vectors");
        Self {
            data: [
                [r[0].x, r[0].y, r[0].z],
                [r[1].x, r[1].y, r[1].z],
                [r[2].x, r[2].y, r[2].z],
            ],
        }
    }

    /// Creates a matrix from an array of column vectors.
    ///
    /// # Arguments
    ///
    /// * `c` - A slice of 3 Vector3 objects representing the columns
    ///
    /// # Returns
    ///
    /// A new Matrix3 where each column corresponds to the given vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::{Matrix3, Vector3};
    ///
    /// let columns = [
    ///     Vector3::new(1.0, 4.0, 7.0),
    ///     Vector3::new(2.0, 5.0, 8.0),
    ///     Vector3::new(3.0, 6.0, 9.0)
    /// ];
    /// let matrix = Matrix3::from_columns(&columns);
    /// ```
    pub fn from_columns(c: &[Vector3]) -> Self {
        assert_eq!(c.len(), 3, "Expected exactly 3 column vectors");
        Self {
            data: [
                [c[0].x, c[1].x, c[2].x],
                [c[0].y, c[1].y, c[2].y],
                [c[0].z, c[1].z, c[2].z],
            ],
        }
    }

    /// Returns the transpose of this matrix.
    ///
    /// The transpose swaps rows and columns, so element (i,j) becomes (j,i).
    ///
    /// # Returns
    ///
    /// A new Matrix3 that is the transpose of this matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Matrix3;
    ///
    /// let matrix = Matrix3::new([
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 9.0]
    /// ]);
    /// let transposed = matrix.transpose();
    /// assert_eq!(transposed.get(0, 1), 4.0); // was (1,0)
    /// ```
    pub fn transpose(&self) -> Self {
        Self {
            data: [
                [self.data[0][0], self.data[1][0], self.data[2][0]],
                [self.data[0][1], self.data[1][1], self.data[2][1]],
                [self.data[0][2], self.data[1][2], self.data[2][2]],
            ],
        }
    }

    pub fn into_faer(self) -> faer::Mat<f64> {
        faer::mat![
            [self.data[0][0], self.data[0][1], self.data[0][2]],
            [self.data[1][0], self.data[1][1], self.data[1][2]],
            [self.data[2][0], self.data[2][1], self.data[2][2]],
        ]
    }
}

/// Implementation of matrix subtraction for Matrix3.
///
/// Performs element-wise subtraction of two matrices.
///
/// # Examples
///
/// ```
/// use fastr::nwtc::Matrix3;
///
/// let a = Matrix3::identity();
/// let b = Matrix3::identity();
/// let result = a - b; // Should be zero matrix
/// ```
impl std::ops::Sub for Matrix3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            data: [
                [
                    self.data[0][0] - other.data[0][0],
                    self.data[0][1] - other.data[0][1],
                    self.data[0][2] - other.data[0][2],
                ],
                [
                    self.data[1][0] - other.data[1][0],
                    self.data[1][1] - other.data[1][1],
                    self.data[1][2] - other.data[1][2],
                ],
                [
                    self.data[2][0] - other.data[2][0],
                    self.data[2][1] - other.data[2][1],
                    self.data[2][2] - other.data[2][2],
                ],
            ],
        }
    }
}

/// Implementation of indexing for Matrix3 using tuple notation.
///
/// Allows accessing matrix elements using `matrix[(row, col)]` syntax.
///
/// # Examples
///
/// ```
/// use fastr::nwtc::Matrix3;
///
/// let matrix = Matrix3::identity();
/// let value = matrix[(0, 0)]; // Gets the element at row 0, column 0
/// assert_eq!(value, 1.0);
/// ```
impl std::ops::Index<(usize, usize)> for Matrix3 {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

/// Implementation of negation for Matrix3.
///
/// Returns a new matrix with all elements negated.
///
/// # Examples
///
/// ```
/// use fastr::nwtc::Matrix3;
///
/// let matrix = Matrix3::identity();
/// let negated = -matrix;
/// assert_eq!(negated.get(0, 0), -1.0);
/// ```
impl std::ops::Neg for Matrix3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            data: [
                [-self.data[0][0], -self.data[0][1], -self.data[0][2]],
                [-self.data[1][0], -self.data[1][1], -self.data[1][2]],
                [-self.data[2][0], -self.data[2][1], -self.data[2][2]],
            ],
        }
    }
}

/// Implementation of matrix multiplication for Matrix3.
///
/// Performs standard matrix multiplication where the result[i,j] is the dot product
/// of the i-th row of the left matrix and the j-th column of the right matrix.
///
/// # Examples
///
/// ```
/// use fastr::nwtc::Matrix3;
///
/// let a = Matrix3::identity();
/// let b = Matrix3::identity();
/// let result = a * b; // Should equal identity matrix
/// ```
impl std::ops::Mul<Matrix3> for Matrix3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = Matrix3::new([[0.0; 3]; 3]);
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] = self.data[i][0] * other.data[0][j]
                    + self.data[i][1] * other.data[1][j]
                    + self.data[i][2] * other.data[2][j];
            }
        }
        result
    }
}

/// Implementation of matrix-vector multiplication for Matrix3.
///
/// Multiplies the matrix by a Vector3, treating the vector as a column vector.
/// The result is a new Vector3 where each component is the dot product of the
/// corresponding matrix row with the input vector.
///
/// # Examples
///
/// ```
/// use fastr::nwtc::{Matrix3, Vector3};
///
/// let matrix = Matrix3::identity();
/// let vector = Vector3::new(1.0, 2.0, 3.0);
/// let result = matrix * vector; // Should equal the original vector
/// ```
impl std::ops::Mul<Vector3> for Matrix3 {
    type Output = Vector3;

    fn mul(self, vec: Vector3) -> Vector3 {
        Vector3 {
            x: self.data[0][0] * vec.x + self.data[0][1] * vec.y + self.data[0][2] * vec.z,
            y: self.data[1][0] * vec.x + self.data[1][1] * vec.y + self.data[1][2] * vec.z,
            z: self.data[2][0] * vec.x + self.data[2][1] * vec.y + self.data[2][2] * vec.z,
        }
    }
}

impl std::fmt::Display for Matrix3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[[{}, {}, {}]\n",
            self.data[0][0], self.data[0][1], self.data[0][2]
        )?;
        write!(
            f,
            " [{}, {}, {}]\n",
            self.data[1][0], self.data[1][1], self.data[1][2]
        )?;
        write!(
            f,
            " [{}, {}, {}]]",
            self.data[2][0], self.data[2][1], self.data[2][2]
        )?;
        Ok(())
    }
}
