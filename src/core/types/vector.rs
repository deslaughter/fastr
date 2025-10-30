use crate::core::Matrix3;

/// A 3D vector for 3D linear algebra operations.
///
/// This vector represents a point or direction in 3D space with x, y, and z components.
/// It provides common vector operations including addition, subtraction, scaling,
/// dot product, cross product, and various utility functions.
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
///
/// // Create a new vector
/// let v1 = Vector3::new(1.0, 2.0, 3.0);
/// let v2 = Vector3::new(4.0, 5.0, 6.0);
///
/// // Vector operations
/// let sum = v1 + v2;
/// let dot_product = v1.dot(&v2);
/// let cross_product = v1.cross(&v2);
/// let magnitude = v1.norm();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    /// The x-component of the vector
    pub x: f64,
    /// The y-component of the vector
    pub y: f64,
    /// The z-component of the vector
    pub z: f64,
}

impl Vector3 {
    /// Creates a new Vector3 with the given x, y, and z components.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-component of the vector
    /// * `y` - The y-component of the vector
    /// * `z` - The z-component of the vector
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let vector = Vector3::new(1.0, 2.0, 3.0);
    /// assert_eq!(vector.x, 1.0);
    /// assert_eq!(vector.y, 2.0);
    /// assert_eq!(vector.z, 3.0);
    /// ```
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Creates a zero vector with all components set to 0.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let zero = Vector3::zero();
    /// assert_eq!(zero.x, 0.0);
    /// assert_eq!(zero.y, 0.0);
    /// assert_eq!(zero.z, 0.0);
    /// ```
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Calculates the Euclidean norm (magnitude) of the vector.
    ///
    /// The norm is calculated as √(x² + y² + z²).
    ///
    /// # Returns
    ///
    /// The magnitude of the vector as an f64
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let vector = Vector3::new(3.0, 4.0, 0.0);
    /// assert_eq!(vector.norm(), 5.0);
    /// ```
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns a unit vector in the same direction as this vector.
    ///
    /// If the vector has zero magnitude (within floating point epsilon),
    /// returns a zero vector to avoid division by zero.
    ///
    /// # Returns
    ///
    /// A new Vector3 with magnitude 1.0 in the same direction, or zero vector if input is zero
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let vector = Vector3::new(3.0, 4.0, 0.0);
    /// let unit = vector.unit();
    /// assert!((unit.norm() - 1.0).abs() < f64::EPSILON);
    /// ```
    pub fn unit(&self) -> Self {
        let norm = self.norm();
        if norm < f64::EPSILON {
            Self::zero()
        } else {
            Self {
                x: self.x / norm,
                y: self.y / norm,
                z: self.z / norm,
            }
        }
    }

    /// Converts the vector to a 3-element array.
    ///
    /// # Returns
    ///
    /// An array [x, y, z] containing the vector components
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let vector = Vector3::new(1.0, 2.0, 3.0);
    /// let array = vector.to_array();
    /// assert_eq!(array, [1.0, 2.0, 3.0]);
    /// ```
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Creates a Vector3 from a 3-element array.
    ///
    /// # Arguments
    ///
    /// * `arr` - An array [x, y, z] containing the vector components
    ///
    /// # Returns
    ///
    /// A new Vector3 with components from the array
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let array = [1.0, 2.0, 3.0];
    /// let vector = Vector3::from_array(array);
    /// assert_eq!(vector.x, 1.0);
    /// ```
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }

    /// Calculates the dot product with another vector.
    ///
    /// The dot product is calculated as x₁x₂ + y₁y₂ + z₁z₂.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector to compute the dot product with
    ///
    /// # Returns
    ///
    /// The dot product as an f64
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let v1 = Vector3::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3::new(4.0, 5.0, 6.0);
    /// let dot = v1.dot(&v2);
    /// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculates the cross product with another vector.
    ///
    /// The cross product produces a vector perpendicular to both input vectors.
    /// The magnitude equals the area of the parallelogram formed by the vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector to compute the cross product with
    ///
    /// # Returns
    ///
    /// A new Vector3 representing the cross product
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let v1 = Vector3::new(1.0, 0.0, 0.0);
    /// let v2 = Vector3::new(0.0, 1.0, 0.0);
    /// let cross = v1.cross(&v2);
    /// assert_eq!(cross, Vector3::new(0.0, 0.0, 1.0));
    /// ```
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Creates the skew-symmetric matrix representation of this vector.
    ///
    /// The skew-symmetric matrix [v]× satisfies the property that for any vector u:
    /// [v]× * u = v × u (cross product)
    ///
    /// The matrix has the form:
    /// ```text
    /// [  0   -z    y ]
    /// [  z    0   -x ]
    /// [ -y    x    0 ]
    /// ```
    ///
    /// # Returns
    ///
    /// A 3x3 skew-symmetric Matrix3
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::Vector3;
    ///
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// let skew = v.skew_symmetric();
    /// // skew * u should equal v.cross(&u) for any vector u
    /// ```
    pub fn skew_symmetric(&self) -> Matrix3 {
        Matrix3::new([
            [0., -self.z, self.y],
            [self.z, 0., -self.x],
            [-self.y, self.x, 0.],
        ])
    }
}

/// Implementation of vector addition for Vector3.
///
/// Performs component-wise addition of two vectors.
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
///
/// let v1 = Vector3::new(1.0, 2.0, 3.0);
/// let v2 = Vector3::new(4.0, 5.0, 6.0);
/// let result = v1 + v2;
/// assert_eq!(result, Vector3::new(5.0, 7.0, 9.0));
/// ```
impl std::ops::Add for Vector3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

/// Implementation of vector subtraction for Vector3.
///
/// Performs component-wise subtraction of two vectors.
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
///
/// let v1 = Vector3::new(5.0, 7.0, 9.0);
/// let v2 = Vector3::new(1.0, 2.0, 3.0);
/// let result = v1 - v2;
/// assert_eq!(result, Vector3::new(4.0, 5.0, 6.0));
/// ```
impl std::ops::Sub for Vector3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

/// Implementation of in-place vector addition for Vector3.
///
/// Adds another vector to this vector, modifying this vector in place.
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
///
/// let mut v1 = Vector3::new(1.0, 2.0, 3.0);
/// let v2 = Vector3::new(4.0, 5.0, 6.0);
/// v1 += v2;
/// assert_eq!(v1, Vector3::new(5.0, 7.0, 9.0));
/// ```
impl std::ops::AddAssign for Vector3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

/// Implementation of in-place vector subtraction for Vector3.
///
/// Subtracts another vector from this vector, modifying this vector in place.
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
///
/// let mut v1 = Vector3::new(5.0, 7.0, 9.0);
/// let v2 = Vector3::new(1.0, 2.0, 3.0);
/// v1 -= v2;
/// assert_eq!(v1, Vector3::new(4.0, 5.0, 6.0));
/// ```
impl std::ops::SubAssign for Vector3 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

/// Implementation of scalar multiplication for Vector3.
///
/// Multiplies all components of the vector by a scalar value.
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
///
/// let vector = Vector3::new(1.0, 2.0, 3.0);
/// let scaled = vector * 2.0;
/// assert_eq!(scaled, Vector3::new(2.0, 4.0, 6.0));
/// ```
impl std::ops::Mul<f64> for Vector3 {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

/// Implementation of scalar division for Vector3.
///
/// Divides all components of the vector by a scalar value.
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
///
/// let vector = Vector3::new(2.0, 4.0, 6.0);
/// let scaled = vector / 2.0;
/// assert_eq!(scaled, Vector3::new(1.0, 2.0, 3.0));
/// ```
impl std::ops::Div<f64> for Vector3 {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

/// Implementation of negation for Vector3.
///
/// Returns a new vector with all components negated.
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
///
/// let vector = Vector3::new(1.0, -2.0, 3.0);
/// let negated = -vector;
/// assert_eq!(negated, Vector3::new(-1.0, 2.0, -3.0));
/// ```
impl std::ops::Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Implementation of vector-matrix multiplication for Vector3.
///
/// Multiplies this vector (treated as a row vector) by a matrix.
/// This is equivalent to multiplying the transpose of the matrix by the vector
/// when the vector is treated as a column vector.
///
/// The result is: [x y z] * Matrix = [x*m₀₀+y*m₁₀+z*m₂₀, x*m₀₁+y*m₁₁+z*m₂₁, x*m₀₂+y*m₁₂+z*m₂₂]
///
/// # Examples
///
/// ```
/// use fastr::core::Vector3;
/// use fastr::core::Matrix3;
///
/// let vector = Vector3::new(1.0, 2.0, 3.0);
/// let matrix = Matrix3::identity();
/// let result = vector * matrix;
/// assert_eq!(result, vector); // Identity matrix preserves the vector
/// ```
impl std::ops::Mul<Matrix3> for Vector3 {
    type Output = Self;

    fn mul(self, mat: Matrix3) -> Self {
        Self {
            x: self.x * mat.data[0][0] + self.y * mat.data[1][0] + self.z * mat.data[2][0],
            y: self.x * mat.data[0][1] + self.y * mat.data[1][1] + self.z * mat.data[2][1],
            z: self.x * mat.data[0][2] + self.y * mat.data[1][2] + self.z * mat.data[2][2],
        }
    }
}
