use crate::nwtc::vector::Vector3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3 {
    pub data: [[f64; 3]; 3],
}

impl Matrix3 {
    pub fn new(data: [[f64; 3]; 3]) -> Self {
        Self { data }
    }
    pub fn identity() -> Self {
        Self {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }
    pub fn column(&self, col: usize) -> [f64; 3] {
        [self.data[0][col], self.data[1][col], self.data[2][col]]
    }
    pub fn row(&self, row: usize) -> [f64; 3] {
        self.data[row]
    }

    pub fn from_rows(r: &[Vector3]) -> Self {
        Self {
            data: [
                [r[0].x, r[0].y, r[0].z],
                [r[1].x, r[1].y, r[1].z],
                [r[2].x, r[2].y, r[2].z],
            ],
        }
    }

    pub fn from_columns(c: &[Vector3]) -> Self {
        Self {
            data: [
                [c[0].x, c[1].x, c[2].x],
                [c[0].y, c[1].y, c[2].y],
                [c[0].z, c[1].z, c[2].z],
            ],
        }
    }

    pub fn transpose(&self) -> Self {
        Self {
            data: [
                [self.data[0][0], self.data[1][0], self.data[2][0]],
                [self.data[0][1], self.data[1][1], self.data[2][1]],
                [self.data[0][2], self.data[1][2], self.data[2][2]],
            ],
        }
    }
}

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

impl std::ops::Index<(usize, usize)> for Matrix3 {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

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
