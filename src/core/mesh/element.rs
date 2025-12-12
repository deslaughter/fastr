use crate::core::types::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Element {
    Point(usize),
    Line2((usize, usize)),
}
