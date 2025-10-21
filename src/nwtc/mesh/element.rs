#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Element {
    Point(usize),
    Line2((usize, usize)),
}
