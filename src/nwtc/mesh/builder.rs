use crate::nwtc::{mesh::Element, mesh::Mesh, mesh::Node, quaternion::Quaternion, vector::Vector3};
use std::sync::atomic::{AtomicUsize, Ordering};

fn get_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub struct MeshBuilder {
    mesh: Mesh,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            mesh: Mesh {
                id: get_id(),
                nodes: Vec::new(),
                elements: Vec::new(),
            },
        }
    }

    pub fn add_node(&mut self) -> NodeBuilder<'_> {
        self.mesh.nodes.push(Node::new(
            self.mesh.nodes.len(),
            Vector3::zero(),
            Quaternion::identity(),
        ));
        NodeBuilder {
            node: self.mesh.nodes.last_mut().unwrap(),
        }
    }

    pub fn add_element(&mut self, element: Element) {
        self.mesh.elements.push(element);
    }

    pub fn add_point_element(&mut self, node_id: usize) {
        self.mesh.elements.push(Element::Point(node_id));
    }

    pub fn add_line2_element(&mut self, node_id1: usize, node_id2: usize) {
        self.mesh
            .elements
            .push(Element::Line2((node_id1, node_id2)));
    }

    pub fn build(self) -> Mesh {
        self.mesh.clone()
    }
}

pub struct NodeBuilder<'a> {
    node: &'a mut Node,
}

impl<'a> NodeBuilder<'a> {
    pub fn set_position(self, x: f64, y: f64, z: f64) -> Self {
        self.node.x0 = Vector3::new(x, y, z);
        self
    }

    pub fn set_orientation(self, r: Quaternion) -> Self {
        self.node.r0 = r;
        self
    }

    pub fn translate(self, dx: Vector3) -> Self {
        self.node.x0 += dx;
        self
    }

    pub fn rotate(self, dr: Vector3) -> Self {
        let dq = Quaternion::from_vector(&dr);
        self.node.r0 = Quaternion::compose(&self.node.r0, &dq);
        self
    }

    pub fn rotate_about_point(self, q: Quaternion, point: Vector3) -> Self {
        self.node.r0 = Quaternion::compose(&self.node.r0, &q);
        let rotated_x0 = q.rotate_vector(&(self.node.x0 - point));
        self.node.x0 = rotated_x0 + point;
        self
    }

    pub fn build(self) -> usize {
        self.node.id
    }
}
