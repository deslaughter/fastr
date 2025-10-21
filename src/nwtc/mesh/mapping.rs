use itertools::Itertools;
use kiddo::{float::kdtree::KdTree, SquaredEuclidean};

use crate::nwtc::Element;
use crate::nwtc::Mesh;
use crate::nwtc::Vector3;

pub struct ElementMapping {
    pub destination_elem_id: usize,
    pub distance: f64,
    pub couple_arm: Vector3,
    pub shape_function: Vec<f64>,
}

pub struct MeshMapping {
    pub source_id: usize,
    pub destination_id: usize,
    point_to_point: Vec<ElementMapping>,
    // line2_to_point: Vec<ElementMapping>,
    // point_to_line2: Vec<ElementMapping>,
    // line2_to_line2: Vec<ElementMapping>,
}

impl MeshMapping {
    pub fn new(source: &Mesh, destination: &Mesh) -> Self {
        let mut kdtree: kiddo::float::kdtree::KdTree<f64, usize, 3, 32, u32> =
            KdTree::with_capacity(destination.nodes.len());

        // Add applicable elements to the kd-tree
        destination
            .elements
            .iter()
            .enumerate()
            .filter_map(|(elem_id, e)| match e {
                Element::Point(node_id) => Some((elem_id, node_id)),
                _ => None,
            })
            .unique_by(|(_, &node_id)| node_id)
            .for_each(|(elem_id, &node_id)| {
                kdtree.add(&destination.nodes[node_id].x0.as_array(), elem_id);
            });

        Self {
            source_id: source.id,
            destination_id: destination.id,
            point_to_point: source
                .nodes
                .iter()
                .map(|node_source| {
                    let nearest_node =
                        kdtree.nearest_one::<SquaredEuclidean>(&node_source.x0.as_array());
                    let destination_elem_id = nearest_node.item;
                    let destination_elem = &destination.elements[destination_elem_id];
                    let destination_node_id = match destination_elem {
                        Element::Point(nid) => *nid,
                        _ => panic!("Mapped element is not a point element"),
                    };
                    ElementMapping {
                        destination_elem_id,
                        distance: nearest_node.distance,
                        couple_arm: node_source.x0 - destination.nodes[destination_node_id].x0,
                        shape_function: Vec::new(),
                    }
                })
                .collect(),
            // line2_to_point: Vec::new(),
            // point_to_line2: Vec::new(),
            // line2_to_line2: Vec::new(),
        }
    }

    pub fn map_motion(&self, source: &Mesh, destination: &mut Mesh) {
        // Map point-to-point motions
        self.point_to_point
            .iter()
            .enumerate()
            .for_each(|(source_node_id, mapping)| {
                let node_s = &source.nodes[source_node_id];
                match &destination.elements[mapping.destination_elem_id] {
                    Element::Point(destination_node_id) => {
                        let node_d = &mut destination.nodes[*destination_node_id];

                        // Initial position vector from source to destination
                        let p0 = node_s.x0 - node_d.x0;

                        // Current position vector from source to destination
                        let p = node_s.ur.rotate_vector(&p0);

                        // Map translational displacement
                        node_d.ux = node_s.ux + p0 - node_s.ur.rotate_vector(&p0);

                        // Map rotational displacement
                        node_d.ur = node_s.ur;

                        // Map translational velocity
                        node_d.vx = node_s.vx + p.cross(&node_s.vr);

                        // Map rotational velocity
                        node_d.vr = node_s.vr;

                        // Map translational acceleration
                        node_d.ax =
                            node_s.ax + p.cross(&node_s.ar) + node_s.vr.cross(&p.cross(&node_s.vr));

                        // Map rotational acceleration
                        node_d.ar = node_s.ar;
                    }
                    _ => panic!("Mapped element is not a point element"),
                }
            });
    }
}
