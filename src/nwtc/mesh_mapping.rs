use faer::{col, Col};
use itertools::Itertools;
use kiddo::{float::kdtree::KdTree, NearestNeighbour, SquaredEuclidean};

use super::mesh::Mesh;

pub struct ElementMapping {
    pub destination_elem_id: usize,
    pub distance: f64,
    pub couple_arm: Col<f64>,
    pub shape_function: Col<f64>,
}

pub struct MeshMapping {
    /// Mapping data structure for loads on the mesh [-]
    pub loads: Vec<ElementMapping>,

    /// Mapping data structure for motions and/or scalars on the mesh [-]
    pub motion: Vec<ElementMapping>,
    // For source line2 loads, we map between source and an augmented source mesh, then between augmented source and destination [-]
    // pub augmented_source_mapping: Vec<ElementMapping>,

    // pub augmented_source: Mesh,
}

impl MeshMapping {
    pub fn new(source: Mesh, destination: Mesh) -> Self {
        let mut kdtree: kiddo::float::kdtree::KdTree<f64, usize, 3, 32, u32> =
            KdTree::with_capacity(destination.n_nodes());

        // Add applicable elements to the kdtree
        destination
            .elements
            .points
            .iter()
            .enumerate()
            .unique_by(|(_, &node_id)| node_id)
            .for_each(|(elem_id, &node_id)| {
                let c = destination.x0_trans.col(node_id);
                kdtree.add(&[c[0], c[1], c[2]], elem_id);
            });

        let mappings: Vec<ElementMapping> = source
            .x0_trans
            .col_iter()
            .map(|source_x0_trans| {
                let nearest = kdtree.nearest_one::<SquaredEuclidean>(&[
                    source_x0_trans[0],
                    source_x0_trans[1],
                    source_x0_trans[2],
                ]);
                ElementMapping {
                    destination_elem_id: nearest.item,
                    distance: nearest.distance,
                    couple_arm: destination
                        .x0_trans
                        .col(destination.elements.points[nearest.item])
                        - source_x0_trans,
                    shape_function: col![0.],
                }
            })
            .collect();

        Self {
            loads: Vec::new(),
            motion: Vec::new(),
            // augmented_source_mapping: Vec::new(),
        }
    }
}
