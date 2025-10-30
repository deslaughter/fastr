use itertools::Itertools;
use kiddo::{float::kdtree::KdTree, SquaredEuclidean};

use crate::core::mesh::{Element, Mesh};
use crate::core::Vector3;

/// Represents a mapping between elements in different meshes.
///
/// This structure stores the relationship between a source element and a destination
/// element, including spatial information and interpolation data needed for
/// transferring motion and loads between meshes.
///
/// # Fields
///
/// * `destination_elem_id` - The ID of the destination element in the target mesh
/// * `distance` - The squared Euclidean distance between source and destination elements
/// * `couple_arm` - The vector from destination to source element position
/// * `shape_function` - Interpolation weights for multi-node elements (currently unused for point elements)
pub struct ElementMapping {
    /// The ID of the element in the target mesh
    pub elem_id: usize,
    /// The squared Euclidean distance between source and destination elements
    pub distance: f64,
    /// The vector from destination to source element position
    pub couple_arm: Vector3,
}

/// A comprehensive mapping between two meshes for motion and load transfer.
///
/// This structure establishes relationships between elements of different types
/// in source and destination meshes. It supports various mapping strategies
/// including point-to-point, line-to-point, point-to-line, and line-to-line
/// mappings for transferring motion, velocities, accelerations, and loads.
///
/// The mapping is built using spatial proximity algorithms (KD-tree) to find
/// the nearest elements between meshes and establish coupling relationships.
///
pub struct Mapping {
    /// The ID of the source mesh
    pub source_id: usize,
    /// The ID of the destination mesh
    pub destination_id: usize,
    /// Mapping from source nodes to destination point elements
    point_to_point: Vec<ElementMapping>,
}

impl Mapping {
    /// Creates a new mesh mapping between source and destination meshes.
    ///
    /// This constructor builds spatial relationships between the meshes using
    /// a KD-tree for efficient nearest neighbor searches. Currently supports
    /// point-to-point mapping where each source node is mapped to the nearest
    /// destination point element.
    ///
    /// # Arguments
    ///
    /// * `source` - The source mesh containing elements that will drive the motion
    /// * `destination` - The destination mesh that will receive the transferred motion
    ///
    /// # Returns
    ///
    /// A new `MeshMapping` instance with established element relationships
    ///
    /// # Algorithm
    ///
    /// 1. Builds a KD-tree from all point elements in the destination mesh
    /// 2. For each source node, finds the nearest destination point element
    /// 3. Stores the mapping relationship with distance and coupling arm information
    ///
    pub fn new_motion(source: &Mesh, destination: &Mesh) -> Self {
        // Build KD-tree for efficient spatial searching of destination elements
        let mut kdtree: kiddo::float::kdtree::KdTree<f64, usize, 3, 32, u32> =
            KdTree::with_capacity(destination.nodes.len());

        // Add applicable elements to the kd-tree
        // Filter for point elements only and ensure uniqueness by node ID
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
                kdtree.add(&destination.nodes[node_id].x0.to_array(), elem_id);
            });

        Self {
            source_id: source.id,
            destination_id: destination.id,
            // Create point-to-point mappings for all source nodes
            point_to_point: source
                .nodes
                .iter()
                .map(|node_source| {
                    // Find the nearest destination point element
                    let nearest_node =
                        kdtree.nearest_one::<SquaredEuclidean>(&node_source.x0.to_array());
                    let destination_elem_id = nearest_node.item;
                    let destination_elem = &destination.elements[destination_elem_id];
                    let destination_node_id = match destination_elem {
                        Element::Point(nid) => *nid,
                        _ => panic!("Mapped element is not a point element"),
                    };
                    ElementMapping {
                        elem_id: destination_elem_id,
                        distance: nearest_node.distance,
                        // Calculate coupling arm from destination to source
                        couple_arm: destination.nodes[destination_node_id].x0 - node_source.x0,
                    }
                })
                .collect(),
        }
    }

    /// Creates a new mesh load mapping between source and destination meshes.
    ///
    /// This constructor builds spatial relationships between the meshes using
    /// a KD-tree for efficient nearest neighbor searches. Currently supports
    /// point-to-point mapping where each source node is mapped to the nearest
    /// destination point element.
    ///
    /// # Arguments
    ///
    /// * `source` - The source mesh containing elements that will drive the motion
    /// * `destination` - The destination mesh that will receive the transferred motion
    ///
    /// # Returns
    ///
    /// A new `MeshMapping` instance with established element relationships
    ///
    /// # Algorithm
    ///
    /// 1. Builds a KD-tree from all point elements in the source mesh
    /// 2. For each destination node, finds the nearest source point element
    /// 3. Stores the mapping relationship with distance and coupling arm information
    ///
    pub fn new_load(source: &Mesh, destination: &Mesh) -> Self {
        // Build KD-tree for efficient spatial searching of source elements
        let mut kdtree: kiddo::float::kdtree::KdTree<f64, usize, 3, 32, u32> =
            KdTree::with_capacity(source.nodes.len());

        // Add unique source elements to the kd-tree
        // Filter for point elements only and ensure uniqueness by node ID
        source
            .elements
            .iter()
            .enumerate()
            .filter_map(|(elem_id, e)| match e {
                Element::Point(node_id) => Some((elem_id, node_id)),
                _ => None,
            })
            .unique_by(|(_, &node_id)| node_id)
            .for_each(|(elem_id, &node_id)| {
                kdtree.add(&source.nodes[node_id].x0.to_array(), elem_id);
            });

        Self {
            source_id: source.id,
            destination_id: destination.id,
            // Create point-to-point mappings for all source nodes
            point_to_point: destination
                .nodes
                .iter()
                .map(|node_destination| {
                    // Find the nearest source point element
                    let nearest_node =
                        kdtree.nearest_one::<SquaredEuclidean>(&node_destination.x0.to_array());
                    let source_elem_id = nearest_node.item;
                    let source_elem = &source.elements[source_elem_id];
                    let source_node_id = match source_elem {
                        Element::Point(nid) => *nid,
                        _ => panic!("Mapped element is not a point element"),
                    };
                    ElementMapping {
                        elem_id: source_elem_id,
                        distance: nearest_node.distance,
                        // Calculate coupling arm from source to source
                        couple_arm: source.nodes[source_node_id].x0 - node_destination.x0,
                    }
                })
                .collect(),
        }
    }

    /// Maps motion, velocity, and acceleration from source mesh to destination mesh.
    ///
    /// This function transfers kinematic quantities (displacements, rotations, velocities,
    /// and accelerations) from the source mesh to the destination mesh using the
    /// established mapping relationships. It accounts for rigid body motion including
    /// the coupling between translational and rotational motion at offset points.
    ///
    /// # Arguments
    ///
    /// * `source` - The source mesh containing the motion to be transferred
    /// * `destination` - The destination mesh that will receive the transferred motion (mutable)
    ///
    /// # Physics
    ///
    /// The motion transfer accounts for rigid body kinematics:
    ///
    /// **Displacement:**
    /// - `u_d = u_s + p0 - R_s * p0` where p0 is the initial coupling arm
    ///
    /// **Velocity:**
    /// - `v_d = v_s + p × ω_s` where p is the current coupling arm and ω_s is angular velocity
    ///
    /// **Acceleration:**
    /// - `a_d = a_s + p × α_s + ω_s × (p × ω_s)` including centripetal acceleration
    ///
    pub fn transfer_motion(&self, source: &Mesh, destination: &mut Mesh) {
        // Map point-to-point motions
        self.point_to_point
            .iter()
            .enumerate()
            .for_each(|(source_node_id, mapping)| {
                // Get the source node
                let node_s = &source.nodes[source_node_id];

                // Get the destination node
                match &destination.elements[mapping.elem_id] {
                    Element::Point(destination_node_id) => {
                        let node_d = &mut destination.nodes[*destination_node_id];

                        // Initial position vector from source to destination
                        let p0 = node_d.x0 - node_s.x0;

                        // Current position vector from source to destination (rotated)
                        let p = node_s.ur.rotate_vector(&p0);

                        // Map translational displacement
                        // u_d = u_s + (p0 - R*p0) accounts for rotation of the coupling arm
                        node_d.ut = node_s.ut + node_s.ur.rotate_vector(&p0) - p0;

                        // Map rotational displacement (rigid body assumption)
                        node_d.ur = node_s.ur;

                        // Map translational velocity
                        // v_d = v_s + p × ω_s (velocity due to rotation at offset point)
                        node_d.vt = node_s.vt + node_s.vr.cross(&p);

                        // Map rotational velocity (rigid body assumption)
                        node_d.vr = node_s.vr;

                        // Map translational acceleration
                        // a_d = a_s + p × α_s + ω_s × (p × ω_s)
                        // Includes both angular acceleration and centripetal acceleration terms
                        node_d.ax =
                            node_s.ax + p.cross(&node_s.ar) + node_s.vr.cross(&node_s.vr.cross(&p));

                        // Map rotational acceleration (rigid body assumption)
                        node_d.ar = node_s.ar;
                    }
                    _ => panic!("Mapped element is not a point element"),
                }
            });
    }

    pub fn transfer_loads(&self, source: &Mesh, destination: &mut Mesh) {
        // Loop through point-to-point mappings
        self.point_to_point
            .iter()
            .enumerate()
            .for_each(|(destination_node_id, mapping)| {
                // Get mutable reference to destination node
                let node_d = &mut destination.nodes[destination_node_id];

                // Get the source element
                match &source.elements[mapping.elem_id] {
                    Element::Point(source_node_id) => {
                        // Get the source node
                        let node_s = &source.nodes[*source_node_id];

                        // Map forces directly
                        node_d.f += node_s.f;

                        // Current vector from destination to source
                        let p = node_d.x() - node_s.x();

                        // Map moments including couple from force at offset point
                        node_d.m += node_s.m + p.cross(&node_s.f);
                    }
                    _ => panic!("Mapped element is not a point element"),
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MeshBuilder, Quaternion};
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;

    fn create_test_meshes() -> (Mesh, Mesh) {
        // Create source mesh with 1 nodes
        let mut mb = MeshBuilder::new();
        let nid = mb
            .add_node()
            .set_position(0.0, 0.0, 0.0)
            .set_orientation(Quaternion::identity())
            .build();
        mb.add_point_element(nid);
        let mesh_source = mb.build();

        // Create destination mesh with 1 nodes slightly offset
        let mut mb = MeshBuilder::new();
        let nid = mb
            .add_node()
            .set_position(1.0, 0.0, 0.0)
            .set_orientation(Quaternion::identity())
            .build();
        mb.add_point_element(nid);
        let mesh_destination = mb.build();

        (mesh_source, mesh_destination)
    }

    #[test]
    fn test_mapping_creation() {
        let (source, destination) = create_test_meshes();

        // Create mapping
        let mapping = Mapping::new_motion(&source, &destination);

        assert_eq!(mapping.source_id, source.id);
        assert_eq!(mapping.destination_id, destination.id);
        assert_eq!(mapping.point_to_point.len(), source.nodes.len());

        // Check mapping distances and couple arms
        assert_relative_eq!(mapping.point_to_point[0].distance, 1., epsilon = 1e-10);
        assert_relative_eq!(mapping.point_to_point[0].couple_arm.x, 1., epsilon = 1e-10);
        assert_relative_eq!(mapping.point_to_point[0].couple_arm.y, 0., epsilon = 1e-10);
        assert_relative_eq!(mapping.point_to_point[0].couple_arm.z, 0., epsilon = 1e-10);
    }

    #[test]
    fn test_map_motion_translational() {
        let (mut source, mut destination) = create_test_meshes();

        // Apply displacement to source mesh
        source.nodes[0].translate(Vector3::new(0.5, 0.3, 0.2));

        // Create mapping and transfer motion
        let mapping = Mapping::new_motion(&source, &destination);
        mapping.transfer_motion(&source, &mut destination);

        // Check if displacement was properly transferred
        assert_relative_eq!(destination.nodes[0].ut.x, 0.5, epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].ut.y, 0.3, epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].ut.z, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_map_motion_rotational() {
        let (mut source, mut destination) = create_test_meshes();

        // Apply rotation to source mesh
        source.nodes[0].rotate(Quaternion::from_vector(Vector3::new(0.0, 0.0, FRAC_PI_2)));

        // Create mapping and transfer motion
        let mapping = Mapping::new_motion(&source, &destination);
        mapping.transfer_motion(&source, &mut destination);

        // Check if rotation was properly transferred
        assert_relative_eq!(
            destination.nodes[0].ur.w,
            source.nodes[0].ur.w,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            destination.nodes[0].ur.x,
            source.nodes[0].ur.x,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            destination.nodes[0].ur.y,
            source.nodes[0].ur.y,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            destination.nodes[0].ur.z,
            source.nodes[0].ur.z,
            epsilon = 1e-10
        );

        // Check displacement due to rotation
        assert_relative_eq!(destination.nodes[0].ut.x, -1., epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].ut.y, 1., epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].ut.z, 0., epsilon = 1e-10);
    }

    #[test]
    fn test_map_velocities() {
        let (mut source, mut destination) = create_test_meshes();

        // Apply velocities to source mesh
        source.nodes[0].vt = Vector3::new(1.0, 2.0, 3.0);
        source.nodes[0].vr = Vector3::new(0.0, 0.0, 2.0); // Angular velocity around z-axis

        // Transfer motion
        let mapping = Mapping::new_motion(&source, &destination);
        mapping.transfer_motion(&source, &mut destination);

        // Check linear velocity: v_d = v_s + p × ω_s
        // p = [1, 0, 0], ω = [0, 0, 2]
        // p × ω_s = [0, 2, 0]
        // v_d = [1, 2, 3] + [0, 2, 0] = [1, 4, 3]
        assert_relative_eq!(destination.nodes[0].vt.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].vt.y, 4.0, epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].vt.z, 3.0, epsilon = 1e-10);

        // Check angular velocity transfer
        // ω_d = ω_s
        assert_relative_eq!(destination.nodes[0].vr.z, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_map_loads() {
        let (mut source, mut destination) = create_test_meshes();

        // Apply forces and moments to source mesh
        source.nodes[0].f = Vector3::new(10.0, 20.0, 30.0);
        source.nodes[0].m = Vector3::new(5.0, 6.0, 7.0);

        // Transfer loads
        let mapping = Mapping::new_load(&source, &destination);
        mapping.transfer_loads(&source, &mut destination);

        // Check force transfer
        assert_relative_eq!(destination.nodes[0].f.x, 10.0, epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].f.y, 20.0, epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].f.z, 30.0, epsilon = 1e-10);

        // Check moment transfer including couple
        // Additional moment = p × F = [1.0, 0, 0] × [10, 20, 30] = [0, -30, 20]
        assert_relative_eq!(destination.nodes[0].m.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].m.y, -24.0, epsilon = 1e-10);
        assert_relative_eq!(destination.nodes[0].m.z, 27.0, epsilon = 1e-10);
    }
}
