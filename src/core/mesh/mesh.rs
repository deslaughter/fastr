use crate::core::mesh::builder;

use super::{Element, Mapping, Node};
use bitflags::bitflags;
use itertools::Itertools;
use vtkio::model::*;

#[derive(Debug, Clone)]
pub struct Mesh {
    pub id: usize,
    pub fields: Fields,
    pub nodes: Vec<Node>,
    pub elements: Vec<Element>,
    pub siblings: Vec<usize>,
}

bitflags! {
    /// Represents which fields are active in a mesh.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Fields: u32 {
        const Force = 1 << 0;
        const Moment = 1 << 1;
        const TranslationalDisplacement = 1 << 2;
        const AngularDisplacement = 1 << 3;
        const TranslationalVelocity = 1 << 4;
        const AngularVelocity = 1 << 5;
        const TranslationalAcceleration = 1 << 6;
        const AngularAcceleration = 1 << 7;
        const Scalar = 1 << 8;

        const Loads = Self::Force.bits() | Self::Moment.bits();

        const Motion = Self::TranslationalDisplacement.bits() | Self::AngularDisplacement.bits() |
                       Self::TranslationalVelocity.bits() | Self::AngularVelocity.bits() |
                       Self::TranslationalAcceleration.bits() | Self::AngularAcceleration.bits();
    }
}

impl Mesh {
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    pub fn copy_motion_from(&mut self, other: &Mesh) -> &mut Self {
        self.nodes
            .iter_mut()
            .zip(other.nodes.iter())
            .for_each(|(node, other_node)| {
                node.copy_motion_from(other_node);
            });
        self
    }

    pub fn copy_loads_from(&mut self, other: &Mesh) -> &mut Self {
        self.nodes
            .iter_mut()
            .zip(other.nodes.iter())
            .for_each(|(node, other_node)| {
                node.copy_loads_from(other_node);
            });
        self
    }

    pub fn reset_loads(&mut self) -> &mut Self {
        self.nodes.iter_mut().for_each(|node| {
            node.reset_loads();
        });
        self
    }

    pub fn create_motion_mapping(&self, destination: &Mesh) -> Mapping {
        Mapping::new_motion(self, destination)
    }

    pub fn create_load_mapping(&self, destination: &Mesh) -> Mapping {
        Mapping::new_load(self, destination)
    }

    pub fn create_sibling(&mut self) -> Mesh {
        let mut sibling = self.clone();
        sibling.id = builder::get_id();
        self.siblings.push(sibling.id);
        sibling.siblings.push(self.id);
        sibling
    }

    pub fn n_values(&self, fields: Fields) -> usize {
        let mut n_per_node = 0;
        if fields.contains(Fields::TranslationalDisplacement) {
            n_per_node += 3;
        }
        if fields.contains(Fields::AngularDisplacement) {
            n_per_node += 4;
        }
        if fields.contains(Fields::TranslationalVelocity) {
            n_per_node += 3;
        }
        if fields.contains(Fields::AngularVelocity) {
            n_per_node += 3;
        }
        if fields.contains(Fields::TranslationalAcceleration) {
            n_per_node += 3;
        }
        if fields.contains(Fields::AngularAcceleration) {
            n_per_node += 3;
        }
        if fields.contains(Fields::Force) {
            n_per_node += 3;
        }
        if fields.contains(Fields::Moment) {
            n_per_node += 3;
        }
        if fields.contains(Fields::Scalar) {
            n_per_node += 1;
        }
        self.n_nodes() * n_per_node
    }

    pub fn pack(&self, fields: Option<Fields>) -> Vec<f64> {
        // Create vector to hold packed data
        let mut data = Vec::new();

        // Either use specified fields or all fields in the mesh
        let fields = fields.unwrap_or(self.fields);

        // If no fields are specified, return empty data
        if fields.is_empty() {
            return data;
        }

        macro_rules! collect_array {
            ($field_name:ident, $node_field:ident) => {
                if fields.contains(Fields::$field_name) {
                    self.nodes.iter().for_each(|node| {
                        data.extend_from_slice(&node.$node_field.to_array());
                    });
                }
            };
        }

        collect_array!(TranslationalDisplacement, ut);
        collect_array!(AngularDisplacement, ur);
        collect_array!(TranslationalVelocity, vt);
        collect_array!(AngularVelocity, vr);
        collect_array!(TranslationalAcceleration, at);
        collect_array!(AngularAcceleration, ar);
        collect_array!(Force, f);
        collect_array!(Moment, m);

        data
    }

    pub fn unpack(&mut self, field: Fields, data: &[f64]) {
        if field.is_empty() {
            return;
        }

        let mut offset = 0;

        macro_rules! assign_vec3 {
            ($field_name:ident, $node_field:ident) => {
                if field.contains(Fields::$field_name) {
                    for node in &mut self.nodes {
                        node.$node_field.copy_from_slice(&data[offset..offset + 3]);
                        offset += 3;
                    }
                }
            };
        }

        macro_rules! assign_quaternion {
            ($field_name:ident, $node_field:ident) => {
                if field.contains(Fields::$field_name) {
                    for node in &mut self.nodes {
                        node.$node_field.copy_from_slice(&data[offset..offset + 4]);
                        offset += 4;
                    }
                }
            };
        }

        assign_vec3!(TranslationalDisplacement, ut);
        assign_quaternion!(AngularDisplacement, ur);
        assign_vec3!(TranslationalVelocity, vt);
        assign_vec3!(AngularVelocity, vr);
        assign_vec3!(TranslationalAcceleration, at);
        assign_vec3!(AngularAcceleration, ar);
        assign_vec3!(Force, f);
        assign_vec3!(Moment, m);
    }

    pub fn to_vtk(&self) -> Vtk {
        // Get the number of nodes in the mesh
        let n_nodes = self.nodes.len();

        // Convert node orientations to rotation matrices
        let rotations = self.nodes.iter().map(|n| n.r().to_matrix()).collect_vec();

        // Macro to define orientation attributes
        macro_rules! build_orientation_attribute {
            ($label:literal, $column:literal) => {
                Attribute::DataArray(DataArrayBase {
                    name: $label.to_string(),
                    elem: ElementType::Vectors,
                    data: IOBuffer::F32(
                        rotations
                            .iter()
                            .flat_map(|r| r.column($column).map(|v| v as f32))
                            .collect_vec(),
                    ),
                })
            };
        }

        // Macro to define data attributes
        macro_rules! build_data_attribute {
            ($label:literal, $field:ident) => {
                Attribute::DataArray(DataArrayBase {
                    name: $label.to_string(),
                    elem: ElementType::Vectors,
                    data: IOBuffer::F32(
                        self.nodes
                            .iter()
                            .flat_map(|n| [n.$field.x as f32, n.$field.y as f32, n.$field.z as f32])
                            .collect_vec(),
                    ),
                })
            };
        }

        Vtk {
            version: Version { major: 4, minor: 2 },
            title: String::new(),
            byte_order: ByteOrder::LittleEndian,
            file_path: None,
            data: DataSet::inline(UnstructuredGridPiece {
                points: IOBuffer::F64(
                    self.nodes
                        .iter()
                        .flat_map(|n| n.x().to_array())
                        .collect_vec(),
                ),
                cells: Cells {
                    cell_verts: VertexNumbers::XML {
                        connectivity: (0..n_nodes as u64).collect_vec(),
                        offsets: (1..n_nodes as u64 + 1).collect_vec(),
                    },
                    types: vec![CellType::Vertex; n_nodes],
                },
                data: Attributes {
                    point: vec![
                        build_orientation_attribute!("OrientationX", 0),
                        build_orientation_attribute!("OrientationY", 1),
                        build_orientation_attribute!("OrientationZ", 2),
                        build_data_attribute!("TranslationalVelocity", vt),
                        build_data_attribute!("AngularVelocity", vr),
                        build_data_attribute!("TranslationalAcceleration", at),
                        build_data_attribute!("AngularAcceleration", ar),
                        build_data_attribute!("Force", f),
                        build_data_attribute!("Moment", m),
                    ],
                    ..Default::default()
                },
            }),
        }
    }
}
