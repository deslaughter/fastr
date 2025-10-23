pub mod builder;
pub mod mapping;
pub mod node;

pub use builder::*;
pub use mapping::*;
pub use node::*;

use bitflags::bitflags;
use itertools::Itertools;
use vtkio::model::*;

#[derive(Debug, Clone)]
pub struct Mesh {
    pub id: usize,
    pub nodes: Vec<Node>,
    pub elements: Vec<Element>,
}

bitflags! {
    /// Represents which fields are active in a mesh.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Fields: u32 {
        const Force = 1 << 0;
        const Moment = 1 << 1;
        const TranslationalDisplacement = 1 << 2;
        const AngularDisplacement = 1 << 3;
        const TranslationalVelocity = 1 << 4;
        const AngularVelocity = 1 << 5;
        const TranslationalAcceleration = 1 << 6;
        const AngularAcceleration = 1 << 7;

        const Load = Self::Force.bits() | Self::Moment.bits();

        const Motion = Self::TranslationalDisplacement.bits() | Self::AngularDisplacement.bits() |
                       Self::TranslationalVelocity.bits() | Self::AngularVelocity.bits() |
                       Self::TranslationalAcceleration.bits() | Self::AngularAcceleration.bits();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Element {
    Point(usize),
    Line2((usize, usize)),
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

    pub fn create_mapping(&self, destination: &Mesh) -> Mapping {
        Mapping::new(self, destination)
    }

    pub fn to_vtk(&self) -> Vtk {
        let rotations = self.nodes.iter().map(|n| n.r().to_matrix()).collect_vec();
        let n_nodes = self.nodes.len();

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
                        Attribute::DataArray(DataArrayBase {
                            name: "OrientationX".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                rotations
                                    .iter()
                                    .flat_map(|r| r.column(0).map(|v| v as f32))
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "OrientationY".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                rotations
                                    .iter()
                                    .flat_map(|r| r.column(1).map(|v| v as f32))
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "OrientationZ".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                rotations
                                    .iter()
                                    .flat_map(|r| r.column(2).map(|v| v as f32))
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "TranslationalVelocity".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.nodes
                                    .iter()
                                    .flat_map(|n| [n.vt.x as f32, n.vt.y as f32, n.vt.z as f32])
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "AngularVelocity".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.nodes
                                    .iter()
                                    .flat_map(|n| [n.vr.x as f32, n.vr.y as f32, n.vr.z as f32])
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "TranslationalAcceleration".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.nodes
                                    .iter()
                                    .flat_map(|n| [n.ax.x as f32, n.ax.y as f32, n.ax.z as f32])
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "AngularAcceleration".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.nodes
                                    .iter()
                                    .flat_map(|n| [n.ar.x as f32, n.ar.y as f32, n.ar.z as f32])
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "Force".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.nodes
                                    .iter()
                                    .flat_map(|n| [n.f.x as f32, n.f.y as f32, n.f.z as f32])
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "Moment".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.nodes
                                    .iter()
                                    .flat_map(|n| [n.m.x as f32, n.m.y as f32, n.m.z as f32])
                                    .collect_vec(),
                            ),
                        }),
                    ],
                    ..Default::default()
                },
            }),
        }
    }
}
