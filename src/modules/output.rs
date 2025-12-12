use crate::{
    core::types::*,
    core::{mesh::Fields, Mesh},
    modules::InputOutputVar,
};

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Output {
    meshes: HashMap<usize, Mesh>,
}

impl Output {
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
        }
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> usize {
        let id = mesh.id;
        self.meshes.insert(mesh.id, mesh);
        id
    }

    pub fn mesh_ids(&self) -> HashSet<usize> {
        self.meshes.keys().cloned().collect()
    }

    pub fn pack(&self, vars: &[InputOutputVar]) -> Vec<f64> {
        let mut packed = Vec::new();
        vars.iter().for_each(|var| {
            // Get the mesh corresponding to the variable
            let mesh = self
                .meshes
                .get(&var.mesh_id)
                .unwrap_or_else(|| panic!("Mesh ID {} not found in output meshes.", var.mesh_id));

            // Pack the mesh data for the specified fields
            packed.extend(mesh.pack(Some(var.mesh_fields)));
        });
        packed
    }
}

impl<'a> IntoIterator for &'a Output {
    type Item = (&'a usize, &'a Mesh);
    type IntoIter = std::collections::hash_map::Iter<'a, usize, Mesh>;

    fn into_iter(self) -> Self::IntoIter {
        self.meshes.iter()
    }
}

impl<'a> IntoIterator for &'a mut Output {
    type Item = (&'a usize, &'a mut Mesh);
    type IntoIter = std::collections::hash_map::IterMut<'a, usize, Mesh>;

    fn into_iter(self) -> Self::IntoIter {
        self.meshes.iter_mut()
    }
}

pub fn extrapolate_interpolate(time: f64, times: &[f64], io_data: &[Output], u: &mut Output) {
    assert!(
        io_data.len() >= 2,
        "At least two output instances are required for extrapolation/interpolation."
    );

    assert!(
        io_data.len() <= 3,
        "At least two output instances are required for extrapolation/interpolation."
    );

    // Construct weights based on lagrange interpolating polynomials
    let n = io_data.len();
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut w = 1.0;
        for j in 0..n {
            if i != j {
                w *= (time - times[j]) / (times[i] - times[j]);
            }
        }
        weights[i] = w;
    }

    // Get the mesh ids present in the destination output
    let mesh_ids = u.mesh_ids();

    // Ensure all output instances contain the same meshes as the destination output
    io_data.iter().for_each(|output| {
        assert!(
            output.mesh_ids().is_superset(&mesh_ids),
            "All output instances must contain the same meshes as the destination output."
        );
    });

    // Loop through mesh ids and perform extrapolation/interpolation
    mesh_ids.iter().for_each(|&mesh_id| {
        let dest_mesh = u.meshes.get_mut(&mesh_id).unwrap();

        // Get packed version of mesh data and fill with zeros
        let mut dest_pack = dest_mesh.pack(None);
        dest_pack.fill(0.);

        // Accumulate weighted contributions from each output instance
        weights.iter().zip(io_data.iter()).for_each(|(&w, output)| {
            let output_mesh = output.meshes.get(&mesh_id).unwrap();
            let output_pack = output_mesh.pack(None);
            for (u_val, &output_val) in dest_pack.iter_mut().zip(output_pack.iter()) {
                *u_val += w * output_val;
            }
        });

        // Normalize quaternions in the destination mesh
        dest_mesh.nodes.iter_mut().for_each(|node| {
            node.ur = node.ur.normalize();
        });

        // Unpack the accumulated data back into the destination mesh
        dest_mesh.unpack(Fields::all(), &dest_pack);
    });
}
