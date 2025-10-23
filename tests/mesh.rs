extern crate fastr;

use std::{f64::consts::PI, fs};

use approx::assert_relative_eq;
use fastr::nwtc::{Matrix3, MeshBuilder, Quaternion, Vector3};
use itertools::Itertools;

fn build_point_mesh_at(x: f64, y: f64, z: f64) -> fastr::nwtc::mesh::Mesh {
    let mut mb = MeshBuilder::new();
    let node_id = mb
        .add_node()
        .set_position(x, y, z)
        .set_orientation(Quaternion::identity())
        .build();
    mb.add_point_element(node_id);
    mb.build()
}

#[test]
fn orbiting_mesh() {
    // Create earth mesh
    let mut earth = build_point_mesh_at(0., 0., 0.);

    // Create moon mesh
    let mut moon = build_point_mesh_at(1., 0., 0.);

    // Create a mapping from earth to moon
    let orbit = earth.create_mapping(&moon);

    // set translational velocity of the earth
    earth.nodes[0].vt.x = 0.05;

    // Set rotational velocity of the earth
    earth.nodes[0].vr.z = 0.1;

    // Transfer motion from earth to moon via the mapping
    orbit.map_motion(&earth, &mut moon);

    // Check that the moon's translational velocity is correct
    let expected_vx = earth.nodes[0].vt + Vector3::new(0., 0.1, 0.);
    assert_eq!(moon.nodes[0].vt, expected_vx);

    let vtk_dir = "tests/vtk";
    fs::create_dir_all(vtk_dir).unwrap();

    let dt = PI / 9.9;
    (0..100).for_each(|step| {
        // Export to VTK
        earth
            .to_vtk()
            .export_ascii(format!("{}/earth_{:03}.vtk", vtk_dir, step))
            .unwrap();

        moon.to_vtk()
            .export_ascii(format!("{}/moon_{:03}.vtk", vtk_dir, step))
            .unwrap();

        // Update earth position and orientation
        let dx = earth.nodes[0].vt * dt;
        earth.nodes[0].translate(dx);
        let dr = Quaternion::from_vector(earth.nodes[0].vr * dt);
        earth.nodes[0].rotate(dr);

        // Map motion to moon
        orbit.map_motion(&earth, &mut moon);
    });
}

#[test]
fn mesh_displacement_linearization() {
    // Create source mesh
    let mut src = build_point_mesh_at(0., 0., 0.);

    // Create destination mesh
    let mut dst = build_point_mesh_at(1., 0., 0.);

    // Displace source node
    src.nodes[0]
        .translate(Vector3::new(0., 0.5, 0.))
        .rotate(Quaternion::from_vector(Vector3::new(0., 0., PI / 6.)));

    // Create a copy of the source mesh for reference
    let src_ref = src.clone();

    // Create a mapping from source to destination
    let mapping = src.create_mapping(&dst);

    // Transfer motion from source to destination via the mapping
    mapping.map_motion(&src, &mut dst);

    // Define perturbation size
    let perturb = 1e-5;

    // Calculate linearization of translational displacement wrt source node displacement
    let m = Matrix3::from_rows(
        &(0..3)
            .map(|i| {
                let p = match i {
                    0 => Vector3::new(perturb, 0., 0.),
                    1 => Vector3::new(0., perturb, 0.),
                    2 => Vector3::new(0., 0., perturb),
                    _ => Vector3::zero(),
                };

                src.copy_motion_from(&src_ref);
                src.nodes[0].translate(p);
                mapping.map_motion(&src, &mut dst);
                let ux_p = dst.nodes[0].ut;

                src.copy_motion_from(&src_ref);
                src.nodes[0].translate(-p);
                mapping.map_motion(&src, &mut dst);
                let ux_m = dst.nodes[0].ut;

                (ux_p - ux_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    let exp = Matrix3::identity();
    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            assert_relative_eq!(m[(i, j)], exp[(i, j)], epsilon = 1e-10);
        });
    });

    // Calculate linearization of translational displacement wrt source node rotation
    let m = Matrix3::from_rows(
        &(0..3)
            .map(|i| {
                // Apply perturbation as a quaternion
                let q = Quaternion::from_vector(match i {
                    0 => Vector3::new(perturb, 0., 0.),
                    1 => Vector3::new(0., perturb, 0.),
                    2 => Vector3::new(0., 0., perturb),
                    _ => Vector3::zero(),
                });

                src.copy_motion_from(&src_ref);
                src.nodes[0].rotate(q);
                mapping.map_motion(&src, &mut dst);
                let ux_p = dst.nodes[0].ut;

                src.copy_motion_from(&src_ref);
                src.nodes[0].rotate(q.inverse());
                mapping.map_motion(&src, &mut dst);
                let ux_m = dst.nodes[0].ut;

                (ux_p - ux_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    src.copy_motion_from(&src_ref);
    mapping.map_motion(&src, &mut dst);

    let exp = (src.nodes[0].x() - dst.nodes[0].x())
        .skew_symmetric()
        .transpose();

    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            assert_relative_eq!(m[(i, j)], exp[(i, j)], epsilon = 1e-10);
        });
    });
}

#[test]
fn mesh_loads_linearization() {
    // Create source mesh
    let mut src = build_point_mesh_at(0., 0., 0.);

    // Create destination mesh
    let mut dst = build_point_mesh_at(1., 0., 0.);

    // Create a mapping from source to destination
    let mapping = src.create_mapping(&dst);

    // Apply loads to source mesh
    src.nodes[0].f = Vector3::new(1., 2., 3.);
    src.nodes[0].m = Vector3::new(4., 5., 6.);

    // Create copies of the source and destination meshes for reference
    let src_ref = src.clone();
    let dst_ref = dst.clone();

    // Define perturbation size
    let perturb = 1e-3;

    // Calculate linearization of moment wrt source translational displacement
    let m = Matrix3::from_rows(
        &(0..3)
            .map(|i| {
                let p = match i {
                    0 => Vector3::new(perturb, 0., 0.),
                    1 => Vector3::new(0., perturb, 0.),
                    2 => Vector3::new(0., 0., perturb),
                    _ => Vector3::zero(),
                };

                src.copy_motion_from(&src_ref);
                src.nodes[0].translate(p);
                mapping.map_loads(&src, &mut dst);
                let m_p = dst.nodes[0].m;

                src.copy_motion_from(&src_ref);
                src.nodes[0].translate(-p);
                mapping.map_loads(&src, &mut dst);
                let m_m = dst.nodes[0].m;

                (m_p - m_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    println!("dM_D/dut_S = \n{}", m);

    // let exp = Matrix3::identity();
    // (0..3).for_each(|i| {
    //     (0..3).for_each(|j| {
    //         assert_relative_eq!(m[(i, j)], exp[(i, j)], epsilon = 1e-10);
    //     });
    // });

    // Calculate linearization of moment wrt destination translational displacement
    let m = Matrix3::from_rows(
        &(0..3)
            .map(|i| {
                let p = match i {
                    0 => Vector3::new(perturb, 0., 0.),
                    1 => Vector3::new(0., perturb, 0.),
                    2 => Vector3::new(0., 0., perturb),
                    _ => Vector3::zero(),
                };

                dst.copy_motion_from(&dst_ref);
                dst.nodes[0].translate(p);
                mapping.map_loads(&src, &mut dst);
                let m_p = dst.nodes[0].m;

                dst.copy_motion_from(&dst_ref);
                dst.nodes[0].translate(-p);
                mapping.map_loads(&src, &mut dst);
                let m_m = dst.nodes[0].m;

                (m_p - m_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    println!("dM_D/dut_D = \n{}", m);
}
