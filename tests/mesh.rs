extern crate fastr;

use std::{f64::consts::PI, fs};

use approx::assert_relative_eq;
use faer::prelude::*;
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
    let orbit = earth.create_motion_mapping(&moon);

    // set translational velocity of the earth
    earth.nodes[0].vt.x = 0.05;

    // Set rotational velocity of the earth
    earth.nodes[0].vr.z = 0.1;

    // Transfer motion from earth to moon via the mapping
    orbit.transfer_motion(&earth, &mut moon);

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
        orbit.transfer_motion(&earth, &mut moon);
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
    let mapping = src.create_motion_mapping(&dst);

    // Transfer motion from source to destination via the mapping
    mapping.transfer_motion(&src, &mut dst);

    // Define perturbation size
    let perturb = 1e-5;

    // Linearization of destination displacement wrt source displacement
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
                mapping.transfer_motion(&src, &mut dst);
                let ux_p = dst.nodes[0].ut;

                src.copy_motion_from(&src_ref);
                src.nodes[0].translate(-p);
                mapping.transfer_motion(&src, &mut dst);
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

    // Linearization of destination displacement wrt source rotation
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
                mapping.transfer_motion(&src, &mut dst);
                let ux_p = dst.nodes[0].ut;

                src.copy_motion_from(&src_ref);
                src.nodes[0].rotate(q.inverse());
                mapping.transfer_motion(&src, &mut dst);
                let ux_m = dst.nodes[0].ut;

                (ux_p - ux_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    src.copy_motion_from(&src_ref);
    mapping.transfer_motion(&src, &mut dst);

    let exp = (src.nodes[0].x() - dst.nodes[0].x())
        .skew_symmetric()
        .transpose();

    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            assert_relative_eq!(m[(i, j)], exp[(i, j)], epsilon = 1e-10);
        });
    });

    // Linearization of destination rotation velocity wrt source rotation velocity
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
                src.nodes[0].vr += p;
                mapping.transfer_motion(&src, &mut dst);
                let vr_p = dst.nodes[0].vr;

                src.copy_motion_from(&src_ref);
                src.nodes[0].vr -= p;
                mapping.transfer_motion(&src, &mut dst);
                let vr_m = dst.nodes[0].vr;

                (vr_p - vr_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    let exp = Matrix3::identity();
    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            assert_relative_eq!(m[(i, j)], exp[(i, j)], epsilon = 1e-10);
        });
    });

    // Linearization of destination translation velocity wrt source rotation velocity
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
                src.nodes[0].vr += p;
                mapping.transfer_motion(&src, &mut dst);
                let vr_p = dst.nodes[0].vr;

                src.copy_motion_from(&src_ref);
                src.nodes[0].vr -= p;
                mapping.transfer_motion(&src, &mut dst);
                let vr_m = dst.nodes[0].vr;

                (vr_p - vr_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    let exp = Matrix3::identity();
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
    let mut dst = build_point_mesh_at(1., 1., 0.);

    // Create a mapping from source to destination
    let mapping = src.create_load_mapping(&dst);

    // Apply loads to source mesh
    src.nodes[0].f = Vector3::new(1., 2., 3.);
    src.nodes[0].m = Vector3::new(4., 5., 6.);

    // Create copies of the source and destination meshes for reference
    let src_ref = src.clone();
    let dst_ref = dst.clone();

    // Define perturbation size
    let perturb = 1e-3;

    // Linearization of destination force wrt source force
    let m = Matrix3::from_rows(
        &(0..3)
            .map(|i| {
                let p = match i {
                    0 => Vector3::new(perturb, 0., 0.),
                    1 => Vector3::new(0., perturb, 0.),
                    2 => Vector3::new(0., 0., perturb),
                    _ => Vector3::zero(),
                };

                src.nodes[0].f = p;
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let f_p = dst.nodes[0].f;

                src.nodes[0].f = -p;
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let f_m = dst.nodes[0].f;

                (f_p - f_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    println!("\ndF/dF_S = \n{}", m);
    let exp = Matrix3::identity();
    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            assert_relative_eq!(m[(i, j)], exp[(i, j)], epsilon = 1e-10);
        });
    });

    // Linearization of moment wrt destination translational displacement
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
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let m_p = dst.nodes[0].m;

                dst.copy_motion_from(&dst_ref);
                dst.nodes[0].translate(-p);
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let m_m = dst.nodes[0].m;

                (m_p - m_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    println!("\ndM/dut_D = \n{}", m);
    println!(
        "dM/dut_D (expected) = \n{}",
        src.nodes[0].f.skew_symmetric()
    );

    // Linearization of moment wrt source translational displacement
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
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let m_p = dst.nodes[0].m;

                src.copy_motion_from(&src_ref);
                src.nodes[0].translate(-p);
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let m_m = dst.nodes[0].m;

                (m_p - m_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    println!("\ndM/dut_S (actual) = \n{}", m);
    println!(
        "dM/dut_S (expected) = \n{}",
        (-src.nodes[0].f).skew_symmetric()
    );

    // let exp = Matrix3::identity();
    // (0..3).for_each(|i| {
    //     (0..3).for_each(|j| {
    //         assert_relative_eq!(m[(i, j)], exp[(i, j)], epsilon = 1e-10);
    //     });
    // });

    // Linearization of destination force wrt source force
    let m = Matrix3::from_rows(
        &(0..3)
            .map(|i| {
                let p = match i {
                    0 => Vector3::new(perturb, 0., 0.),
                    1 => Vector3::new(0., perturb, 0.),
                    2 => Vector3::new(0., 0., perturb),
                    _ => Vector3::zero(),
                };

                src.nodes[0].f = p;
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let m_p = dst.nodes[0].m;

                src.nodes[0].f = -p;
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let m_m = dst.nodes[0].m;

                (m_p - m_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    println!("\ndM_D/dF_S (actual) = \n{}", m);
    println!(
        "dM_D/dF_S (expected) = \n{}",
        (src.nodes[0].x() - dst.nodes[0].x()).skew_symmetric()
    );

    // Linearization of destination force wrt source force
    let m = Matrix3::from_rows(
        &(0..3)
            .map(|i| {
                let p = match i {
                    0 => Vector3::new(perturb, 0., 0.),
                    1 => Vector3::new(0., perturb, 0.),
                    2 => Vector3::new(0., 0., perturb),
                    _ => Vector3::zero(),
                };

                src.nodes[0].m = p;
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let m_p = dst.nodes[0].m;

                src.nodes[0].m = -p;
                dst.reset_loads();
                mapping.transfer_loads(&src, &mut dst);
                let m_m = dst.nodes[0].m;

                (m_p - m_m) / (2. * perturb)
            })
            .collect_vec(),
    );

    println!("\ndM_D/dM_S = \n{}", m);
}

#[test]
fn test_mesh_velocity_linearization() {
    let p_s = Vector3::new(0., 0., 0.);
    let p_d = Vector3::new(5., 0., 0.);

    let omega_s = Vector3::new(PI / 8., PI / 12., PI / 4.); // rad/s

    // Change in source displacement
    let du_s = Vector3::new(1., 2., 3.);

    // Change in theta
    let dtheta_s = Vector3::new(0.02, 0.03, 0.01);

    // Change in destination displacement due to change in source theta
    let du_d = du_s + (p_s - p_d).skew_symmetric() * dtheta_s;

    // Change in velocity
    let dv_s = Vector3::new(0.5, 0.7, 0.1);

    // Change in angular velocity
    let domega = Vector3::new(0.1, 0.2, 0.3);

    let v_1 = col![
        dtheta_s.x, dtheta_s.y, dtheta_s.z, dv_s.x, dv_s.y, dv_s.z, domega.x, domega.y, domega.z
    ];

    let mut m1 = Mat::<f64>::zeros(3, 9);
    let dp = p_s - p_d;
    m1.submatrix_mut(0, 0, 3, 3).copy_from(
        mat![
            [dp.x * omega_s.x, dp.x * omega_s.y, dp.x * omega_s.z],
            [dp.y * omega_s.x, dp.y * omega_s.y, dp.y * omega_s.z],
            [dp.z * omega_s.x, dp.z * omega_s.y, dp.z * omega_s.z]
        ] - dp.dot(&omega_s) * Mat::<f64>::identity(3, 3),
    );
    m1.submatrix_mut(0, 3, 3, 3)
        .copy_from(Mat::<f64>::identity(3, 3));
    m1.submatrix_mut(0, 6, 3, 3)
        .copy_from(dp.skew_symmetric().into_faer());

    let v_2 = col![
        du_d.x, du_d.y, du_d.z, du_s.x, du_s.y, du_s.z, dv_s.x, dv_s.y, dv_s.z, domega.x, domega.y,
        domega.z
    ];

    let mut m2 = Mat::<f64>::zeros(3, 12);
    m2.submatrix_mut(0, 0, 3, 3)
        .copy_from(omega_s.skew_symmetric().into_faer());
    m2.submatrix_mut(0, 3, 3, 3)
        .copy_from((-omega_s).skew_symmetric().into_faer());
    m2.submatrix_mut(0, 6, 3, 3)
        .copy_from(Mat::<f64>::identity(3, 3));
    m2.submatrix_mut(0, 9, 3, 3)
        .copy_from(dp.skew_symmetric().into_faer());

    println!("m1*v_1 = {:?}", m1 * v_1);
    println!("m2*v_2 = {:?}", m2 * v_2);
}
