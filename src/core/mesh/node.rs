use crate::core::types::*;

/// A node in a computational mesh representing a point in 3D space with kinematic and load information.
///
/// This structure represents a mesh node that can undergo both translational and rotational motion.
/// It stores reference positions/orientations, current displacements/rotations, velocities,
/// accelerations, and applied forces/moments. This is commonly used in structural dynamics,
/// fluid-structure interaction, and other multi-physics simulations.
///
/// # Coordinate System
///
/// - All positions are in global coordinates
/// - Rotations are represented as quaternions
/// - Angular velocities and accelerations are in global coordinates
///
/// # Examples
///
/// ```
/// use fastr::core::mesh::Node;
/// use fastr::core::{Vector3, Quaternion};
///
/// // Create a node at origin with identity orientation
/// let mut node = Node::new(0, Vector3::ZERO, Quaternion::IDENTITY);
///
/// // Move the node
/// node.translate(Vector3::new(1.0, 0.0, 0.0));
///
/// // Get current position
/// let current_pos = node.x();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier for this node
    pub id: usize,
    /// Reference (initial) position vector
    pub x0: Vector3,
    /// Translational displacement from reference position
    pub ut: Vector3,
    /// Translational velocity vector
    pub vt: Vector3,
    /// Translational acceleration vector
    pub at: Vector3,
    /// Reference (initial) orientation quaternion
    pub r0: Quaternion,
    /// Rotational displacement from reference orientation
    pub ur: Quaternion,
    /// Angular velocity vector
    pub vr: Vector3,
    /// Angular acceleration vector
    pub ar: Vector3,
    /// Applied force vector
    pub f: Vector3,
    /// Applied moment vector
    pub m: Vector3,
}

impl Node {
    /// Creates a new node with specified ID, reference position, and reference orientation.
    ///
    /// All kinematic quantities (displacements, velocities, accelerations) and loads
    /// are initialized to zero, representing a node at rest with no applied forces.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this node
    /// * `x0` - Reference (initial) position in global coordinates
    /// * `r0` - Reference (initial) orientation as a quaternion
    ///
    /// # Returns
    ///
    /// A new `Node` instance with zero initial motion and loads
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    ///
    /// let node = Node::new(1, Vector3::new(1.0, 2.0, 3.0), Quaternion::IDENTITY);
    /// assert_eq!(node.id, 1);
    /// assert_eq!(node.x(), Vector3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn new(id: usize, x0: Vector3, r0: Quaternion) -> Self {
        Self {
            id,
            x0,
            ut: Vector3::ZERO,
            vt: Vector3::ZERO,
            at: Vector3::ZERO,
            r0,
            ur: Quaternion::IDENTITY,
            vr: Vector3::ZERO,
            ar: Vector3::ZERO,
            f: Vector3::ZERO,
            m: Vector3::ZERO,
        }
    }

    /// Applies a translational displacement to the node.
    ///
    /// This method adds the given displacement vector to the current translational
    /// displacement, effectively moving the node by the specified amount.
    ///
    /// # Arguments
    ///
    /// * `dux` - Additional translational displacement vector
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    ///
    /// let mut node = Node::new(0, Vector3::ZERO, Quaternion::IDENTITY);
    /// node.translate(Vector3::new(1.0, 0.0, 0.0))
    ///     .translate(Vector3::new(0.0, 1.0, 0.0));
    /// assert_eq!(node.ut, Vector3::new(1.0, 1.0, 0.0));
    /// ```
    pub fn translate(&mut self, dux: Vector3) -> &mut Node {
        self.ut += dux;
        self
    }

    /// Applies a rotational displacement to the node.
    ///
    /// This method composes the given quaternion rotation with the current
    /// rotational displacement. The rotation is applied in the current orientation.
    ///
    /// # Arguments
    ///
    /// * `dq` - Additional rotation as a quaternion
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let mut node = Node::new(0, Vector3::ZERO, Quaternion::IDENTITY);
    /// let rotation = Quaternion::from_rotation_z(FRAC_PI_2);
    /// node.rotate(rotation);
    /// ```
    pub fn rotate(&mut self, dq: Quaternion) {
        self.ur = dq * self.ur;
    }

    /// Returns the current global position of the node.
    ///
    /// Calculates the current position by adding the translational displacement
    /// to the reference position: x = x₀ + uₜ
    ///
    /// # Returns
    ///
    /// The current position vector in global coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    ///
    /// let mut node = Node::new(0, Vector3::new(1.0, 0.0, 0.0), Quaternion::IDENTITY);
    /// node.translate(Vector3::new(0.5, 0.0, 0.0));
    /// assert_eq!(node.x(), Vector3::new(1.5, 0.0, 0.0));
    /// ```
    pub fn x(&self) -> Vector3 {
        self.x0 + self.ut
    }

    /// Returns the current global orientation of the node.
    ///
    /// Calculates the current orientation by composing the rotational displacement
    /// with the reference orientation: r = uᵣ * r₀
    ///
    /// # Returns
    ///
    /// The current orientation quaternion in global coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    ///
    /// let node = Node::new(0, Vector3::ZERO, Quaternion::IDENTITY);
    /// let current_orientation = node.r();
    /// ```
    pub fn r(&self) -> Quaternion {
        self.ur * self.r0
    }

    /// Sets the current global position of the node.
    ///
    /// This method calculates and updates the translational displacement required
    /// to achieve the specified global position: uₜ = x - x₀
    ///
    /// # Arguments
    ///
    /// * `x` - The desired global position
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    ///
    /// let mut node = Node::new(0, Vector3::new(1.0, 0.0, 0.0), Quaternion::IDENTITY);
    /// node.set_x(Vector3::new(2.0, 1.0, 0.0));
    /// assert_eq!(node.ut, Vector3::new(1.0, 1.0, 0.0));
    /// ```
    pub fn set_x(&mut self, x: Vector3) -> &mut Node {
        self.ut = x - self.x0;
        self
    }

    /// Sets the current global orientation of the node.
    ///
    /// This method calculates and updates the rotational displacement required
    /// to achieve the specified global orientation: uᵣ = r₀⁻¹ * q
    ///
    /// # Arguments
    ///
    /// * `q` - The desired global orientation quaternion
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    /// use std::f64::consts::PI;
    ///
    /// let mut node = Node::new(0, Vector3::ZERO, Quaternion::IDENTITY);
    /// let target_rotation = Quaternion::from_rotation_z(PI/4.0);
    /// node.set_r(target_rotation);
    /// ```
    pub fn set_r(&mut self, q: Quaternion) -> &mut Node {
        self.ur = self.r0.inverse() * q;
        self
    }

    /// Copies all motion-related quantities from another node.
    ///
    /// This method transfers position, displacement, velocity, acceleration, orientation,
    /// and rotational motion data from the source node to this node. Load data (forces
    /// and moments) are not copied.
    ///
    /// # Arguments
    ///
    /// * `other` - The source node to copy motion data from
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    ///
    /// let source = Node::new(1, Vector3::new(1.0, 0.0, 0.0), Quaternion::IDENTITY);
    /// let mut target = Node::new(2, Vector3::ZERO, Quaternion::IDENTITY);
    /// target.copy_motion_from(&source);
    /// assert_eq!(target.x0, source.x0);
    /// ```
    pub fn copy_motion_from(&mut self, other: &Node) -> &mut Self {
        self.x0 = other.x0;
        self.ut = other.ut;
        self.vt = other.vt;
        self.at = other.at;
        self.r0 = other.r0;
        self.ur = other.ur;
        self.vr = other.vr;
        self.ar = other.ar;
        self
    }

    /// Copies all load-related quantities from another node.
    ///
    /// This method transfers force and moment data from the source node to this node.
    /// Motion-related data is not affected.
    ///
    /// # Arguments
    ///
    /// * `other` - The source node to copy load data from
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    ///
    /// let source = Node::new(1, Vector3::ZERO, Quaternion::IDENTITY);
    /// let mut target = Node::new(2, Vector3::ZERO, Quaternion::IDENTITY);
    /// target.copy_loads_from(&source);
    /// assert_eq!(target.f, source.f);
    /// assert_eq!(target.m, source.m);
    /// ```
    pub fn copy_loads_from(&mut self, other: &Node) -> &mut Self {
        self.f = other.f;
        self.m = other.m;
        self
    }

    /// Resets all applied loads to zero.
    ///
    /// This method clears both force and moment vectors, effectively removing
    /// all applied loads from the node. This is commonly used at the beginning
    /// of load application cycles or when reinitializing the node state.
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::core::mesh::Node;
    /// use fastr::core::{Vector3, Quaternion};
    ///
    /// let mut node = Node::new(0, Vector3::ZERO, Quaternion::IDENTITY);
    /// // ... apply some loads ...
    /// node.reset_loads();
    /// assert_eq!(node.f, Vector3::ZERO);
    /// assert_eq!(node.m, Vector3::ZERO);
    /// ```
    pub fn reset_loads(&mut self) -> &mut Self {
        self.f = Vector3::ZERO;
        self.m = Vector3::ZERO;
        self
    }
}
