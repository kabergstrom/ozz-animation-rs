//!
//! Offline skeleton and its builder (ozz `RawSkeleton`, `SkeletonBuilder`).
//!

use bimap::BiHashMap;
use glam::Quat;
use glam_ext::Transform3A;

use super::OfflineError;
use crate::base::{DeterministicState, SKELETON_MAX_JOINTS, SKELETON_NO_PARENT};
use crate::math::SoaTransform;
use crate::skeleton::{Skeleton, SkeletonRaw};

/// Offline skeleton joint.
#[derive(Debug, Clone, PartialEq)]
pub struct RawJoint {
    /// Joint name, unique within the skeleton.
    pub name: String,
    /// Rest pose in parent space.
    pub transform: Transform3A,
    /// Children joints.
    pub children: Vec<RawJoint>,
}

impl RawJoint {
    /// A joint with the given name and rest pose, without children.
    pub fn new(name: impl Into<String>, transform: Transform3A) -> RawJoint {
        RawJoint {
            name: name.into(),
            transform,
            children: Vec::new(),
        }
    }
}

///
/// Offline skeleton: a hierarchy of named joints with rest poses.
/// `SkeletonBuilder` converts it to the runtime `Skeleton`.
///
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawSkeleton {
    /// Skeleton roots. Can be empty if the skeleton has no joint.
    pub roots: Vec<RawJoint>,
}

impl RawSkeleton {
    /// Tests validity: joint count within `SKELETON_MAX_JOINTS`.
    pub fn validate(&self) -> Result<(), OfflineError> {
        let num_joints = self.num_joints();
        if num_joints > SKELETON_MAX_JOINTS as usize {
            return Err(OfflineError::TooManyJoints(num_joints));
        }
        Ok(())
    }

    /// Number of joints. Not constant time: iterates the hierarchy.
    pub fn num_joints(&self) -> usize {
        let mut count = 0;
        self.iter_depth_first(|_, _| count += 1);
        count
    }

    /// Iterates joints in depth-first order, the runtime `Skeleton` order.
    ///
    /// * `f` - Called with `(joint, parent)`; `parent` is `None` for roots.
    pub fn iter_depth_first<'t, F>(&'t self, mut f: F)
    where
        F: FnMut(&'t RawJoint, Option<&'t RawJoint>),
    {
        fn recurse<'t, F: FnMut(&'t RawJoint, Option<&'t RawJoint>)>(
            children: &'t [RawJoint],
            parent: Option<&'t RawJoint>,
            f: &mut F,
        ) {
            for joint in children {
                f(joint, parent);
                recurse(&joint.children, Some(joint), f);
            }
        }
        recurse(&self.roots, None, &mut f);
    }
}

///
/// Builds a runtime `Skeleton` from a `RawSkeleton`.
///
/// Joints are ordered depth-first, which makes a sub-hierarchy a contiguous range.
/// Rest pose rotations are normalized (identity when degenerate); SoA padding lanes are identity.
///
#[derive(Debug, Default, Clone, Copy)]
pub struct SkeletonBuilder;

impl SkeletonBuilder {
    /// Builds the runtime `Skeleton`.
    ///
    /// Fails if `raw` doesn't validate, has no joint (the runtime `Skeleton` can't be empty), or if two joints
    /// share a name (the runtime name map is one to one).
    pub fn build(&self, raw: &RawSkeleton) -> Result<Skeleton, OfflineError> {
        raw.validate()?;

        // Lists joints in depth-first order with their parent index.
        let mut joints: Vec<(&RawJoint, i16)> = Vec::with_capacity(raw.num_joints());
        fn list<'t>(children: &'t [RawJoint], parent: i16, joints: &mut Vec<(&'t RawJoint, i16)>) {
            for joint in children {
                let index = joints.len() as i16;
                joints.push((joint, parent));
                list(&joint.children, index, joints);
            }
        }
        list(&raw.roots, SKELETON_NO_PARENT as i16, &mut joints);

        let num_joints = joints.len();
        if num_joints == 0 {
            return Err(OfflineError::EmptySkeleton);
        }
        let mut joint_names =
            BiHashMap::with_capacity_and_hashers(num_joints, DeterministicState::new(), DeterministicState::new());
        for (index, (joint, _)) in joints.iter().enumerate() {
            if joint_names
                .insert_no_overwrite(joint.name.clone(), index as i16)
                .is_err()
            {
                return Err(OfflineError::DuplicateJointName(joint.name.clone()));
            }
        }

        let mut joint_rest_poses = vec![SoaTransform::IDENTITY; num_joints.div_ceil(4)];
        for (index, (joint, _)) in joints.iter().enumerate() {
            let transform = Transform3A {
                translation: joint.transform.translation,
                rotation: normalize_safe4(joint.transform.rotation),
                scale: joint.transform.scale,
            };
            joint_rest_poses[index / 4].set_transform(index % 4, transform);
        }

        let raw = SkeletonRaw {
            joint_rest_poses,
            joint_parents: joints.iter().map(|(_, parent)| *parent).collect(),
            joint_names,
        };
        Ok(Skeleton::from_raw(&raw))
    }
}

/// SIMD NormalizeSafe4 as ozz SSE2 builds compute it: dot as (x² + z²) + (y² + w²).
fn normalize_safe4(q: Quat) -> Quat {
    let sq_len = (q.x * q.x + q.z * q.z) + (q.y * q.y + q.w * q.w);
    if sq_len <= 0.0 {
        return Quat::IDENTITY;
    }
    let inv_len = 1.0 / sq_len.sqrt();
    Quat::from_xyzw(q.x * inv_len, q.y * inv_len, q.z * inv_len, q.w * inv_len)
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3, Vec3A};

    use super::*;

    fn joint(name: &str, x: f32, children: Vec<RawJoint>) -> RawJoint {
        RawJoint {
            name: name.into(),
            transform: Transform3A {
                translation: Vec3A::new(x, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3A::ONE,
            },
            children,
        }
    }

    #[test]
    fn test_build_depth_first() {
        let raw = RawSkeleton {
            roots: vec![
                joint(
                    "a",
                    1.0,
                    vec![joint("b", 2.0, vec![joint("c", 3.0, vec![])]), joint("d", 4.0, vec![])],
                ),
                joint("e", 5.0, vec![]),
            ],
        };
        assert_eq!(raw.num_joints(), 5);
        let skeleton = SkeletonBuilder.build(&raw).unwrap();
        assert_eq!(skeleton.num_joints(), 5);
        assert_eq!(skeleton.joint_parents(), &[-1, 0, 1, 0, -1]);
        for (i, name) in ["a", "b", "c", "d", "e"].iter().enumerate() {
            assert_eq!(skeleton.joint_by_name(name), Some(i as i16));
        }
        let poses = skeleton.joint_rest_poses();
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0].translation.vec3(3), Vec3::new(4.0, 0.0, 0.0));
        assert_eq!(poses[1].translation.vec3(0), Vec3::new(5.0, 0.0, 0.0));
        // Padding lanes are identity.
        assert_eq!(poses[1].transform(1).rotation, Quat::IDENTITY);
        assert_eq!(Vec3::from(poses[1].transform(3).scale), Vec3::ONE);
    }

    #[test]
    fn test_build_normalizes_rotation() {
        let mut root = joint("root", 0.0, vec![]);
        root.transform.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
        let mut child = joint("child", 0.0, vec![]);
        child.transform.rotation = Quat::from_xyzw(0.0, 0.0, 2.0, 0.0);
        root.children.push(child);
        let skeleton = SkeletonBuilder.build(&RawSkeleton { roots: vec![root] }).unwrap();
        assert_eq!(skeleton.joint_rest_poses()[0].rotation.quat(0), Quat::IDENTITY);
        assert_eq!(
            skeleton.joint_rest_poses()[0].rotation.quat(1),
            Quat::from_xyzw(0.0, 0.0, 1.0, 0.0)
        );
    }

    #[test]
    fn test_build_rejects() {
        let raw = RawSkeleton {
            roots: vec![joint("a", 0.0, vec![joint("a", 0.0, vec![])])],
        };
        assert_eq!(
            SkeletonBuilder.build(&raw).unwrap_err(),
            OfflineError::DuplicateJointName("a".into())
        );

        let raw = RawSkeleton {
            roots: (0..=SKELETON_MAX_JOINTS)
                .map(|i| joint(&i.to_string(), 0.0, vec![]))
                .collect(),
        };
        assert!(matches!(
            SkeletonBuilder.build(&raw).unwrap_err(),
            OfflineError::TooManyJoints(_)
        ));
        assert_eq!(
            SkeletonBuilder.build(&RawSkeleton::default()).unwrap_err(),
            OfflineError::EmptySkeleton
        );
    }
}
