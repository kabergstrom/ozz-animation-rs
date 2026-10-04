//!
//! glTF to raw skeleton and raw animation (gltf2ozz's job, feature `gltf`).
//!
//! - `skin_roots` picks skeleton roots as gltf2ozz does: the roots of the default scene's skins,
//!   or the scene's nodes when it has no skin.
//! - `import_skeleton` turns node hierarchies into a `RawSkeleton`, every descendant a joint.
//! - `import_animation` turns one glTF animation into a `RawAnimation` over those joints:
//!   LINEAR keys copied, STEP keys doubled into steps, CUBICSPLINE sampled at `sample_rate`;
//!   components a clip leaves untouched get the node's rest pose.
//!
//! Joints are named as gltf2ozz names nodes: unnamed nodes become `node_{index}`, duplicates get
//! `_{index}` appended until unique.
//!

use std::collections::HashSet;

use glam::{Quat, Vec3, Vec3A, Vec4};
use glam_ext::Transform3A;
use gltf::animation::{Interpolation, Property};
use gltf::Document;
use thiserror::Error;

use super::raw_animation::RawKey;
use super::{
    normalize_safe_quat, OfflineError, RawAnimation, RawJoint, RawJointTrack, RawRotationKey, RawScaleKey, RawSkeleton,
    RawTranslationKey,
};

/// Sample rate used for CUBICSPLINE channels when `import_animation` gets 0, as gltf2ozz.
pub const DEFAULT_SAMPLE_RATE: f32 = 30.0;

/// glTF import error.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GltfError {
    /// The document has no scene.
    #[error("No scene")]
    NoScene,
    /// No animation with this index.
    #[error("No animation {0}")]
    NoAnimation(usize),
    /// A channel's input or output can't be read (missing buffer or unsupported accessor).
    #[error("Animation {animation} channel {channel}: unreadable sampler data")]
    UnreadableChannel { animation: usize, channel: usize },
    /// A channel's output count doesn't match its input count (3 per input for CUBICSPLINE).
    #[error("Animation {animation} channel {channel}: {inputs} inputs, {outputs} outputs")]
    KeyCountMismatch {
        animation: usize,
        channel: usize,
        inputs: usize,
        outputs: usize,
    },
    /// A channel's input times decrease.
    #[error("Animation {animation} channel {channel}: keyframe times are not sorted")]
    UnsortedKeys { animation: usize, channel: usize },
    /// Two channels of one animation target the same node property.
    #[error("Animation {animation} channel {channel}: node {node} property already animated")]
    DuplicateChannel {
        animation: usize,
        channel: usize,
        node: usize,
    },
    /// The resulting raw data is invalid.
    #[error("{0}")]
    Offline(#[from] OfflineError),
}

/// A `RawSkeleton` and the glTF node of each joint.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfSkeleton {
    /// The skeleton.
    pub raw: RawSkeleton,
    /// glTF node index per joint, in depth-first joint order (the runtime `Skeleton` order).
    pub joint_nodes: Vec<usize>,
}

/// Node names made unique the way gltf2ozz does (`FixupNames`), indexed by node.
pub fn node_names(document: &Document) -> Vec<String> {
    let mut names = HashSet::new();
    document
        .nodes()
        .map(|node| {
            let index = node.index();
            let mut name = match node.name() {
                Some(name) if !name.is_empty() => name.to_string(),
                _ => format!("node_{}", index),
            };
            while names.contains(&name) {
                name = format!("{}_{}", name, index);
            }
            names.insert(name.clone());
            name
        })
        .collect()
}

/// Skeleton roots as gltf2ozz picks them (sorted, unique node indices).
///
/// - Takes the default scene (scene 0 if none is set).
/// - For each skin whose first joint is in the scene: its `skeleton` node if set, else the topmost
///   ancestor of its first joint; ancestors that are another skin's `skeleton` node stop the search
///   and add nothing.
/// - Without skins, the scene's root nodes.
pub fn skin_roots(document: &Document) -> Result<Vec<usize>, GltfError> {
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or(GltfError::NoScene)?;

    let num_nodes = document.nodes().len();
    let mut in_scene = vec![false; num_nodes];
    let mut open: Vec<_> = scene.nodes().collect();
    while let Some(node) = open.pop() {
        if !in_scene[node.index()] {
            in_scene[node.index()] = true;
            open.extend(node.children());
        }
    }

    let skins: Vec<_> = document
        .skins()
        .filter(|skin| skin.joints().next().is_some_and(|joint| in_scene[joint.index()]))
        .collect();

    let mut roots = Vec::new();
    if skins.is_empty() {
        roots.extend(scene.nodes().map(|node| node.index()));
    } else {
        const NO_PARENT: isize = -1;
        const VISITED: isize = -2;
        let mut parents = vec![NO_PARENT; num_nodes];
        for node in document.nodes() {
            for child in node.children() {
                parents[child.index()] = node.index() as isize;
            }
        }
        for skin in &skins {
            if let Some(skeleton) = skin.skeleton() {
                parents[skeleton.index()] = VISITED;
                roots.push(skeleton.index());
                continue;
            }
            let mut root = skin.joints().next().unwrap().index() as isize;
            while root != VISITED && parents[root as usize] != NO_PARENT {
                root = parents[root as usize];
            }
            if root != VISITED {
                roots.push(root as usize);
            }
        }
    }

    roots.sort_unstable();
    roots.dedup();
    Ok(roots)
}

fn node_transform(node: &gltf::Node) -> Transform3A {
    let (t, r, s) = node.transform().decomposed();
    Transform3A {
        translation: Vec3A::from_array(t),
        rotation: Quat::from_array(r),
        scale: Vec3A::from_array(s),
    }
}

/// Builds a `RawSkeleton` from the hierarchies under `roots` (node indices). Every descendant node
/// becomes a joint, with the node's local transform as rest pose.
pub fn import_skeleton(document: &Document, roots: &[usize]) -> GltfSkeleton {
    let names = node_names(document);
    let nodes: Vec<_> = document.nodes().collect();

    // Depth-first order matches `SkeletonBuilder`'s.
    fn import(node: &gltf::Node, names: &[String], joint_nodes: &mut Vec<usize>) -> RawJoint {
        joint_nodes.push(node.index());
        let mut joint = RawJoint::new(names[node.index()].clone(), node_transform(node));
        joint.children = node
            .children()
            .map(|child| import(&child, names, joint_nodes))
            .collect();
        joint
    }

    let mut joint_nodes = Vec::new();
    let roots = roots
        .iter()
        .map(|&root| import(&nodes[root], &names, &mut joint_nodes))
        .collect();
    GltfSkeleton {
        raw: RawSkeleton { roots },
        joint_nodes,
    }
}

/// Value types sampled from glTF channels.
trait ChannelValue: Copy {
    fn hermite(alpha: f32, p0: Self, m0: Self, p1: Self, m1: Self) -> Self;
    fn scale(self, s: f32) -> Self;
}

impl ChannelValue for Vec3 {
    fn hermite(alpha: f32, p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3) -> Vec3 {
        let (a, b, c, d) = hermite_coefficients(alpha);
        p0 * a + m0 * b + p1 * c + m1 * d
    }

    fn scale(self, s: f32) -> Vec3 {
        self * s
    }
}

impl ChannelValue for Quat {
    // Component-wise, normalized afterwards like every imported rotation.
    fn hermite(alpha: f32, p0: Quat, m0: Quat, p1: Quat, m1: Quat) -> Quat {
        let (a, b, c, d) = hermite_coefficients(alpha);
        let v = Vec4::from(p0) * a + Vec4::from(m0) * b + Vec4::from(p1) * c + Vec4::from(m1) * d;
        Quat::from_vec4(v)
    }

    fn scale(self, s: f32) -> Quat {
        Quat::from_vec4(Vec4::from(self) * s)
    }
}

/// Hermite basis: p(t) = (2t³ - 3t² + 1)p0 + (t³ - 2t² + t)m0 + (-2t³ + 3t²)p1 + (t³ - t²)m1.
fn hermite_coefficients(alpha: f32) -> (f32, f32, f32, f32) {
    let t1 = alpha;
    let t2 = alpha * alpha;
    let t3 = t2 * alpha;
    (
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + t1,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
    )
}

/// The previous float toward zero (C `nexttowardf(x, 0)`).
fn next_toward_zero(x: f32) -> f32 {
    if x == 0.0 || x.is_nan() {
        x
    } else {
        f32::from_bits(x.to_bits() - 1)
    }
}

/// Keyframes of one channel, as gltf2ozz samples them. `outputs` holds 3 values per input for
/// CUBICSPLINE (in-tangent, value, out-tangent).
fn sample_channel<K: RawKey>(
    interpolation: Interpolation,
    times: &[f32],
    outputs: &[K::Value],
    sample_rate: f32,
) -> Vec<K>
where
    K::Value: ChannelValue,
{
    let count = times.len();
    if count == 0 {
        return Vec::new();
    }
    match interpolation {
        Interpolation::Linear => times.iter().zip(outputs).map(|(&t, &v)| K::new(t, v)).collect(),
        Interpolation::Step => {
            // A step is 2 keys: the value at its time and just before the next time.
            let mut keys = Vec::with_capacity(count * 2 - 1);
            for i in 0..count {
                keys.push(K::new(times[i], outputs[i]));
                if i + 1 < count {
                    keys.push(K::new(next_toward_zero(times[i + 1]), outputs[i]));
                }
            }
            keys
        }
        Interpolation::CubicSpline => {
            if count == 1 {
                return vec![K::new(times[0], outputs[1])];
            }
            // Fixed rate between the first and last time stamps (ozz `FixedRateSamplingTime`).
            let duration = times[count - 1] - times[0];
            let period = 1.0 / sample_rate;
            let num_keys = (1.0 + duration * sample_rate).ceil() as usize;
            let mut keys = Vec::with_capacity(num_keys);
            let mut k0 = 0;
            for k in 0..num_keys {
                let time = f32::min(k as f32 * period, duration) + times[0];
                while k0 + 2 < count && times[k0 + 1] < time {
                    k0 += 1;
                }
                let t0 = times[k0];
                let t1 = times[k0 + 1];
                let alpha = (time - t0) / (t1 - t0);
                let p0 = outputs[k0 * 3 + 1];
                let m0 = outputs[k0 * 3 + 2].scale(t1 - t0);
                let p1 = outputs[(k0 + 1) * 3 + 1];
                let m1 = outputs[(k0 + 1) * 3].scale(t1 - t0);
                keys.push(K::new(time, K::Value::hermite(alpha, p0, m0, p1, m1)));
            }
            keys
        }
    }
}

/// Samples a channel into `dest`: checks counts and order, drops keys at equal times (keeping the first).
fn import_channel<K: RawKey>(
    channel_id: (usize, usize),
    interpolation: Interpolation,
    times: &[f32],
    outputs: &[K::Value],
    sample_rate: f32,
) -> Result<Vec<K>, GltfError>
where
    K::Value: ChannelValue,
{
    let (animation, channel) = channel_id;
    let per_input = if interpolation == Interpolation::CubicSpline {
        3
    } else {
        1
    };
    if outputs.len() != times.len() * per_input {
        return Err(GltfError::KeyCountMismatch {
            animation,
            channel,
            inputs: times.len(),
            outputs: outputs.len(),
        });
    }
    let mut keys = sample_channel::<K>(interpolation, times, outputs, sample_rate);
    if keys.windows(2).any(|w| w[1].time() < w[0].time()) {
        return Err(GltfError::UnsortedKeys { animation, channel });
    }
    keys.dedup_by(|b, a| a.time() == b.time());
    Ok(keys)
}

/// Builds a `RawAnimation` from glTF animation `animation`, with one track per joint of `joint_nodes`
/// (glTF node index per joint, as `GltfSkeleton::joint_nodes`).
///
/// - `buffers`: buffer data by glTF buffer index (the GLB binary chunk for a .glb).
/// - `sample_rate`: CUBICSPLINE sampling rate in Hz; 0 means `DEFAULT_SAMPLE_RATE`.
/// - Channels on other nodes and morph target weight channels are ignored.
/// - Duration is the largest input `max` among channels on joints.
pub fn import_animation<B: AsRef<[u8]>>(
    document: &Document,
    buffers: &[B],
    animation: usize,
    joint_nodes: &[usize],
    sample_rate: f32,
) -> Result<RawAnimation, GltfError> {
    let gltf_animation = document
        .animations()
        .nth(animation)
        .ok_or(GltfError::NoAnimation(animation))?;
    let sample_rate = if sample_rate == 0.0 {
        DEFAULT_SAMPLE_RATE
    } else {
        sample_rate
    };

    let num_nodes = document.nodes().len();
    let mut node_joint = vec![None; num_nodes];
    for (joint, &node) in joint_nodes.iter().enumerate() {
        node_joint[node] = Some(joint);
    }

    let mut raw = RawAnimation {
        tracks: vec![RawJointTrack::default(); joint_nodes.len()],
        duration: 0.0,
        name: gltf_animation.name().unwrap_or_default().to_string(),
    };
    // Per joint: translation, rotation, scale channel seen.
    let mut animated = vec![[false; 3]; joint_nodes.len()];

    for (channel_index, channel) in gltf_animation.channels().enumerate() {
        let target = channel.target();
        let node = target.node().index();
        let Some(joint) = node_joint[node] else {
            continue;
        };
        let component = match target.property() {
            Property::Translation => 0,
            Property::Rotation => 1,
            Property::Scale => 2,
            Property::MorphTargetWeights => continue,
        };
        let channel_id = (animation, channel_index);
        if std::mem::replace(&mut animated[joint][component], true) {
            return Err(GltfError::DuplicateChannel {
                animation,
                channel: channel_index,
                node,
            });
        }

        let sampler = channel.sampler();
        let interpolation = sampler.interpolation();
        let reader = channel.reader(|buffer| buffers.get(buffer.index()).map(|b| b.as_ref()));
        let unreadable = GltfError::UnreadableChannel {
            animation,
            channel: channel_index,
        };
        let times: Vec<f32> = reader.read_inputs().ok_or(unreadable.clone())?.collect();

        let duration = sampler
            .input()
            .max()
            .and_then(|max| max.as_array().and_then(|a| a.first()).and_then(|v| v.as_f64()))
            .map(|max| max as f32)
            .or_else(|| times.last().copied())
            .unwrap_or(0.0);
        raw.duration = f32::max(raw.duration, duration);

        if times.is_empty() {
            continue;
        }

        let track = &mut raw.tracks[joint];
        match reader.read_outputs().ok_or(unreadable)? {
            gltf::animation::util::ReadOutputs::Translations(values) => {
                let values: Vec<Vec3> = values.map(Vec3::from_array).collect();
                track.translations = import_channel(channel_id, interpolation, &times, &values, sample_rate)?;
            }
            gltf::animation::util::ReadOutputs::Rotations(values) => {
                let values: Vec<Quat> = values.into_f32().map(Quat::from_array).collect();
                let mut keys: Vec<RawRotationKey> =
                    import_channel(channel_id, interpolation, &times, &values, sample_rate)?;
                for key in &mut keys {
                    key.value = normalize_safe_quat(key.value, Quat::IDENTITY);
                }
                track.rotations = keys;
            }
            gltf::animation::util::ReadOutputs::Scales(values) => {
                let values: Vec<Vec3> = values.map(Vec3::from_array).collect();
                track.scales = import_channel(channel_id, interpolation, &times, &values, sample_rate)?;
            }
            gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => {}
        }
    }

    // Rest pose for components the clip leaves untouched.
    let nodes: Vec<_> = document.nodes().collect();
    for (joint, track) in raw.tracks.iter_mut().enumerate() {
        let (t, r, s) = nodes[joint_nodes[joint]].transform().decomposed();
        if track.translations.is_empty() {
            track.translations.push(RawTranslationKey {
                time: 0.0,
                value: Vec3::from_array(t),
            });
        }
        if track.rotations.is_empty() {
            track.rotations.push(RawRotationKey {
                time: 0.0,
                value: Quat::from_array(r),
            });
        }
        if track.scales.is_empty() {
            track.scales.push(RawScaleKey {
                time: 0.0,
                value: Vec3::from_array(s),
            });
        }
    }

    raw.validate()?;
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(keys: &[RawTranslationKey]) -> Vec<(f32, f32)> {
        keys.iter().map(|k| (k.time, k.value.x)).collect()
    }

    #[test]
    fn test_sample_linear_and_step() {
        let times = [0.0, 0.5, 1.0];
        let values = [Vec3::splat(1.0), Vec3::splat(2.0), Vec3::splat(3.0)];
        let linear = sample_channel::<RawTranslationKey>(Interpolation::Linear, &times, &values, 30.0);
        assert_eq!(t(&linear), vec![(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]);

        let step = sample_channel::<RawTranslationKey>(Interpolation::Step, &times, &values, 30.0);
        assert_eq!(step.len(), 5);
        assert_eq!(step[1].time, f32::from_bits(0.5f32.to_bits() - 1));
        assert_eq!(step[1].value.x, 1.0);
        assert_eq!(step[2].time, 0.5);
        assert_eq!(step[3].value.x, 2.0);
        assert_eq!(step[4].time, 1.0);
    }

    #[test]
    fn test_sample_cubic_spline() {
        // Zero tangents: ease in/out between 0 and 1 over [0, 1], sampled at 4 Hz.
        let times = [0.0, 1.0];
        let z = Vec3::ZERO;
        let values = [z, Vec3::ZERO, z, z, Vec3::ONE, z];
        let keys = sample_channel::<RawTranslationKey>(Interpolation::CubicSpline, &times, &values, 4.0);
        assert_eq!(keys.len(), 5);
        assert_eq!(keys[0].time, 0.0);
        assert_eq!(keys[4].time, 1.0);
        assert_eq!(keys[2].value.x, 0.5);
        assert_eq!(keys[1].value.x, 0.15625); // 3t² - 2t³ at 0.25
        assert_eq!(keys[4].value.x, 1.0);

        // Linear tangents reproduce a line.
        let m = Vec3::ONE;
        let values = [m, Vec3::ZERO, m, m, Vec3::ONE, m];
        let keys = sample_channel::<RawTranslationKey>(Interpolation::CubicSpline, &times, &values, 4.0);
        for key in keys {
            assert!((key.value.x - key.time).abs() < 1e-6);
        }
    }

    #[test]
    fn test_import_channel_checks() {
        let values = [Vec3::ZERO, Vec3::ONE, Vec3::ONE];
        let keys: Vec<RawTranslationKey> =
            import_channel((0, 0), Interpolation::Linear, &[0.0, 0.5, 0.5], &values, 30.0).unwrap();
        assert_eq!(t(&keys), vec![(0.0, 0.0), (0.5, 1.0)]);

        let err = import_channel::<RawTranslationKey>((0, 1), Interpolation::Linear, &[0.0, 1.0, 0.5], &values, 30.0)
            .unwrap_err();
        assert_eq!(
            err,
            GltfError::UnsortedKeys {
                animation: 0,
                channel: 1
            }
        );

        let err = import_channel::<RawTranslationKey>((0, 2), Interpolation::CubicSpline, &[0.0, 1.0], &values, 30.0)
            .unwrap_err();
        assert!(matches!(err, GltfError::KeyCountMismatch { outputs: 3, .. }));
    }
}
