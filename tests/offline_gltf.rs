//!
//! Round trip of the offline builders against gltf2ozz (ozz-animation 0.16, default settings:
//! optimizer 1 mm at 10 cm, iframe interval 10 s, 30 Hz sampling).
//!
//! resource/fox: Fox.glb from the glTF sample models (model CC0 by PixelMannen, rigging and
//! animation CC-BY 4.0 by @tomkranis), and the gltf2ozz archives deferred-ngp shipped for it.
//!
#![cfg(feature = "gltf")]

use glam::{Mat4, Quat, Vec3};
use ozz_animation_rs::offline::gltf::{import_animation, import_skeleton, skin_roots};
use ozz_animation_rs::offline::{AnimationBuilder, AnimationOptimizer, SkeletonBuilder};
use ozz_animation_rs::*;
use std::cell::RefCell;
use std::rc::Rc;

const SAMPLE_STEP: f32 = 0.001;
// Error bounds, in Fox units (the skeleton is about 100 units long).
const MAX_LOCAL_TRANSLATION_ERROR: f32 = 1e-3;
const MAX_ROTATION_ERROR_RAD: f32 = 1e-3;
const MAX_SCALE_ERROR: f32 = 1e-4;
const MAX_MODEL_POSITION_ERROR: f32 = 1e-2;

fn load_fox() -> (gltf::Document, Vec<Vec<u8>>) {
    let bytes = std::fs::read("./resource/fox/Fox.glb").unwrap();
    let gltf = gltf::Gltf::from_slice(&bytes).unwrap();
    let blob = gltf.blob.clone().unwrap();
    (gltf.document, vec![blob])
}

fn build_fox_skeleton(document: &gltf::Document) -> (Skeleton, Vec<usize>) {
    let roots = skin_roots(document).unwrap();
    let imported = import_skeleton(document, &roots);
    let skeleton = SkeletonBuilder.build(&imported.raw).unwrap();
    (skeleton, imported.joint_nodes)
}

fn read_skeleton(bytes: &[u8]) -> Skeleton {
    Skeleton::from_archive(&mut Archive::from_slice(bytes).unwrap()).unwrap()
}

fn read_animation(bytes: &[u8]) -> Animation {
    Animation::from_archive(&mut Archive::from_slice(bytes).unwrap()).unwrap()
}

#[test]
fn test_fox_skeleton_matches_gltf2ozz() {
    let (document, _) = load_fox();
    let (built, _) = build_fox_skeleton(&document);

    // Written and read back through `Archive`.
    let skeleton = read_skeleton(&built.to_archive_bytes().unwrap());
    let reference = Skeleton::from_path("./resource/fox/skeleton.ozz").unwrap();

    assert_eq!(skeleton.num_joints(), reference.num_joints());
    assert_eq!(skeleton.joint_parents(), reference.joint_parents());
    for joint in 0..skeleton.num_joints() as i16 {
        assert_eq!(skeleton.name_by_joint(joint), reference.name_by_joint(joint));
    }
    for joint in 0..skeleton.num_joints() {
        let a = skeleton.joint_rest_poses()[joint / 4].transform(joint % 4);
        let b = reference.joint_rest_poses()[joint / 4].transform(joint % 4);
        assert!(a.translation.abs_diff_eq(b.translation, 1e-6), "joint {}", joint);
        assert!(a.rotation.abs_diff_eq(b.rotation, 1e-6), "joint {}", joint);
        assert!(a.scale.abs_diff_eq(b.scale, 1e-6), "joint {}", joint);
    }
    let bytes = std::fs::read("./resource/fox/skeleton.ozz").unwrap();
    println!(
        "skeleton: byte-identical to gltf2ozz: {}",
        built.to_archive_bytes().unwrap() == bytes
    );
}

#[derive(Debug, Default)]
struct SampleErrors {
    translation: f32,
    rotation: f32,
    scale: f32,
    model_position: f32,
}

/// Samples both animations every `SAMPLE_STEP` seconds over the duration (end included) and
/// returns the largest per-joint differences.
fn compare_by_sampling(skeleton: &Rc<Skeleton>, a: Animation, b: Animation) -> SampleErrors {
    let duration = a.duration();
    let num_joints = skeleton.num_joints();
    let mut jobs = [Rc::new(a), Rc::new(b)].map(|animation| {
        let mut sample_job: SamplingJob = SamplingJob::default();
        sample_job.set_context(SamplingContext::new(animation.num_tracks()));
        sample_job.set_animation(animation);
        let sample_out = Rc::new(RefCell::new(vec![SoaTransform::default(); skeleton.num_soa_joints()]));
        sample_job.set_output(sample_out.clone());

        let mut l2m_job: LocalToModelJob = LocalToModelJob::default();
        l2m_job.set_skeleton(skeleton.clone());
        l2m_job.set_input(sample_out.clone());
        let l2m_out = Rc::new(RefCell::new(vec![Mat4::default(); num_joints]));
        l2m_job.set_output(l2m_out.clone());
        (sample_job, l2m_job, sample_out, l2m_out)
    });

    let mut errors = SampleErrors::default();
    let steps = (duration / SAMPLE_STEP).ceil() as usize;
    for step in 0..=steps {
        let time = f32::min(step as f32 * SAMPLE_STEP, duration);
        for (sample_job, l2m_job, _, _) in jobs.iter_mut() {
            sample_job.set_ratio(time / duration);
            sample_job.run().unwrap();
            l2m_job.run().unwrap();
        }
        let [(_, _, local_a, model_a), (_, _, local_b, model_b)] = &jobs;
        let (local_a, local_b) = (local_a.borrow(), local_b.borrow());
        let (model_a, model_b) = (model_a.borrow(), model_b.borrow());
        for joint in 0..num_joints {
            let ta = local_a[joint / 4].transform(joint % 4);
            let tb = local_b[joint / 4].transform(joint % 4);
            errors.translation = errors
                .translation
                .max(Vec3::from(ta.translation).distance(tb.translation.into()));
            errors.rotation = errors.rotation.max(rotation_angle(ta.rotation, tb.rotation));
            errors.scale = errors
                .scale
                .max((Vec3::from(ta.scale) - Vec3::from(tb.scale)).abs().max_element());
            let pa = model_a[joint].w_axis.truncate();
            let pb = model_b[joint].w_axis.truncate();
            errors.model_position = errors.model_position.max(pa.distance(pb));
        }
    }
    errors
}

/// Shortest angle between two rotations (q and -q are the same rotation).
fn rotation_angle(a: Quat, b: Quat) -> f32 {
    let d = a.conjugate() * b;
    2.0 * f32::atan2(Vec3::new(d.x, d.y, d.z).length(), d.w.abs())
}

#[test]
fn test_fox_clips_match_gltf2ozz() {
    let (document, buffers) = load_fox();
    let (built_skeleton, joint_nodes) = build_fox_skeleton(&document);
    let skeleton = Rc::new(read_skeleton(&built_skeleton.to_archive_bytes().unwrap()));

    let names: Vec<_> = document.animations().map(|a| a.name().unwrap().to_string()).collect();
    assert_eq!(names, ["Survey", "Walk", "Run"]);

    for (index, name) in names.iter().enumerate() {
        let raw = import_animation(&document, &buffers, index, &joint_nodes, 0.0).unwrap();
        let optimized = AnimationOptimizer::default().optimize(&raw, &skeleton).unwrap();
        let built = AnimationBuilder { iframe_interval: 10.0 }.build(&optimized).unwrap();

        // Written and read back through `Archive`.
        let written = built.to_archive_bytes().unwrap();
        let animation = read_animation(&written);

        let reference_bytes = std::fs::read(format!("./resource/fox/{}.ozz", name)).unwrap();
        let reference = read_animation(&reference_bytes);

        assert_eq!(animation.name(), reference.name());
        assert_eq!(animation.num_tracks(), reference.num_tracks());
        assert_eq!(animation.duration(), reference.duration());

        let counts = |a: &Animation| {
            (
                a.timepoints().len(),
                a.translations().len(),
                a.rotations().len(),
                a.scales().len(),
            )
        };
        println!(
            "{}: keys (timepoints, T, R, S) rust {:?}, gltf2ozz {:?}; byte-identical: {}",
            name,
            counts(&animation),
            counts(&reference),
            written == reference_bytes
        );

        let errors = compare_by_sampling(&skeleton, animation, reference);
        println!(
            "{}: max error over 1 ms samples: local translation {:e}, rotation {:e} rad, scale {:e}, model position {:e}",
            name, errors.translation, errors.rotation, errors.scale, errors.model_position
        );
        assert!(
            errors.translation <= MAX_LOCAL_TRANSLATION_ERROR,
            "{} {:?}",
            name,
            errors
        );
        assert!(errors.rotation <= MAX_ROTATION_ERROR_RAD, "{} {:?}", name, errors);
        assert!(errors.scale <= MAX_SCALE_ERROR, "{} {:?}", name, errors);
        assert!(
            errors.model_position <= MAX_MODEL_POSITION_ERROR,
            "{} {:?}",
            name,
            errors
        );

        // Unoptimized build: the optimizer's error stays within its tolerance's order.
        let unoptimized = read_animation(
            &AnimationBuilder { iframe_interval: 10.0 }
                .build(&raw)
                .unwrap()
                .to_archive_bytes()
                .unwrap(),
        );
        let errors = compare_by_sampling(&skeleton, unoptimized, read_animation(&written));
        println!(
            "{}: unoptimized vs optimized: local translation {:e}, rotation {:e} rad, model position {:e}",
            name, errors.translation, errors.rotation, errors.model_position
        );
    }
}
