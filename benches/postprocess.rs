use criterion::{black_box, criterion_group, criterion_main, Criterion};
use yolo_v8::{image::Image, YoloV8ObjectDetection, YoloV8Segmentation};

fn bench_segmentation_postprocess(c: &mut Criterion) {
    let image = Image::new("images/bus.jpg", (640, 640)).expect("can't load image");
    let yolo = YoloV8Segmentation::new().expect("can't create yolo model");
    let result = yolo.predict(&image, 0.25, 0.7);
    c.bench_function("bench_segmentation_postprocess", |b| {
        b.iter(|| black_box(result.postprocess()))
    });
}

fn bench_detection_postprocess(c: &mut Criterion) {
    let image = Image::new("images/bus.jpg", (640, 640)).expect("can't load image");
    let yolo = YoloV8ObjectDetection::new().expect("can't create yolo model");
    let result = yolo.predict(&image, 0.25, 0.7);
    c.bench_function("bench_detection_postprocess", |b| {
        b.iter(|| black_box(result.postprocess()))
    });
}

criterion_group!(
    benches,
    bench_segmentation_postprocess,
    bench_detection_postprocess
);
criterion_main!(benches);
