use criterion::{black_box, criterion_group, criterion_main, Criterion};
use yolo_v8::{image::Image, YoloV8ObjectDetection, YoloV8Segmentation};

fn bench_segmentation_prediction(c: &mut Criterion) {
    let image = Image::new("images/bus.jpg", (640, 640));
    let yolo = YoloV8Segmentation::new();
    c.bench_function("bench_segmentation_prediction", |b| {
        b.iter(|| black_box(yolo.predict(black_box(&image), black_box(0.25), black_box(0.7))))
    });
}

fn bench_detection_prediction(c: &mut Criterion) {
    let image = Image::new("images/bus.jpg", (640, 640));
    let yolo = YoloV8ObjectDetection::new();
    c.bench_function("bench_detection_prediction", |b| {
        b.iter(|| black_box(yolo.predict(black_box(&image), black_box(0.25), black_box(0.7))))
    });
}

criterion_group!(
    benches,
    bench_segmentation_prediction,
    bench_detection_prediction
);
criterion_main!(benches);
