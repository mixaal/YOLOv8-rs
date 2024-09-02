use criterion::{black_box, criterion_group, criterion_main, Criterion};
use yolo_v8::{image::Image, YoloV8ObjectDetection, YoloV8Segmentation};

fn bench_segmentation_e2e(c: &mut Criterion) {
    c.bench_function("bench_segmentation_e2e", |b| {
        b.iter(|| {
            let image = Image::new(black_box("images/bus.jpg"), black_box((640, 640)));
            let yolo = YoloV8Segmentation::new();
            let result = yolo.predict(black_box(&image), black_box(0.25), black_box(0.7));
            black_box(result.postprocess())
        })
    });
}

fn bench_detection_e2e(c: &mut Criterion) {
    c.bench_function("bench_detection_e2e", |b| {
        b.iter(|| {
            let image = Image::new(black_box("images/bus.jpg"), black_box((640, 640)));
            let yolo = YoloV8ObjectDetection::new();
            let result = yolo.predict(black_box(&image), black_box(0.25), black_box(0.7));
            black_box(result.postprocess())
        })
    });
}

criterion_group!(benches, bench_segmentation_e2e, bench_detection_e2e);
criterion_main!(benches);
