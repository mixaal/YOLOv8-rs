use criterion::{black_box, criterion_group, criterion_main, Criterion};
use yolo_v8::utils::preprocess;

fn bench_tensor_preprocess(c: &mut Criterion) {
    let image = tch::vision::image::load("images/bus.jpg").expect("can't load image");
    c.bench_function("bench_tensor_preprocess", |b| {
        b.iter(|| preprocess(black_box(&image), black_box(640)))
    });
}

criterion_group!(benches, bench_tensor_preprocess);
criterion_main!(benches);
