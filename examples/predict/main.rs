use std::time::Instant;

use tch::{TchError, Tensor};
use yolo_v8::{image::Image, YoloV8Classifier, YoloV8ObjectDetection, YoloV8Segmentation};

fn object_detection(path: &str) {
    // Load image to perform object detection, note that YOLOv8 resolution must match
    // scaling width and height here
    let mut timings = vec![];
    let start = Instant::now();
    let mut image = Image::new(path, YoloV8ObjectDetection::input_dimension());
    timings.push(("load image", start.elapsed()));

    let start = Instant::now();
    // Load exported torchscript for object detection
    let yolo = YoloV8ObjectDetection::new().post_process_on_cpu();
    timings.push(("load model", start.elapsed()));

    let start = Instant::now();
    // Predict with non-max-suppression in the end
    let prediction = yolo.predict(&image, 0.25, 0.7);
    timings.push(("prediction", start.elapsed()));

    let start = Instant::now();
    // extract bboxes from prediction in post-processing
    let bboxes = prediction.postprocess();
    timings.push(("post-process", start.elapsed()));
    println!("bboxes={:?}", bboxes);

    let start = Instant::now();
    // Draw rectangles around detected objects
    image.draw_rectangle(&bboxes.0);
    timings.push(("draw rectangles", start.elapsed()));

    let start = Instant::now();
    // Finally save the result
    image.save("images/result2.jpg");
    timings.push(("save result", start.elapsed()));

    println!("timings:{:?}", timings);
}

fn image_classification(path: &str) {
    // Load image to perform image classification
    let image = Image::new(path, YoloV8Classifier::input_dimension());

    // Load exported torchscript for object detection
    let yolo = YoloV8Classifier::new();

    let classes = yolo.predict(&image);
    println!("classes={:?}", classes);
}

fn image_segmentation(path: &str) {
    let image = Image::new(path, YoloV8Segmentation::input_dimension());

    // Load exported torchscript for object detection
    let yolo = YoloV8Segmentation::new();

    let segmentation = yolo.predict(&image, 0.25, 0.7).postprocess();
    println!("segmentation={:?}", segmentation);
    let mut mask_no = 0;
    for seg in segmentation {
        let mask = seg.mask.reshape([-1]);
        let name = seg.segbox.name;
        let mut rgb = Vec::new();
        let mut vec = Vec::<f64>::try_from(&mask).unwrap();
        rgb.append(&mut vec.clone());
        rgb.append(&mut vec.clone());
        rgb.append(&mut vec);
        let im = Tensor::from_slice(&rgb)
            .reshape([3, 160, 160])
            .g_mul_scalar(255.);
        let imgname = format!("mask-{name}-{mask_no}.jpg");
        tch::vision::image::save(&im, imgname).expect("can't save image");
        mask_no += 1;
    }
}

// YOLOv8n for object detection in image
fn main() -> Result<(), TchError> {
    object_detection("images/bus.jpg");
    image_classification("images/bus.jpg");
    image_segmentation("images/test.jpg");
    Ok(())
}
