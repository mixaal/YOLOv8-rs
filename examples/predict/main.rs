use tch::TchError;
use yolo_v8::{Image, YoloV8Classifier, YoloV8ObjectDetection, YoloV8Segmentation};

fn object_detection() {
    // Load image to perform object detection, note that YOLOv8 resolution must match
    // scaling width and height here
    let mut image = Image::new("images/bus.jpg", YoloV8ObjectDetection::input_dimension());

    // Load exported torchscript for object detection
    let yolo = YoloV8ObjectDetection::new();

    // Predict with non-max-suppression in the end
    let bboxes = yolo.predict(&image, 0.15, 0.35);
    println!("bboxes={:?}", bboxes);

    // Draw rectangles around detected objects
    image.draw_rectangle(&bboxes);
    // Finally save the result
    image.save("images/result2.jpg");
}

fn image_classification() {
    // Load image to perform image classification
    let image = Image::new("images/test.jpg", YoloV8Classifier::input_dimension());

    // Load exported torchscript for object detection
    let yolo = YoloV8Classifier::new();

    let classes = yolo.predict(&image);
    println!("classes={:?}", classes);
}

fn image_segmentation() {
    let image = Image::new("images/test.jpg", YoloV8Segmentation::input_dimension());

    // Load exported torchscript for object detection
    let yolo = YoloV8Segmentation::new();

    let classes = yolo.predict(&image);
}

// YOLOv8n (nano model) for object detection in image
fn main() -> Result<(), TchError> {
    object_detection();
    // image_classification();
    // image_segmentation();
    Ok(())
}
