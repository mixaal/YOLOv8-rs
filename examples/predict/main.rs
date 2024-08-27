use tch::TchError;
use yolo_v8::{Image, YOLOv8};

// YOLOv8n (nano model) for object detection in image
fn main() -> Result<(), TchError> {
    // Load image to perform object detection, note that YOLOv8 resolution must match
    // scaling width and height here
    let mut image = Image::new("images/test.jpg", 640, 640);

    // Load exported torchscript for object detection
    let yolo = YOLOv8::new("models/yolov8n.torchscript")?;

    // Predict with non-max-suppression in the end
    let bboxes = yolo.predict(&image, 0.25, 0.35);
    println!("bboxes={:?}", bboxes);

    // Draw rectangles around detected objects
    image.draw_rectangle(&bboxes);
    // Finally save the result
    image.save("images/result2.jpg");
    Ok(())
}
