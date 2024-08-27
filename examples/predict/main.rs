use tch::TchError;
use yolo_v8::{Image, YOLOv8};

fn main() -> Result<(), TchError> {
    let mut image = Image::new("images/test.jpg", 640, 640);
    let yolo = YOLOv8::new("models/yolov8n.torchscript")?;
    let bboxes = yolo.predict(&image, 0.25, 0.35);
    println!("bboxes={:?}", bboxes);
    image.draw_rectangle(&bboxes);
    image.save("images/result2.jpg");
    Ok(())
}
