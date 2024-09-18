use std::{env::args, time::Instant};

use opencv::{
    core::{Mat, MatTrait, MatTraitConst, Rect, Scalar, Vector},
    highgui::{destroy_all_windows, imshow, wait_key},
    imgcodecs::imwrite,
    imgproc::{rectangle, LINE_8},
    videoio::{VideoCaptureTrait, CAP_ANY},
};
use tch::{Kind, Tensor};
use yolo_v8::image::Image;

fn get_model(arg_model: String) -> yolo_v8::YOLOModel {
    match arg_model.to_lowercase().as_str() {
        "nano" => yolo_v8::YOLOModel::Nano,
        "small" => yolo_v8::YOLOModel::Small,
        "medium" => yolo_v8::YOLOModel::Medium,
        "large" => yolo_v8::YOLOModel::Large,
        "extra" => yolo_v8::YOLOModel::Extra,
        _ => panic!("YOLO model can be: nano, small, medium, large or extra"),
    }
}

fn main() -> Result<(), opencv::Error> {
    let filename = args().nth(1).unwrap_or("test3.mp4".to_owned());
    let model = args().nth(2).unwrap_or("nano".to_owned());
    println!("filename={filename} model={model}");
    let mut cap = opencv::videoio::VideoCapture::from_file(&filename, CAP_ANY)?;
    let yolo = yolo_v8::YoloV8ObjectDetection::with_model(get_model(model)); //.post_process_on_cpu();
    let device = tch::Device::cuda_if_available();
    println!("device: {:?}", device);

    loop {
        let mut timings = Vec::new();
        let mut frame = Mat::default();
        let start = Instant::now();
        let have_image = cap.read(&mut frame)?;
        if !have_image {
            break;
        }

        let mut image = Image::from_opencv_mat(&frame, (640, 640))?;
        timings.push(("read_frame", start.elapsed()));
        let start = Instant::now();
        let predictions = yolo.predict(&image, 0.25, 0.7); //.postprocess();
        timings.push(("detection", start.elapsed()));
        // image.draw_rectangle(&predictions);
        // image.save("result.jpg");

        let start = Instant::now();
        let predictions = predictions.postprocess();
        timings.push(("postprocess", start.elapsed()));
        let start = Instant::now();
        for bbox in predictions.0 {
            let w = bbox.xmax - bbox.xmin;
            let h = bbox.ymax - bbox.ymin;
            let class = format!("{} {}%", bbox.name, (bbox.conf * 100.0) as i32);
            let _ = opencv::imgproc::put_text(
                &mut frame,
                &class,
                (bbox.xmin as i32, bbox.ymin as i32).into(),
                0,
                1.0,
                Scalar::new(255.0, 255.0, 255.0, 255.0),
                1,
                LINE_8,
                false,
            );
            rectangle(
                &mut frame,
                Rect::new(bbox.xmin as i32, bbox.ymin as i32, w as i32, h as i32),
                Scalar::new(255.0, 128.0, 0.0, 255.0),
                2,
                1,
                0,
            )?;
        }

        imshow("Image", &frame)?;
        let key = wait_key(1)?;
        if key > 0 && key != 255 {
            break;
        }
        timings.push(("draw_boxes", start.elapsed()));
        println!("timings:{:?}", timings);
    }
    cap.release()?;
    destroy_all_windows()?;
    Ok(())
}

fn square64(size: i64, w: i64, h: i64) -> (i64, i64) {
    let aspect = w as f32 / h as f32;
    if w > h {
        let tw = size;
        let th = (tw as f32 / aspect) as i64;
        (tw, th)
    } else {
        let th = size;
        let tw = (size as f32 * aspect) as i64;
        (tw, th)
    }
}
