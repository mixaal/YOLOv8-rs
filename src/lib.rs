// Original rusty-yolo:
//   https://github.com/igor-yusupov/rusty-yolo/
//
// YOLOv8 article:
//   https://linzichun.com/posts/rust-opencv-onnx-yolov8-detect/

pub mod image;
pub mod utils;

use image::{Image, ImageCHW};
use tch::{IValue, Tensor};
use utils::{get_model, DetectionTools, SegmentationTools};

pub(crate) mod classes;

pub enum YOLOModel {
    Nano,
    Small,
    Medium,
    Large,
    Extra,
}

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
    pub conf: f64,
    pub cls: usize,
    pub name: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct SegBBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub conf: f32,
    pub cls: usize,
    pub cls_weight: [f32; 32],
    pub name: &'static str,
}

pub struct SegmentationPrediction {
    pred: IValue,
    image_dim: ImageCHW,
    scaled_image_dim: ImageCHW,
    conf_threshold: f32,
    iou_threshold: f32,
    device: tch::Device,
}

impl SegmentationPrediction {
    pub fn postprocess(&self) -> Vec<SegmentationResult> {
        SegmentationTools::postprocess(
            &self.pred,
            self.image_dim,
            self.scaled_image_dim,
            self.conf_threshold,
            self.iou_threshold,
            self.device,
        )
    }
}

pub struct ObjectDetectionPrediction {
    pred: Tensor,
    image_dim: ImageCHW,
    scaled_image_dim: ImageCHW,
    conf_threshold: f64,
    iou_threshold: f64,
}

impl ObjectDetectionPrediction {
    pub fn postprocess(&self) -> Vec<BBox> {
        DetectionTools::non_max_suppression(
            self.image_dim,
            self.scaled_image_dim,
            &self.pred,
            self.conf_threshold,
            self.iou_threshold,
        )
    }
}

#[derive(Debug)]
pub struct SegmentationResult {
    pub segbox: SegBBox,
    pub mask: Tensor,
}

#[derive(Debug)]
pub struct ClassConfidence {
    pub name: &'static str,
    pub conf: f64,
}

impl ClassConfidence {
    fn new(idx: usize, conf: f64) -> Self {
        Self {
            name: classes::CLASSES[idx],
            conf,
        }
    }
}

pub struct YoloV8Classifier {
    yolo: YOLOv8,
}

impl YoloV8Classifier {
    pub fn with_model(model_type: YOLOModel) -> Self {
        Self {
            yolo: YOLOv8::new(&get_model(model_type, utils::YOLOSpec::Classification))
                .expect("can't load model"),
        }
    }

    pub fn new() -> Self {
        Self {
            yolo: YOLOv8::new(&get_model(YOLOModel::Nano, utils::YOLOSpec::Classification))
                .expect("can't load model"),
        }
    }

    pub fn predict(&self, image: &Image) -> Vec<ClassConfidence> {
        let t = self.yolo.predict(image);
        Self::top_n(t, 5)
    }

    fn top_n(t: Tensor, n: usize) -> Vec<ClassConfidence> {
        let v = Vec::<f64>::try_from(t.get(0)).expect("no classification tensor");

        let mut top_val = vec![0.0; n];
        let mut top_idx = vec![0; n];

        for (idx, conf) in v.iter().enumerate() {
            for i in 0..n {
                if (i == 0 && *conf > top_val[0])
                    || (i > 0 && *conf > top_val[i] && *conf < top_val[i - 1])
                {
                    top_val[i] = *conf;
                    top_idx[i] = idx;
                }
            }
        }

        let mut r = Vec::new();
        for i in 0..n {
            r.push(ClassConfidence::new(top_idx[i], top_val[i]));
        }
        r
    }

    pub fn input_dimension() -> (i64, i64) {
        (224, 224)
    }
}

pub struct YoloV8ObjectDetection {
    yolo: YOLOv8,
}

impl YoloV8ObjectDetection {
    pub fn with_model(model_type: YOLOModel) -> Self {
        Self {
            yolo: YOLOv8::new(&get_model(model_type, utils::YOLOSpec::ObjectDetection))
                .expect("can't load model"),
        }
    }

    pub fn new() -> Self {
        Self {
            yolo: YOLOv8::new(&get_model(
                YOLOModel::Nano,
                utils::YOLOSpec::ObjectDetection,
            ))
            .expect("can't load model"),
        }
    }

    pub fn input_dimension() -> (i64, i64) {
        (640, 640)
    }

    pub fn predict(
        &self,
        image: &Image,
        conf_threshold: f64,
        iou_threshold: f64,
    ) -> ObjectDetectionPrediction {
        let pred = self.yolo.predict(image);

        ObjectDetectionPrediction {
            image_dim: image.image_dim,
            scaled_image_dim: image.scaled_image_dim,
            pred,
            conf_threshold,
            iou_threshold,
        }
    }
}

pub struct YoloV8Segmentation {
    yolo: YOLOv8,
}

impl YoloV8Segmentation {
    pub fn with_model(model_type: YOLOModel) -> Self {
        Self {
            yolo: YOLOv8::new(&utils::get_model(model_type, utils::YOLOSpec::Segmentation))
                .expect("can't load model"),
        }
    }

    pub fn new() -> Self {
        Self {
            yolo: YOLOv8::new(&utils::get_model(
                YOLOModel::Nano,
                utils::YOLOSpec::Segmentation,
            ))
            .expect("can't load model"),
        }
    }

    pub fn predict(
        &self,
        image: &Image,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> SegmentationPrediction {
        let img = &image.scaled_image;

        // println!("img={:?}", img);

        let img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.yolo.device)
            .g_div_scalar(255.);

        let t = tch::IValue::Tensor(img);
        let pred = self.yolo.model.forward_is(&[t]).unwrap();
        // println!("pred={:?}", pred);

        SegmentationPrediction {
            pred,
            image_dim: image.image_dim,
            scaled_image_dim: image.scaled_image_dim,
            conf_threshold,
            iou_threshold,
            device: self.yolo.device,
        }
    }

    pub fn input_dimension() -> (i64, i64) {
        (640, 640)
    }
}

pub struct YOLOv8 {
    device: tch::Device,
    model: tch::CModule,
}

impl YOLOv8 {
    pub fn new(path: &str) -> Result<Self, tch::TchError> {
        let device = tch::Device::cuda_if_available();
        let model = tch::CModule::load_on_device(path, device)?;
        Ok(Self { device, model })
    }

    pub fn predict(&self, image: &Image) -> Tensor {
        let img = &image.scaled_image;

        let img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.);

        let pred = self
            .model
            .forward_ts(&[img])
            .unwrap()
            .to_device(tch::Device::Cpu);
        // .to_device(self.device);

        pred
    }
}

#[cfg(test)]
mod test {
    use crate::{image::Image, BBox, YoloV8ObjectDetection, YoloV8Segmentation};

    #[test]
    fn test_segmentation() {
        let image = Image::new("images/bus.jpg", YoloV8Segmentation::input_dimension());
        let yolo = YoloV8Segmentation::new();
        let segmentation = yolo.predict(&image, 0.25, 0.7).postprocess();
        assert_eq!(3, segmentation.len());
    }

    #[test]
    fn test_detection() {
        let image = Image::new("images/bus.jpg", YoloV8ObjectDetection::input_dimension());
        let yolo = YoloV8ObjectDetection::new();
        let detection = yolo.predict(&image, 0.25, 0.7).postprocess();
        println!("detection={:?}", detection);
        assert_eq!(3, detection.len());
        bbox_eq(
            BBox {
                xmin: 1028.787,
                ymin: 387.95,
                xmax: 1085.012,
                ymax: 477.393,
                conf: 0.5158080458641052,
                cls: 0,
                name: "person",
            },
            detection[0],
        );

        bbox_eq(
            BBox {
                xmin: 892.104,
                ymin: 352.016,
                xmax: 976.45,
                ymax: 457.242,
                conf: 0.4701675474643707,
                cls: 0,
                name: "person",
            },
            detection[1],
        );
        bbox_eq(
            BBox {
                xmin: 26.115,
                ymin: 139.375,
                xmax: 1139.667,
                ymax: 661.516,
                conf: 0.9144105911254883,
                cls: 5,
                name: "bus",
            },
            detection[2],
        );
    }

    fn bbox_eq(a: BBox, b: BBox) {
        assert_eq!(a.cls, b.cls);
        feq(a.conf, b.conf);
        assert_eq!(a.name, b.name);
        feq(a.xmin, b.xmin);
        feq(a.xmax, b.xmax);
        feq(a.ymin, b.ymin);
        feq(a.ymax, b.ymax);
    }

    fn feq(a: f64, b: f64) {
        let d = (a - b).abs();
        if d > 0.07 {
            println!("a={a} b={b} d={d}");
            assert!(false, "distance too big");
        } else {
            assert!(true, "distance ok");
        }
    }
}
