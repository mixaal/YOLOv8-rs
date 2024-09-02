// Original rusty-yolo:
//   https://github.com/igor-yusupov/rusty-yolo/
//
// YOLOv8 article:
//   https://linzichun.com/posts/rust-opencv-onnx-yolov8-detect/

pub mod utils;

use classes::DETECT_CLASSES;
use tch::{IValue, Tensor};

pub(crate) mod classes;

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
    pub fn new() -> Self {
        Self {
            yolo: YOLOv8::new("models/yolov8n-cls.torchscript").expect("can't load model"),
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
    pub fn new() -> Self {
        Self {
            yolo: YOLOv8::new("models/yolov8n.torchscript").expect("can't load model"),
        }
    }

    pub fn input_dimension() -> (i64, i64) {
        (640, 640)
    }

    pub fn predict(&self, image: &Image, conf_thresh: f64, iou_thresh: f64) -> Vec<BBox> {
        // println!("predict(): image={:?}", image.scaled_image);
        let pred = self.yolo.predict(image);
        // println!("pred={:?}", pred);
        self.non_max_suppression(image, &pred.get(0), conf_thresh, iou_thresh)
    }

    fn non_max_suppression(
        &self,
        image: &Image,
        prediction: &tch::Tensor,
        conf_thresh: f64,
        iou_thresh: f64,
    ) -> Vec<BBox> {
        let prediction = prediction.transpose(1, 0);
        let (anchors, classes_no) = prediction.size2().unwrap();

        let nclasses = (classes_no - 4) as usize;
        // println!("classes_no={classes_no}, anchors={anchors}");

        let mut bboxes: Vec<Vec<BBox>> = (0..nclasses).map(|_| vec![]).collect();

        for index in 0..anchors {
            let pred = Vec::<f64>::try_from(prediction.get(index)).expect("wrong type of tensor");

            // println!("index={index}, pred={}", pred.len());

            for i in 4..classes_no as usize {
                let confidence = pred[i];
                if confidence > conf_thresh {
                    let class_index = i - 4;
                    // println!(
                    //     "confidence={confidence}, class_index={class_index} class_name={}",
                    //     CLASSES[class_index]
                    // );

                    let (_, orig_h, orig_w) = image.image.size3().unwrap();
                    let (_, sh, sw) = image.scaled_image.size3().unwrap();
                    let cx = sw as f64 / 2.0;
                    let cy = sh as f64 / 2.0;
                    let mut dx = pred[0] - cx;
                    let mut dy = pred[1] - cy;
                    let mut w = pred[2];
                    let mut h = pred[3];

                    let aspect = orig_w as f64 / orig_h as f64;

                    if orig_w > orig_h {
                        dy *= aspect;
                        h *= aspect;
                    } else {
                        dx /= aspect;
                        w /= aspect;
                    }

                    let x = cx + dx;
                    let y = cy + dy;

                    let bbox = BBox {
                        xmin: x - w / 2.,
                        ymin: y - h / 2.,
                        xmax: x + w / 2.,
                        ymax: y + h / 2.,
                        conf: confidence,
                        cls: class_index,
                        name: DETECT_CLASSES[class_index],
                    };
                    bboxes[class_index].push(bbox)
                }
            }
        }

        for bboxes_for_class in bboxes.iter_mut() {
            bboxes_for_class.sort_by(|b1, b2| b2.conf.partial_cmp(&b1.conf).unwrap());

            let mut current_index = 0;
            for index in 0..bboxes_for_class.len() {
                let mut drop = false;
                for prev_index in 0..current_index {
                    let iou = YOLOv8::iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);

                    if iou > iou_thresh {
                        drop = true;
                        break;
                    }
                }
                if !drop {
                    bboxes_for_class.swap(current_index, index);
                    current_index += 1;
                }
            }
            bboxes_for_class.truncate(current_index);
        }

        let mut result = vec![];

        for bboxes_for_class in bboxes.iter() {
            for bbox in bboxes_for_class.iter() {
                result.push(*bbox);
            }
        }

        return result;
    }
}

pub struct YoloV8Segmentation {
    yolo: YOLOv8,
}

impl YoloV8Segmentation {
    pub fn new() -> Self {
        Self {
            yolo: YOLOv8::new("models/yolov8n-seg.torchscript").expect("can't load model"),
        }
    }

    pub fn predict(
        &self,
        image: &Image,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Vec<SegmentationResult> {
        let img = &image.scaled_image;
        let mut result = Vec::new();

        // println!("img={:?}", img);

        let img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.yolo.device)
            .g_div_scalar(255.);

        let t = tch::IValue::Tensor(img);
        let pred = self.yolo.model.forward_is(&[t]).unwrap();
        println!("pred={:?}", pred);
        // https://github.com/ultralytics/ultralytics/issues/2953
        if let IValue::Tuple(iv) = pred {
            let mut segboxes = Vec::new();
            if let IValue::Tensor(bboxes) = &iv[0] {
                let t = bboxes.get(0);
                println!("bboxes={:?}", t);
                segboxes = self.non_max_suppression(image, &t, conf_threshold, iou_threshold);
                println!("r={:?}", segboxes);
            }

            if let IValue::Tensor(seg) = &iv[1] {
                for segbox in segboxes {
                    let weights = Tensor::from_slice(&segbox.cls_weight).reshape([1, 32]);
                    println!("weights={:?}", weights);

                    let t = seg.get(0).reshape([32, 160 * 160]);
                    println!("seg={:?}", t);
                    let mask = weights.matmul(&t).reshape([1, 160, 160]).gt_(0.0);
                    println!("r={}", mask);
                    result.push(SegmentationResult { segbox, mask });
                }
            }
        }
        result
    }

    fn non_max_suppression(
        &self,
        image: &Image,
        prediction: &tch::Tensor,
        conf_thresh: f32,
        iou_thresh: f32,
    ) -> Vec<SegBBox> {
        let prediction = prediction.transpose(1, 0);
        let (anchors, classes_no) = prediction.size2().unwrap();

        let nclasses = (classes_no - 4) as usize;
        println!("classes_no={classes_no}, anchors={anchors}");

        let mut bboxes: Vec<Vec<SegBBox>> = (0..nclasses).map(|_| vec![]).collect();

        for index in 0..anchors {
            let pred = Vec::<f32>::try_from(prediction.get(index)).expect("wrong type of tensor");

            // println!("index={index}, pred={}", pred.len());

            //FIXME
            let weights: [f32; 32] = pred[84..116].try_into().expect("cccc");

            for i in 4..84 as usize {
                let confidence = pred[i];
                if confidence > conf_thresh {
                    let class_index = i - 4;
                    // println!(
                    //     "confidence={confidence}, class_index={class_index} class_name={}",
                    //     CLASSES[class_index]
                    // );

                    let (_, orig_h, orig_w) = image.image.size3().unwrap();
                    let (_, sh, sw) = image.scaled_image.size3().unwrap();
                    let cx = sw as f32 / 2.0;
                    let cy = sh as f32 / 2.0;
                    let mut dx = pred[0] - cx;
                    let mut dy = pred[1] - cy;
                    let mut w = pred[2];
                    let mut h = pred[3];

                    let aspect = orig_w as f32 / orig_h as f32;

                    if orig_w > orig_h {
                        dy *= aspect;
                        h *= aspect;
                    } else {
                        dx /= aspect;
                        w /= aspect;
                    }

                    let x = cx + dx;
                    let y = cy + dy;

                    let bbox = SegBBox {
                        xmin: x - w / 2.,
                        ymin: y - h / 2.,
                        xmax: x + w / 2.,
                        ymax: y + h / 2.,
                        conf: confidence,
                        cls: class_index,
                        name: DETECT_CLASSES[class_index],
                        cls_weight: weights,
                    };
                    bboxes[class_index].push(bbox)
                }
            }
        }

        for bboxes_for_class in bboxes.iter_mut() {
            bboxes_for_class.sort_by(|b1, b2| b2.conf.partial_cmp(&b1.conf).unwrap());

            let mut current_index = 0;
            for index in 0..bboxes_for_class.len() {
                let mut drop = false;
                for prev_index in 0..current_index {
                    let iou =
                        YOLOv8::iou_seg(&bboxes_for_class[prev_index], &bboxes_for_class[index]);

                    if iou > iou_thresh {
                        drop = true;
                        break;
                    }
                }
                if !drop {
                    bboxes_for_class.swap(current_index, index);
                    current_index += 1;
                }
            }
            bboxes_for_class.truncate(current_index);
        }

        let mut result = vec![];

        for bboxes_for_class in bboxes.iter() {
            for bbox in bboxes_for_class.iter() {
                result.push(*bbox);
            }
        }

        return result;
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

        pred
    }

    fn iou(b1: &BBox, b2: &BBox) -> f64 {
        let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
        let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
        let i_xmin = b1.xmin.max(b2.xmin);
        let i_xmax = b1.xmax.min(b2.xmax);
        let i_ymin = b1.ymin.max(b2.ymin);
        let i_ymax = b1.ymax.min(b2.ymax);
        let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
        i_area / (b1_area + b2_area - i_area)
    }

    //FIXME !!!
    fn iou_seg(b1: &SegBBox, b2: &SegBBox) -> f32 {
        let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
        let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
        let i_xmin = b1.xmin.max(b2.xmin);
        let i_xmax = b1.xmax.min(b2.xmax);
        let i_ymin = b1.ymin.max(b2.ymin);
        let i_ymax = b1.ymax.min(b2.ymax);
        let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
        i_area / (b1_area + b2_area - i_area)
    }
}

pub struct Image {
    width: i64,
    height: i64,
    image: Tensor,
    scaled_image: Tensor,
}

impl Image {
    pub fn from_slice(
        slice: &[u8],
        orig_width: i64,
        orig_height: i64,
        width: i64,
        height: i64,
    ) -> Self {
        let image = Tensor::from_slice(slice).view((3, orig_height, orig_width));
        println!("image={:?}", image);
        let scaled_image =
            tch::vision::image::resize(&image, width, height).expect("can't resize image");
        Self {
            width,
            height,
            image,
            scaled_image,
        }
    }

    pub fn new(path: &str, dimension: (i64, i64)) -> Self {
        let width = dimension.0;
        let height = dimension.1;
        let image = tch::vision::image::load(path).expect("can't load image");
        let scaled_image = utils::preprocess(&image, dimension.0);
        Self {
            width,
            height,
            image,
            scaled_image,
        }
    }

    fn draw_line(t: &mut tch::Tensor, x1: i64, x2: i64, y1: i64, y2: i64) {
        let color = Tensor::from_slice(&[255., 255., 0.]).view([3, 1, 1]);
        t.narrow(2, x1, x2 - x1)
            .narrow(1, y1, y2 - y1)
            .copy_(&color)
    }

    pub fn draw_rectangle(&mut self, bboxes: &Vec<BBox>) {
        let image = &mut self.image;
        let (_, initial_h, initial_w) = image.size3().expect("can't get image size");
        let w_ratio = initial_w as f64 / self.width as f64;
        let h_ratio = initial_h as f64 / self.height as f64;

        for bbox in bboxes.iter() {
            let xmin = ((bbox.xmin * w_ratio) as i64).clamp(0, initial_w - 1);
            let ymin = ((bbox.ymin * h_ratio) as i64).clamp(0, initial_h - 1);
            let xmax = ((bbox.xmax * w_ratio) as i64).clamp(0, initial_w - 1);
            let ymax = ((bbox.ymax * h_ratio) as i64).clamp(0, initial_h - 1);
            Self::draw_line(image, xmin, xmax, ymin, ymax.min(ymin + 2));
            Self::draw_line(image, xmin, xmax, ymin.max(ymax - 2), ymax);
            Self::draw_line(image, xmin, xmax.min(xmin + 2), ymin, ymax);
            Self::draw_line(image, xmin.max(xmax - 2), xmax, ymin, ymax);
        }
    }

    pub fn save(&self, path: &str) {
        tch::vision::image::save(&self.image, path).expect("can't save image");
    }
}
