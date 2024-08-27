// Original rusty-yolo:
//   https://github.com/igor-yusupov/rusty-yolo/
//
// YOLOv8 article:
//   https://linzichun.com/posts/rust-opencv-onnx-yolov8-detect/

use tch::Tensor;

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

static CLASSES: [&'static str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

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

    pub fn predict(&self, image: &Image, conf_thresh: f64, iou_thresh: f64) -> Vec<BBox> {
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

        let result = self.non_max_suppression(&pred.get(0), conf_thresh, iou_thresh);

        result
    }

    fn iou(&self, b1: &BBox, b2: &BBox) -> f64 {
        let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
        let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
        let i_xmin = b1.xmin.max(b2.xmin);
        let i_xmax = b1.xmax.min(b2.xmax);
        let i_ymin = b1.ymin.max(b2.ymin);
        let i_ymax = b1.ymax.min(b2.ymax);
        let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
        i_area / (b1_area + b2_area - i_area)
    }

    fn non_max_suppression(
        &self,
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

                    let bbox = BBox {
                        xmin: pred[0] - pred[2] / 2.,
                        ymin: pred[1] - pred[3] / 2.,
                        xmax: pred[0] + pred[2] / 2.,
                        ymax: pred[1] + pred[3] / 2.,
                        conf: confidence,
                        cls: class_index,
                        name: CLASSES[class_index],
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
                    let iou = self.iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);

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

pub struct Image {
    width: i64,
    height: i64,
    image: Tensor,
    scaled_image: Tensor,
}

impl Image {
    pub fn new(path: &str, width: i64, height: i64) -> Self {
        let image = tch::vision::image::load(path).expect("can't load image");
        let scaled_image =
            tch::vision::image::resize(&image, width, height).expect("can't resize image");
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
