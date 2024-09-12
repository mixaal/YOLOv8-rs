use tch::{IValue, Tensor};

use crate::{image::ImageCHW, BBox, SegBBox, SegmentationResult};

pub struct SegmentationTools {}
pub struct DetectionTools {}

impl SegmentationTools {
    pub fn postprocess(
        pred: &IValue,
        image_dim: ImageCHW,
        scaled_image_dim: ImageCHW,
        conf_threshold: f32,
        iou_threshold: f32,
        device: tch::Device,
    ) -> Vec<SegmentationResult> {
        let mut result = Vec::new();

        // https://github.com/ultralytics/ultralytics/issues/2953
        if let IValue::Tuple(iv) = pred {
            let mut segboxes = Vec::new();
            if let IValue::Tensor(bboxes) = &iv[0] {
                let t = bboxes.get(0);
                // println!("bboxes={:?}", t);
                segboxes = Self::non_max_suppression(
                    image_dim,
                    scaled_image_dim,
                    &t,
                    conf_threshold,
                    iou_threshold,
                );
                // println!("r={:?}", segboxes);
            }

            if let IValue::Tensor(seg) = &iv[1] {
                for segbox in segboxes {
                    let weights = Tensor::from_slice(&segbox.cls_weight)
                        .reshape([1, 32])
                        .to_device(device);
                    // println!("weights={:?}", weights);

                    let t = seg.get(0).reshape([32, 160 * 160]);
                    // println!("seg={:?}", t);
                    let mask = weights.matmul(&t).reshape([1, 160, 160]).gt_(0.0);
                    // println!("r={}", mask);
                    result.push(SegmentationResult { segbox, mask });
                }
            }
        }
        result
    }

    fn non_max_suppression(
        image_dim: ImageCHW,
        scaled_image_dim: ImageCHW,
        prediction: &tch::Tensor,
        conf_thresh: f32,
        iou_thresh: f32,
    ) -> Vec<SegBBox> {
        let prediction = prediction.transpose(1, 0);
        let (anchors, classes_no) = prediction.size2().unwrap();

        let nclasses = (classes_no - 4) as usize;
        // println!("classes_no={classes_no}, anchors={anchors}");

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

                    let (_, orig_h, orig_w) = image_dim;
                    let (_, sh, sw) = scaled_image_dim;
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
                        name: crate::classes::DETECT_CLASSES[class_index],
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
                    let iou = Self::iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);

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

    //FIXME !!!
    fn iou(b1: &SegBBox, b2: &SegBBox) -> f32 {
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

impl DetectionTools {
    pub(crate) fn non_max_suppression(
        image_dim: ImageCHW,
        scaled_image_dim: ImageCHW,
        prediction: &tch::Tensor,
        conf_thresh: f64,
        iou_thresh: f64,
    ) -> Vec<BBox> {
        let prediction = prediction.get(0);
        let prediction = prediction.transpose(1, 0);
        let (anchors, classes_no) = prediction.size2().unwrap();

        let initial_w = image_dim.2 as f64;
        let initial_h = image_dim.1 as f64;
        let w_ratio = initial_w / scaled_image_dim.2 as f64;
        let h_ratio = initial_h / scaled_image_dim.1 as f64;

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

                    let (_, orig_h, orig_w) = image_dim;
                    let (_, sh, sw) = scaled_image_dim;
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

                    let xmin = ((x - w / 2.) * w_ratio).clamp(0.0, initial_w - 1.0);
                    let ymin = ((y - h / 2.) * h_ratio).clamp(0.0, initial_h - 1.0);
                    let xmax = ((x + w / 2.) * w_ratio).clamp(0.0, initial_w - 1.0);
                    let ymax = ((y + h / 2.) * h_ratio).clamp(0.0, initial_h - 1.0);

                    let bbox = BBox {
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        conf: confidence,
                        cls: class_index,
                        name: crate::classes::DETECT_CLASSES[class_index],
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
                    let iou = Self::iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);

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
}
// global image preprocessing:
//  1) resize, keep aspect ratio
//  2) padding to square tensor
pub fn preprocess(image: &Tensor, square_size: i64) -> Tensor {
    let (_, height, width) = image.size3().unwrap();
    let (uw, uh) = square64(square_size, width, height);
    let scaled_image = tch::vision::image::resize(&image, uw, uh).expect("can't resize image");

    let gray: Vec<u8> = vec![114; (square_size * square_size * 3) as usize];
    let bg = Tensor::from_slice(&gray).reshape([3, square_size, square_size]);
    let dh = (square_size - uh) / 2;
    let dw = (square_size - uw) / 2;

    bg.narrow(2, dw, uw).narrow(1, dh, uh).copy_(&scaled_image);
    bg
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

#[cfg(test)]
mod test {
    use super::preprocess;

    #[test]
    fn tensor_padding() {
        let image = tch::vision::image::load("images/bus.jpg").expect("can't load image");
        let t = preprocess(&image, 640);
        println!("t={:?}", t);
        let r = t.size3();
        assert!(r.is_ok());
        let (ch, h, w) = r.unwrap();
        assert_eq!(3, ch);
        assert_eq!(640, h);
        assert_eq!(640, w);
        tch::vision::image::save(&t, "bus_padded.jpg").expect("can't save image");

        let image = tch::vision::image::load("images/katri.jpg").expect("can't load image");

        let t = preprocess(&image, 640);
        println!("t={:?}", t);
        let r = t.size3();
        assert!(r.is_ok());
        let (ch, h, w) = r.unwrap();
        assert_eq!(3, ch);
        assert_eq!(640, h);
        assert_eq!(640, w);
        tch::vision::image::save(&t, "katri_padded.jpg").expect("can't save image");
    }
}
