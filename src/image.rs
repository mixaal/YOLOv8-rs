use tch::Tensor;

use crate::{utils, BBox};

// Image channels, height and width
pub type ImageCHW = (i64, i64, i64);

pub struct Image {
    width: i64,
    height: i64,
    pub(crate) image: Tensor,
    pub(crate) scaled_image: Tensor,
    pub(crate) image_dim: ImageCHW,
    pub(crate) scaled_image_dim: ImageCHW,
}

impl Image {
    fn from_tensor(image: Tensor, dimension: (i64, i64)) -> Self {
        let width = dimension.0;
        let height = dimension.1;

        let scaled_image = utils::preprocess(&image, dimension.0);
        let image_dim = image.size3().unwrap();
        let scaled_image_dim = scaled_image.size3().unwrap();
        Self {
            width,
            height,
            image,
            scaled_image,
            image_dim,
            scaled_image_dim,
        }
    }

    pub fn from_slice(
        slice: &[u8],
        orig_width: i64,
        orig_height: i64,
        dimension: (i64, i64),
    ) -> Self {
        let image = Tensor::from_slice(slice).view((3, orig_height, orig_width));
        Self::from_tensor(image, dimension)
    }

    pub fn new(path: &str, dimension: (i64, i64)) -> Self {
        let image = tch::vision::image::load(path).expect("can't load image");
        Self::from_tensor(image, dimension)
    }

    fn draw_line(t: &mut tch::Tensor, x1: i64, x2: i64, y1: i64, y2: i64) {
        let color = Tensor::from_slice(&[255., 255., 0.]).view([3, 1, 1]);
        t.narrow(2, x1, x2 - x1)
            .narrow(1, y1, y2 - y1)
            .copy_(&color)
    }

    pub fn draw_rectangle(&mut self, bboxes: &Vec<BBox>) {
        let image = &mut self.image;

        for bbox in bboxes.iter() {
            let xmin = bbox.xmin as i64;
            let ymin = bbox.ymin as i64;
            let xmax = bbox.xmax as i64;
            let ymax = bbox.ymax as i64;
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
