use tch::Tensor;

use crate::{utils, BBox};

// Image channels, height and width
pub type ImageCHW = (i64, i64, i64);

pub struct Image {
    width: i64,
    height: i64,
    image: Option<Tensor>,
    pub(crate) scaled_image: Tensor,
    pub(crate) image_dim: ImageCHW,
    pub(crate) scaled_image_dim: ImageCHW,
}

impl Image {
    pub fn from_tensor(image: Tensor, dimension: (i64, i64)) -> Self {
        let width = dimension.0;
        let height = dimension.1;

        let scaled_image = utils::preprocess(&image, dimension.0);
        let image_dim = image.size3().unwrap();
        let scaled_image_dim = scaled_image.size3().unwrap();
        Self {
            width,
            height,
            image: Some(image),
            scaled_image,
            image_dim,
            scaled_image_dim,
        }
    }

    #[cfg(feature = "opencv")]
    pub fn from_opencv_mat(
        src_frame: &opencv::core::Mat,
        dimension: (i64, i64),
    ) -> Result<Self, opencv::Error> {
        let width = dimension.0;
        let height = dimension.1;
        let square_size = width;
        let size = opencv::core::MatTraitConst::size(src_frame)?;
        let uh = size.height as i64;
        let uw = size.width as i64;
        let image_dim = (3 as i64, uh, uw);
        let (sw, sh) = utils::square64(width, size.width.into(), size.height.into());
        let mut frame = opencv::core::Mat::default();
        // opencv resize is much faster than tch::resize
        opencv::imgproc::resize(
            src_frame,
            &mut frame,
            (sw as i32, sh as i32).into(),
            0.0,
            0.0,
            0,
        )?;

        let size = opencv::core::MatTraitConst::size(&frame)?;
        let scaled_image = unsafe {
            Tensor::from_blob(
                opencv::core::MatTraitConst::data(&frame),
                &[
                    size.height as i64,
                    size.width as i64,
                    opencv::core::MatTraitConst::channels(&frame) as i64,
                ],
                &[],
                tch::Kind::Uint8,
                tch::Device::Cpu,
            )
        };

        let scaled_image_dim = (3 as i64, width, width);

        let scaled_image = scaled_image
            .permute([2, 0, 1]) // swap [[b0, g0, r0], [b1, g1, r1], ...] array to [[b0, b1, ...], [g0, g1, ...], [r0, r1, ..]]
            .flip(0); // swap [[B], [G], [R]] to [[R], [G], B]

        let gray: Vec<u8> = vec![114; (square_size * square_size * 3) as usize];
        let bg = Tensor::from_slice(&gray).reshape([3, square_size, square_size]);
        let dh = (square_size - sh) / 2;
        let dw = (square_size - sw) / 2;

        bg.narrow(2, dw, sw).narrow(1, dh, sh).copy_(&scaled_image);

        Ok(Self {
            width,
            height,
            image: None,
            scaled_image: bg,
            image_dim,
            scaled_image_dim,
        })
    }

    #[cfg(feature = "opencv")]
    pub fn from_slice(
        slice: &[u8],
        orig_width: i64,
        orig_height: i64,
        dimension: (i64, i64),
    ) -> Result<Self, opencv::Error> {
        let cv_mat = opencv::core::Mat::new_rows_cols_with_bytes::<rgb::RGB8>(
            orig_height as i32,
            orig_width as i32,
            slice,
        )?
        .clone_pointee();
        Self::from_opencv_mat(&cv_mat, dimension)
    }

    #[cfg(not(feature = "opencv"))]
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
        if let Some(ref mut image) = &mut self.image {
            // let image = &mut self.image;

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
    }

    pub fn save(&self, path: &str) {
        if let Some(ref image) = self.image {
            tch::vision::image::save(image, path).expect("can't save image");
        }
    }
}
