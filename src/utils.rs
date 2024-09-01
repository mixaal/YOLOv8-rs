use opencv::{
    core::{copy_make_border, Vector, BORDER_CONSTANT},
    imgcodecs::imwrite,
    prelude::*,
    Error,
};
use tch::Tensor;

pub fn plain_resize(image: &str) -> Result<Tensor, Error> {
    let img = opencv::imgcodecs::imread(image, opencv::imgcodecs::IMREAD_COLOR)?;
    let mut result = Mat::zeros(640, 640, opencv::core::CV_8UC3)
        .unwrap()
        .to_mat()
        .unwrap();

    opencv::imgproc::resize(
        &img,
        &mut result,
        (640, 640).into(),
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;
    imwrite("resize.jpg", &result, &Vector::new())?;
    let t = unsafe {
        Tensor::from_blob(
            result.data(),
            &[640, 640, 3],
            &[],
            tch::Kind::Uint8,
            tch::Device::Cpu,
        )
    };
    let t = t.permute([2, 0, 1]);
    let t = swap_bgr_to_rgb(t);
    println!("after: t={:?}", t);
    tch::vision::image::save(&t, "mezi.jpg").expect("can't save image");
    Ok(t)
}

// Preprocess input image: resize and pad the image
pub fn preprocess(image: &str, square_size: i32, center: bool) -> Result<Tensor, Error> {
    let img = opencv::imgcodecs::imread(image, opencv::imgcodecs::IMREAD_COLOR)?;
    let size = img.size()?;
    let (width, height) = (size.width, size.height);
    println!("{width}x{height} -> {square_size}x{square_size}");
    let (uw, uh) = square(square_size, width, height);
    println!("{uw}x{uh}");
    let (mut dw, mut dh) = (square_size - uw, square_size - uh);
    if center {
        dw /= 2;
        dh /= 2;
    }
    let (top, bottom) = if center {
        (
            (dh as f32 - 0.1).round() as i32,
            (dh as f32 - 0.1).round() as i32,
        )
    } else {
        (0, (dh as f32 + 0.1).round() as i32)
    };
    let (left, right) = if center {
        (
            (dw as f32 - 0.1).round() as i32,
            (dw as f32 - 0.1).round() as i32,
        )
    } else {
        (0, (dw as f32 + 0.1).round() as i32)
    };
    let mut result = Mat::zeros(dh, dw, opencv::core::CV_8UC3)
        .unwrap()
        .to_mat()
        .unwrap();
    opencv::imgproc::resize(
        &img,
        &mut result,
        (uw, uh).into(),
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;
    let mut border = Mat::zeros(square_size, square_size, opencv::core::CV_8UC3)
        .unwrap()
        .to_mat()
        .unwrap();
    copy_make_border(
        &result,
        &mut border,
        top,
        bottom,
        left,
        right,
        BORDER_CONSTANT,
        (114, 114, 114).into(),
    )?;
    imwrite("resize.jpg", &border, &Vector::new())?;
    println!("{top},{bottom} -> {left},{right}");

    let t = unsafe {
        Tensor::from_blob(
            border.data(),
            &[640, 640, 3],
            &[],
            tch::Kind::Uint8,
            tch::Device::Cpu,
        )
    };

    //       im = np.stack(self.pre_transform(im))
    // im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    // im = np.ascontiguousarray(im)  # contiguous
    // im = torch.from_numpy(im)

    println!("before: t={:?}", t);
    let t = t.permute([2, 0, 1]);
    let t = swap_bgr_to_rgb(t);
    println!("after: t={:?}", t);
    Ok(t)
}

fn square(size: i32, w: i32, h: i32) -> (i32, i32) {
    let aspect = w as f32 / h as f32;
    if w > h {
        let tw = size;
        let th = (tw as f32 / aspect) as i32;
        (tw, th)
    } else {
        let th = size;
        let tw = (size as f32 * aspect) as i32;
        (tw, th)
    }
}

fn swap_bgr_to_rgb(img_tensor: Tensor) -> Tensor {
    // Ensure the input tensor is of the correct shape
    // Swap channels using indexing
    // The order [2, 1, 0] corresponds to BGR to RGB
    let b = img_tensor.narrow_copy(0, 0, 1);
    img_tensor
        .narrow(0, 0, 1)
        .copy_(&img_tensor.narrow(0, 2, 1));
    img_tensor.narrow(0, 2, 1).copy_(&b);
    img_tensor
}

pub fn print_tensor(t: &Tensor) {
    println!("tensor={}", t);
}

#[cfg(test)]
mod test {

    use tch::Tensor;

    use crate::utils::swap_bgr_to_rgb;

    use super::square;

    #[test]
    fn test_square() {
        assert_eq!((640, 320), square(640, 1280, 640));
        assert_eq!((320, 640), square(640, 640, 1280));
    }

    #[test]
    fn bgr2rgb() {
        let t =
            Tensor::from_slice(&[11, 11, 11, 11, 22, 22, 22, 22, 33, 33, 33, 33]).reshape([3, 4]);
        println!("t={}", t);
        let t = swap_bgr_to_rgb(t);
        // let b = t.narrow(0, 0, 1);
        // let g = t.narrow(0, 1, 1);
        // let r = t.narrow(0, 2, 1);
        // println!("r={}", r);
        // println!("g={}", g);
        // println!("b={}", b);
        // t.narrow_tensor(0, &r, 1);
        println!("t={}", t);
    }
}
