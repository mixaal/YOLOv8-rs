use opencv::{
    core::{copy_make_border, Vector, BORDER_CONSTANT},
    imgcodecs::imwrite,
    prelude::*,
    Error,
};

// Preprocess input image: resize and pad the image
pub fn preprocess(image: &str, square_size: i32) -> Result<(), Error> {
    let img = opencv::imgcodecs::imread(image, opencv::imgcodecs::IMREAD_COLOR)?;
    let size = img.size()?;
    let (width, height) = (size.width, size.height);
    println!("{width}x{height} -> {square_size}x{square_size}");
    let (uw, uh) = square(square_size, width, height);
    println!("{uw}x{uh}");
    let (mut dw, mut dh) = (square_size - uw, square_size - uh);
    dw /= 2;
    dh /= 2;
    let (top, bottom) = (
        (dh as f32 - 0.1).round() as i32,
        (dh as f32 - 0.1).round() as i32,
    );
    let (left, right) = (
        (dw as f32 - 0.1).round() as i32,
        (dw as f32 - 0.1).round() as i32,
    );
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
    Ok(())
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

#[cfg(test)]
mod test {
    use super::square;

    #[test]
    fn test_square() {
        assert_eq!((640, 320), square(640, 1280, 640));
        assert_eq!((320, 640), square(640, 640, 1280));
    }
}
