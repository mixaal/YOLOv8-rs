use tch::Tensor;

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
