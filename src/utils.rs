use tch::Tensor;

pub fn preprocess_torch(path: &str, square_size: i32) -> Tensor {
    let image = tch::vision::image::load(path).expect("can't load image");
    let (_, height, width) = image.size3().unwrap();
    let (uw, uh) = square(square_size, width as i32, height as i32);
    let scaled_image =
        tch::vision::image::resize(&image, uw as i64, uh as i64).expect("can't resize image");
    let scaled_image = Vec::<u8>::try_from(scaled_image.reshape([-1])).expect("vec");
    let mut gray: Vec<u8> = vec![114; (square_size * square_size * 3) as usize];
    let dh = (square_size - uh) / 2;
    let dw = (square_size - uw) / 2;
    let mut src_y = 0;
    if uw > uh {
        for y in dh..dh + uh {
            let line = get_hline(&scaled_image, (uw as usize, uh as usize), src_y);
            // println!("line={:?}", line);
            put_hline(
                &mut gray,
                (square_size as usize, square_size as usize),
                0,
                y as usize,
                line,
            );
            src_y += 1;
        }
    }
    if uh > uw {
        for y in 0..square_size {
            let line = get_hline(&scaled_image, (uw as usize, uh as usize), src_y);
            // println!("line={:?}", line);
            put_hline(
                &mut gray,
                (square_size as usize, square_size as usize),
                dw as usize,
                y as usize,
                line,
            );
            src_y += 1;
        }
    }

    let border = Tensor::from_slice(&gray).reshape([3, square_size as i64, square_size as i64]);
    tch::vision::image::save(&border, "border.jpg").expect("can't save image");
    border
}

fn put_hline(
    v: &mut Vec<u8>,
    (w, h): (usize, usize),
    x_off: usize,
    y: usize,
    (r, g, b): (Vec<u8>, Vec<u8>, Vec<u8>),
) {
    let r_off = 0;
    let g_off = w * h;
    let b_off = 2 * w * h;
    let mut s_off = y * w;
    for i in 0..r.len() {
        // println!("getline: y={y}, i={i}, s_off={s_off} b_off={b_off} idx={idx}");
        v[r_off + x_off + s_off] = r[i];
        v[g_off + x_off + s_off] = g[i];
        v[b_off + x_off + s_off] = b[i];
        s_off += 1;
    }
}

fn get_hline(v: &Vec<u8>, (w, h): (usize, usize), y: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let r_off = 0;
    let g_off = w * h;
    let b_off = 2 * w * h;
    let mut s_off = y * w;
    let mut r = vec![0; w];
    let mut g = vec![0; w];
    let mut b = vec![0; w];
    for i in 0..w {
        // println!("getline: y={y}, i={i}, s_off={s_off} b_off={b_off} idx={idx}");
        r[i] = v[r_off + s_off];
        g[i] = v[g_off + s_off];
        b[i] = v[b_off + s_off];
        s_off += 1;
    }
    (r, g, b)
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
