use glam::Vec4;

use crate::{
    as_rgba_f32, cast, ch, convert_channels, convert_channels_for,
    encode::write_util::for_each_chunk, fp10, fp11, fp16, n1, n10, n16, n2, n4, n5, n6, n8,
    rgb9995f, s16, s8, util, xr10, yuv10, yuv16, yuv8, Channels, ColorFormat, ColorFormatSet,
    EncodingError, Precision, Report,
};

use super::{
    encoder::{Args, Encoder, EncoderSet, Flags},
    Dithering,
};

// helpers

const REPORT_FREQUENCY: usize = 2048;

fn uncompressed_universal<EncodedPixel>(
    args: Args,
    process: fn(&[[f32; 4]], &mut [EncodedPixel]),
) -> Result<(), EncodingError>
where
    EncodedPixel: Default + Copy + cast::ToLe + cast::Castable,
{
    let Args {
        image,
        writer,
        mut progress,
        ..
    } = args;
    let color = image.color();

    const BUFFER_PIXELS: usize = 512;
    let mut intermediate_buffer = [[0_f32; 4]; BUFFER_PIXELS];
    let mut encoded_buffer = [EncodedPixel::default(); BUFFER_PIXELS];

    let chunk_count = util::div_ceil(image.size().pixels() as usize, BUFFER_PIXELS);
    let mut chunk_index = 0;
    for_each_chunk(
        image,
        &mut encoded_buffer,
        1,
        |partial, encoded| {
            let intermediate = &mut intermediate_buffer[..encoded.len()];
            process(as_rgba_f32(color, partial, intermediate), encoded);
        },
        |encoded| {
            // occasionally report progress
            if chunk_index % REPORT_FREQUENCY == 0 {
                progress.report(chunk_index as f32 / chunk_count as f32);
            }
            chunk_index += 1;

            cast::ToLe::to_le(encoded);
            writer.write_all(cast::as_bytes(encoded))
        },
    )?;

    Ok(())
}

fn uncompressed_universal_dither<EncodedPixel, F>(args: Args, f: F) -> Result<(), EncodingError>
where
    EncodedPixel: Default + Copy + cast::ToLe + cast::Castable,
    F: Fn(Vec4) -> (EncodedPixel, Vec4),
{
    let Args {
        image,
        writer,
        options,
        mut progress,
        ..
    } = args;
    let color = image.color();
    let bytes_per_pixel = color.bytes_per_pixel() as usize;
    let width = image.width() as usize;
    let height = image.height() as usize;

    let error_padding = 2;
    let mut error_buffer = vec![Vec4::ZERO; 2 * (width + error_padding * 2)];
    let (mut current_line_error, mut next_line_error) =
        error_buffer.split_at_mut(width + error_padding * 2);

    let error_mask = match options.dithering {
        Dithering::None => Vec4::ZERO,
        Dithering::ColorAndAlpha => Vec4::ONE,
        Dithering::Color => Vec4::new(1.0, 1.0, 1.0, 0.0),
        Dithering::Alpha => Vec4::new(0.0, 0.0, 0.0, 1.0),
    };

    const BUFFER_PIXELS: usize = 512;
    let mut intermediate_buffer = [[0_f32; 4]; BUFFER_PIXELS];
    let mut encoded_buffer = [EncodedPixel::default(); BUFFER_PIXELS];

    let chunk_size = BUFFER_PIXELS * bytes_per_pixel;
    let chunk_count = height * util::div_ceil(width * bytes_per_pixel, chunk_size);
    let mut chunk_index: usize = 0;
    for row in image.rows() {
        debug_assert!(row.len() == width * bytes_per_pixel);

        // prepare error buffers
        std::mem::swap(&mut current_line_error, &mut next_line_error);
        next_line_error.fill(Vec4::ZERO);
        let mut error_offset = error_padding;
        let mut next_error_add = Vec4::ZERO;

        for line in row.chunks(chunk_size) {
            // occasionally report progress
            if chunk_index % REPORT_FREQUENCY == 0 {
                progress.report(chunk_index as f32 / chunk_count as f32);
            }
            chunk_index += 1;

            debug_assert!(line.len() % bytes_per_pixel == 0);
            let pixels = line.len() / bytes_per_pixel;

            let intermediate = &mut intermediate_buffer[..pixels];
            let encoded = &mut encoded_buffer[..pixels];
            let intermediate = as_rgba_f32(color, line, intermediate);

            for (i, out) in intermediate.iter().zip(encoded.iter_mut()) {
                let error = current_line_error[error_offset] + next_error_add;
                let (encoded_pixel, mut error) = f(Vec4::from(*i) + error);

                // diffuse error with Floyd-Steinberg weights
                error *= error_mask;
                next_error_add = error * (7.0 / 16.0);
                next_line_error[error_offset - 1] += error * (3.0 / 16.0);
                next_line_error[error_offset] += error * (5.0 / 16.0);
                next_line_error[error_offset + 1] += error * (1.0 / 16.0);

                *out = encoded_pixel;
                error_offset += 1;
            }

            cast::ToLe::to_le(encoded);
            writer.write_all(cast::as_bytes(encoded))?;
        }
    }

    Ok(())
}

fn uncompressed_untyped(
    args: Args,
    bytes_per_encoded_pixel: usize,
    f: fn(&[u8], ColorFormat, &mut [u8]),
) -> Result<(), EncodingError> {
    let Args {
        image,
        writer,
        mut progress,
        ..
    } = args;
    let color = image.color();

    let mut raw_buffer = [0_u32; 1024];
    let encoded_buffer = cast::as_bytes_mut(&mut raw_buffer);

    let chunk_count = util::div_ceil(
        image.size().pixels() as usize,
        encoded_buffer.len() / bytes_per_encoded_pixel,
    );
    let mut chunk_index = 0;

    for_each_chunk(
        image,
        encoded_buffer,
        bytes_per_encoded_pixel,
        |partial, encoded| f(partial, color, encoded),
        |encoded| {
            // occasionally report progress
            if chunk_index % REPORT_FREQUENCY == 0 {
                progress.report(chunk_index as f32 / chunk_count as f32);
            }
            chunk_index += 1;

            writer.write_all(encoded)
        },
    )?;

    Ok(())
}
fn simple_color_convert(
    line: &[u8],
    color: ColorFormat,
    out: &mut [u8],
    target: ColorFormat,
    snorm: bool,
) {
    assert!(color.precision == target.precision);

    convert_channels_for(color, target.channels, line, out);

    if snorm {
        match target.precision {
            Precision::U8 => {
                out.iter_mut().for_each(|o| *o = s8::from_n8(*o));
            }
            Precision::U16 => {
                let chunked: &mut [[u8; 2]] =
                    cast::as_array_chunks_mut(out).expect("invalid buffer size");
                chunked.iter_mut().for_each(|o| {
                    *o = s16::from_n16(u16::from_ne_bytes(*o)).to_ne_bytes();
                });
            }
            Precision::F32 => unreachable!(),
        }
    }

    cast::slice_ne_to_le(target.precision, out);
}

macro_rules! color_convert {
    ($target:expr, snorm = $snorm:literal) => {
        Encoder::new(
            ColorFormatSet::from_precision($target.precision),
            Flags::exact_for($target.precision),
            |args| {
                uncompressed_untyped(
                    args,
                    $target.bytes_per_pixel() as usize,
                    |line, color, out| simple_color_convert(line, color, out, $target, $snorm),
                )
            },
        )
    };
    ($target:expr) => {
        color_convert!($target, snorm = false)
    };
}

macro_rules! universal {
    ($out:ty, $f:expr) => {{
        fn process_line(line: &[[f32; 4]], out: &mut [$out]) {
            assert!(line.len() == out.len());
            let f = util::closure_types::<[f32; 4], $out, _>($f);
            for (i, o) in line.iter().zip(out.iter_mut()) {
                *o = f(*i);
            }
        }
        Encoder::new_universal(|args| uncompressed_universal(args, process_line))
    }};
}
macro_rules! universal_grayscale {
    ($out:ty, $f:expr) => {
        universal!($out, |rgba| ($f)(ch::rgba_to_grayscale(rgba)[0]))
    };
}
macro_rules! universal_dither {
    ($out:ty, $f:expr) => {
        Encoder::new_universal(|args| uncompressed_universal_dither::<$out, _>(args, $f))
    };
}

// encoders

pub(crate) const R8G8B8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGB_U8),
    color_convert!(ColorFormat::RGB_U8),
    universal!([u8; 3], |[r, g, b, _]| [r, g, b].map(n8::from_f32)),
]);

pub(crate) const B8G8R8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::new(ColorFormatSet::U8, Flags::EXACT_U8, |args| {
        fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
            assert!(color.precision == Precision::U8);
            convert_channels::<u8>(color.channels, Channels::Rgb, line, out);

            // swap R and B
            let chunked: &mut [[u8; 3]] =
                cast::as_array_chunks_mut(out).expect("invalid buffer size");
            chunked.iter_mut().for_each(|p| p.swap(0, 2));
        }

        uncompressed_untyped(args, 3, process_line)
    }),
    universal!([u8; 3], |[r, g, b, _]| [b, g, r].map(n8::from_f32)),
]);

pub(crate) const R8G8B8A8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGBA_U8),
    color_convert!(ColorFormat::RGBA_U8),
    universal!([u8; 4], |rgba| rgba.map(n8::from_f32)),
]);

pub(crate) const R8G8B8A8_SNORM: EncoderSet = EncoderSet::new(&[
    color_convert!(ColorFormat::RGBA_U8, snorm = true),
    universal!([u8; 4], |rgba| rgba.map(s8::from_uf32)),
]);

pub(crate) const B8G8R8A8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::new(ColorFormatSet::U8, Flags::EXACT_U8, |args| {
        fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
            assert!(color.precision == Precision::U8);
            convert_channels::<u8>(color.channels, Channels::Rgba, line, out);

            // swap R and B
            let chunked: &mut [[u8; 4]] =
                cast::as_array_chunks_mut(out).expect("invalid buffer size");
            chunked.iter_mut().for_each(|p| p.swap(0, 2));
        }

        uncompressed_untyped(args, 4, process_line)
    }),
    universal!([u8; 4], |[r, g, b, a]| [b, g, r, a].map(n8::from_f32)),
]);

pub(crate) const B8G8R8X8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::new(ColorFormatSet::U8, Flags::EXACT_U8, |args| {
        fn process_line(line: &[u8], color: ColorFormat, out: &mut [u8]) {
            assert!(color.precision == Precision::U8);
            convert_channels::<u8>(color.channels, Channels::Rgba, line, out);

            // swap R and B and set X to 0xFF
            let chunked: &mut [[u8; 4]] =
                cast::as_array_chunks_mut(out).expect("invalid buffer size");
            chunked.iter_mut().for_each(|p| {
                p.swap(0, 2);
                p[3] = 0xFF;
            });
        }

        uncompressed_untyped(args, 4, process_line)
    }),
    universal!([u8; 4], |[r, g, b, _]| [
        n8::from_f32(b),
        n8::from_f32(g),
        n8::from_f32(r),
        0xFF
    ]),
]);

pub(crate) const B5G6R5_UNORM: EncoderSet = EncoderSet::new(&[
    universal!(u16, |[r, g, b, _]| {
        let r = n5::from_f32(r) as u16;
        let g = n6::from_f32(g) as u16;
        let b = n5::from_f32(b) as u16;
        b | (g << 5) | (r << 11)
    }),
    universal_dither!(u16, |pixel| {
        let r = n5::from_f32(pixel[0]) as u16;
        let g = n6::from_f32(pixel[1]) as u16;
        let b = n5::from_f32(pixel[2]) as u16;

        let back = Vec4::new(n5::f32(r as u8), n6::f32(g as u8), n5::f32(b as u8), 1.0);
        let error = pixel - back;

        (b | (g << 5) | (r << 11), error)
    })
    .add_flags(Flags::DITHER_COLOR),
]);

pub(crate) const B5G5R5A1_UNORM: EncoderSet = EncoderSet::new(&[
    universal!(u16, |[r, g, b, a]| {
        let r = n5::from_f32(r) as u16;
        let g = n5::from_f32(g) as u16;
        let b = n5::from_f32(b) as u16;
        let a = n1::from_f32(a) as u16;
        b | (g << 5) | (r << 10) | (a << 15)
    }),
    universal_dither!(u16, |pixel| {
        let r = n5::from_f32(pixel[0]) as u16;
        let g = n5::from_f32(pixel[1]) as u16;
        let b = n5::from_f32(pixel[2]) as u16;
        let a = n1::from_f32(pixel[3]) as u16;

        let back = Vec4::new(
            n5::f32(r as u8),
            n5::f32(g as u8),
            n5::f32(b as u8),
            n1::f32(a as u8),
        );
        let error = pixel - back;

        (b | (g << 5) | (r << 10) | (a << 15), error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

fn rgba4_encode_with_error(pixel: Vec4) -> ([u8; 4], Vec4) {
    let encoded = pixel.to_array().map(n4::from_f32);
    let back = Vec4::from(encoded.map(n4::f32));
    let error = pixel - back;
    (encoded, error)
}

pub(crate) const B4G4R4A4_UNORM: EncoderSet = EncoderSet::new(&[
    universal!(u16, |[r, g, b, a]| {
        let r = n4::from_f32(r) as u16;
        let g = n4::from_f32(g) as u16;
        let b = n4::from_f32(b) as u16;
        let a = n4::from_f32(a) as u16;
        b | (g << 4) | (r << 8) | (a << 12)
    }),
    universal_dither!(u16, |pixel| {
        let (encoded, error) = rgba4_encode_with_error(pixel);
        let [r, g, b, a] = encoded.map(|c| c as u16);
        (b | (g << 4) | (r << 8) | (a << 12), error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const A4B4G4R4_UNORM: EncoderSet = EncoderSet::new(&[
    universal!(u16, |[r, g, b, a]| {
        let r = n4::from_f32(r) as u16;
        let g = n4::from_f32(g) as u16;
        let b = n4::from_f32(b) as u16;
        let a = n4::from_f32(a) as u16;
        a | (b << 4) | (g << 8) | (r << 12)
    }),
    universal_dither!(u16, |pixel| {
        let (encoded, error) = rgba4_encode_with_error(pixel);
        let [r, g, b, a] = encoded.map(|c| c as u16);
        (a | (b << 4) | (g << 8) | (r << 12), error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const R8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::GRAYSCALE_U8),
    color_convert!(ColorFormat::GRAYSCALE_U8),
    universal_grayscale!(u8, n8::from_f32),
]);

pub(crate) const R8_SNORM: EncoderSet = EncoderSet::new(&[
    color_convert!(ColorFormat::GRAYSCALE_U8, snorm = true),
    universal_grayscale!(u8, s8::from_uf32),
]);

pub(crate) const R8G8_UNORM: EncoderSet =
    EncoderSet::new(&[
        universal!([u8; 2], |[r, g, _, _]| [r, g].map(n8::from_f32)).add_flags(Flags::EXACT_U8)
    ]);

pub(crate) const R8G8_SNORM: EncoderSet =
    EncoderSet::new(&[
        universal!([u8; 2], |[r, g, _, _]| [r, g].map(s8::from_uf32)).add_flags(Flags::EXACT_U8),
    ]);

pub(crate) const A8_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::ALPHA_U8),
    color_convert!(ColorFormat::ALPHA_U8),
    universal!(u8, |[_, _, _, a]| n8::from_f32(a)),
]);

pub(crate) const R16_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::GRAYSCALE_U16),
    color_convert!(ColorFormat::GRAYSCALE_U16),
    universal_grayscale!(u16, n16::from_f32),
]);

pub(crate) const R16_SNORM: EncoderSet = EncoderSet::new(&[
    color_convert!(ColorFormat::GRAYSCALE_U16, snorm = true),
    universal_grayscale!(u16, s16::from_uf32),
]);

pub(crate) const R16G16_UNORM: EncoderSet =
    EncoderSet::new(&[
        universal!([u16; 2], |[r, g, _, _]| [r, g].map(n16::from_f32)).add_flags(Flags::EXACT_U16),
    ]);

pub(crate) const R16G16_SNORM: EncoderSet =
    EncoderSet::new(&[
        universal!([u16; 2], |[r, g, _, _]| [r, g].map(s16::from_uf32)).add_flags(Flags::EXACT_U16),
    ]);

pub(crate) const R16G16B16A16_UNORM: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGBA_U16),
    color_convert!(ColorFormat::RGBA_U16),
    universal!([u16; 4], |rgba| rgba.map(n16::from_f32)),
]);

pub(crate) const R16G16B16A16_SNORM: EncoderSet = EncoderSet::new(&[
    color_convert!(ColorFormat::RGBA_U16, snorm = true),
    universal!([u16; 4], |rgba| rgba.map(s16::from_uf32)),
]);

pub(crate) const R10G10B10A2_UNORM: EncoderSet = EncoderSet::new(&[
    universal!(u32, |[r, g, b, a]| {
        let r = n10::from_f32(r) as u32;
        let g = n10::from_f32(g) as u32;
        let b = n10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;
        (a << 30) | (b << 20) | (g << 10) | r
    }),
    universal_dither!(u32, |pixel| {
        let [r, g, b, a] = pixel.to_array();
        let r = n10::from_f32(r) as u32;
        let g = n10::from_f32(g) as u32;
        let b = n10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;

        let back = Vec4::new(
            n10::f32(r as u16),
            n10::f32(g as u16),
            n10::f32(b as u16),
            n2::f32(a as u8),
        );
        let error = pixel - back;

        ((a << 30) | (b << 20) | (g << 10) | r, error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const R11G11B10_FLOAT: EncoderSet = EncoderSet::new(&[
    universal!(u32, |[r, g, b, _]| {
        let r11 = fp11::from_f32(r) as u32;
        let g11 = fp11::from_f32(g) as u32;
        let b10 = fp10::from_f32(b) as u32;
        (b10 << 22) | (g11 << 11) | r11
    }),
    universal_dither!(u32, |pixel| {
        let r11 = fp11::from_f32(pixel[0]) as u32;
        let g11 = fp11::from_f32(pixel[1]) as u32;
        let b10 = fp10::from_f32(pixel[2]) as u32;

        let back = Vec4::new(
            fp11::f32(r11 as u16),
            fp11::f32(g11 as u16),
            fp10::f32(b10 as u16),
            1.0,
        );
        let error = pixel - back;

        ((b10 << 22) | (g11 << 11) | r11, error)
    })
    .add_flags(Flags::DITHER_COLOR),
]);

pub(crate) const R9G9B9E5_SHAREDEXP: EncoderSet =
    EncoderSet::new(&[
        universal!(u32, |[r, g, b, _]| { rgb9995f::from_f32([r, g, b]) })
            .add_flags(Flags::EXACT_U8),
    ]);

pub(crate) const R16_FLOAT: EncoderSet =
    EncoderSet::new(&[universal_grayscale!(u16, fp16::from_f32).add_flags(Flags::EXACT_U8)]);

pub(crate) const R16G16_FLOAT: EncoderSet =
    EncoderSet::new(&[
        universal!([u16; 2], |[r, g, _, _]| [r, g].map(fp16::from_f32)).add_flags(Flags::EXACT_U8),
    ]);

pub(crate) const R16G16B16A16_FLOAT: EncoderSet =
    EncoderSet::new(&[
        universal!([u16; 4], |rgba| rgba.map(fp16::from_f32)).add_flags(Flags::EXACT_U8)
    ]);

pub(crate) const R32_FLOAT: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::GRAYSCALE_F32),
    color_convert!(ColorFormat::GRAYSCALE_F32),
    universal_grayscale!(f32, |r| r),
]);

pub(crate) const R32G32_FLOAT: EncoderSet =
    EncoderSet::new(&[universal!([f32; 2], |[r, g, _, _]| [r, g]).add_flags(Flags::EXACT_F32)]);

pub(crate) const R32G32B32_FLOAT: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGB_F32),
    color_convert!(ColorFormat::RGB_F32),
    universal!([f32; 3], |[r, g, b, _]| [r, g, b]),
]);

pub(crate) const R32G32B32A32_FLOAT: EncoderSet = EncoderSet::new(&[
    Encoder::copy(ColorFormat::RGBA_F32),
    color_convert!(ColorFormat::RGBA_F32),
    universal!([f32; 4], |[r, g, b, a]| [r, g, b, a]),
]);

pub(crate) const R10G10B10_XR_BIAS_A2_UNORM: EncoderSet = EncoderSet::new(&[
    universal!(u32, |[r, g, b, a]| {
        let r = xr10::from_f32(r) as u32;
        let g = xr10::from_f32(g) as u32;
        let b = xr10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;
        (a << 30) | (b << 20) | (g << 10) | r
    }),
    universal_dither!(u32, |pixel| {
        let [r, g, b, a] = pixel.to_array();
        let r = xr10::from_f32(r) as u32;
        let g = xr10::from_f32(g) as u32;
        let b = xr10::from_f32(b) as u32;
        let a = n2::from_f32(a) as u32;

        let back = Vec4::new(
            xr10::f32(r as u16),
            xr10::f32(g as u16),
            xr10::f32(b as u16),
            n2::f32(a as u8),
        );
        let error = pixel - back;

        ((a << 30) | (b << 20) | (g << 10) | r, error)
    })
    .add_flags(Flags::DITHER_ALL),
]);

pub(crate) const AYUV: EncoderSet = EncoderSet::new(&[universal!([u8; 4], |[r, g, b, a]| {
    let [y, u, v] = yuv8::from_rgb_f32([r, g, b]);
    let a = n8::from_f32(a);
    [v, u, y, a]
})]);

pub(crate) const Y410: EncoderSet = EncoderSet::new(&[
    universal!(u32, |[r, g, b, a]| {
        let [y, u, v] = yuv10::from_rgb_f32([r, g, b]);
        let a = n2::from_f32(a) as u32;
        (a << 30) | ((v as u32) << 20) | ((y as u32) << 10) | (u as u32)
    }),
    universal_dither!(u32, |pixel| {
        let [r, g, b, a_f32] = pixel.to_array();
        let [y, u, v] = yuv10::from_rgb_f32([r, g, b]);
        let a = n2::from_f32(a_f32) as u32;

        let a_back = n2::f32(a as u8);
        let error = Vec4::new(0.0, 0.0, 0.0, a_f32 - a_back);

        (
            (a << 30) | ((v as u32) << 20) | ((y as u32) << 10) | (u as u32),
            error,
        )
    })
    .add_flags(Flags::DITHER_ALPHA),
]);

pub(crate) const Y416: EncoderSet = EncoderSet::new(&[universal!([u16; 4], |[r, g, b, a]| {
    let [y, u, v] = yuv16::from_rgb_f32([r, g, b]);
    let a = n16::from_f32(a);
    [u, y, v, a]
})
.add_flags(Flags::EXACT_U8)]);
