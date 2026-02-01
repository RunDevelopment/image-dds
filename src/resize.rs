use fast_image_resize as fr;

use crate::{cast, util, Channels, ColorFormat, ImageView, ResizeFilter, Size};

pub(crate) use aligned_types::*;

mod aligned_types {
    use crate::{cast, resize::is_aligned, ColorFormat, ImageView, Precision, Size};

    pub(crate) type AlignTo = u32;

    #[derive(Clone, Copy)]
    pub(crate) struct AlignedView<'a> {
        view: &'a [u8],
        size: Size,
        color: ColorFormat,
    }
    impl<'a> AlignedView<'a> {
        pub fn new(view: &'a [u8], size: Size, color: ColorFormat) -> Self {
            debug_assert_eq!(view.len(), color.buffer_size(size).expect("Invalid size"));
            debug_assert!(is_aligned(view, color.precision.size() as usize));

            Self { view, size, color }
        }

        pub fn view(&self) -> &'a [u8] {
            self.view
        }
        pub fn size(&self) -> Size {
            self.size
        }
        pub fn color(&self) -> ColorFormat {
            self.color
        }

        pub fn as_image_view(&self) -> ImageView<'a> {
            ImageView::new(self.view, self.size, self.color).expect("invalid aligned view")
        }
    }

    pub(crate) enum Backing {
        U8(Vec<u8>),
        U16(Vec<u16>),
        F32(Vec<f32>),
    }
    impl Backing {
        pub fn as_bytes(&self) -> &[u8] {
            match self {
                Backing::U8(v) => cast::as_bytes(v),
                Backing::U16(v) => cast::as_bytes(v),
                Backing::F32(v) => cast::as_bytes(v),
            }
        }
        pub fn as_bytes_mut(&mut self) -> &mut [u8] {
            match self {
                Backing::U8(v) => cast::as_bytes_mut(v),
                Backing::U16(v) => cast::as_bytes_mut(v),
                Backing::F32(v) => cast::as_bytes_mut(v),
            }
        }
    }

    pub(crate) struct AlignedBuffer {
        buffer: Backing,
        size: Size,
        color: ColorFormat,
    }
    impl AlignedBuffer {
        pub fn create(size: Size, color: ColorFormat) -> Self {
            let bytes = color.buffer_size(size).expect("Invalid size");
            let buffer = match color.precision {
                Precision::U8 => Backing::U8(vec![0u8; bytes]),
                Precision::U16 => Backing::U16(vec![0u16; bytes / 2]),
                Precision::F32 => Backing::F32(vec![0f32; bytes / 4]),
            };

            Self {
                buffer,
                size,
                color,
            }
        }

        pub fn as_view(&self) -> AlignedView<'_> {
            let byte_slice = self.buffer.as_bytes();
            let len = self.color.buffer_size(self.size).expect("Invalid size");

            AlignedView {
                view: &byte_slice[..len],
                size: self.size,
                color: self.color,
            }
        }
        pub fn as_bytes_mut(&mut self) -> &mut [u8] {
            self.buffer.as_bytes_mut()
        }
    }
}

pub(crate) struct Aligner {
    buffer: Vec<AlignTo>,
}
impl Aligner {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn align<'a>(&'a mut self, image: ImageView<'a>) -> AlignedView<'a> {
        let size = image.size();
        let color = image.color();
        let data = image.data();

        let bytes_per_pixel = color.bytes_per_pixel() as usize;

        let view = if !image.is_contiguous() {
            // Right now, the implementation assumes that the data to be
            // contiguous, so we need to copy it to an aligned buffer line by line.
            let aligned_slice = get_aligned_slice(&mut self.buffer, size, color);
            let bytes_per_row = size.width as usize * bytes_per_pixel;
            for (y, data_row) in image.rows().enumerate() {
                debug_assert_eq!(data_row.len(), bytes_per_row);
                let a_start = y * bytes_per_row;
                let a_end = a_start + bytes_per_row;
                aligned_slice[a_start..a_end].copy_from_slice(data_row);
            }
            aligned_slice
        } else if is_aligned(data, color.precision.size() as usize) {
            data
        } else {
            // the image data isn't aligned, so we need to copy it to an aligned buffer
            let aligned_slice = get_aligned_slice(&mut self.buffer, size, color);
            aligned_slice.copy_from_slice(data);
            aligned_slice
        };

        AlignedView::new(view, size, color)
    }
}

pub(crate) fn resize(
    src: AlignedView,
    new_size: Size,
    straight_alpha: bool,
    filter: ResizeFilter,
) -> AlignedBuffer {
    let options = fr::ResizeOptions {
        algorithm: to_resize_alg(filter),
        cropping: fr::SrcCropping::None,
        mul_div_alpha: straight_alpha && src.color().channels == Channels::Rgba,
    };
    let pixel_type = to_pixel_type(src.color());
    let src_size = src.size();

    let src_ref =
        fr::images::ImageRef::new(src_size.width, src_size.height, src.view(), pixel_type).unwrap();

    let mut dst = AlignedBuffer::create(new_size, src.color());
    let mut dst_ref = fr::images::Image::from_slice_u8(
        new_size.width,
        new_size.height,
        dst.as_bytes_mut(),
        pixel_type,
    )
    .unwrap();

    let mut resizer = fr::Resizer::new();
    resizer
        .resize(&src_ref, &mut dst_ref, &options)
        .expect("resize failed");

    dst
}

fn get_aligned_slice(buffer: &mut Vec<AlignTo>, size: Size, color: ColorFormat) -> &mut [u8] {
    let slice_len = color
        .buffer_size(size)
        .expect("size too big for aligned slice");

    // reserve enough space in the buffer
    let buffer_len = util::div_ceil(slice_len, std::mem::size_of::<AlignTo>());
    if buffer.len() < buffer_len {
        buffer.resize(buffer_len, 0);
    }

    &mut cast::as_bytes_mut(buffer.as_mut_slice())[..slice_len]
}
fn is_aligned(slice: &[u8], alignment: usize) -> bool {
    (slice.as_ptr() as usize) % alignment == 0
}

fn to_pixel_type(color: ColorFormat) -> fr::PixelType {
    match color {
        ColorFormat::ALPHA_U8 => fr::PixelType::U8,
        ColorFormat::ALPHA_U16 => fr::PixelType::U16,
        ColorFormat::ALPHA_F32 => fr::PixelType::F32,
        ColorFormat::GRAYSCALE_U8 => fr::PixelType::U8,
        ColorFormat::GRAYSCALE_U16 => fr::PixelType::U16,
        ColorFormat::GRAYSCALE_F32 => fr::PixelType::F32,
        ColorFormat::RGB_U8 => fr::PixelType::U8x3,
        ColorFormat::RGB_U16 => fr::PixelType::U16x3,
        ColorFormat::RGB_F32 => fr::PixelType::F32x3,
        ColorFormat::RGBA_U8 => fr::PixelType::U8x4,
        ColorFormat::RGBA_U16 => fr::PixelType::U16x4,
        ColorFormat::RGBA_F32 => fr::PixelType::F32x4,
    }
}
fn to_resize_alg(filter: ResizeFilter) -> fr::ResizeAlg {
    match filter {
        ResizeFilter::Nearest => fr::ResizeAlg::Nearest,
        ResizeFilter::Box => fr::ResizeAlg::Convolution(fr::FilterType::Box),
        ResizeFilter::Triangle => fr::ResizeAlg::Convolution(fr::FilterType::Bilinear),
        ResizeFilter::Mitchell => fr::ResizeAlg::Convolution(fr::FilterType::Mitchell),
        ResizeFilter::Lanczos3 => fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3),
    }
}
