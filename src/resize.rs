use crate::{cast, util, Channels, ColorFormat, ImageView, Precision, ResizeFilter, Size};

pub(crate) use aligned_types::*;
use pic_scale_safe::{ImageSize, ResamplingFunction};

mod aligned_types {
    use crate::{cast, resize::is_aligned, Channels, ColorFormat, ImageView, Precision, Size};

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
        pub fn precision(&self) -> Precision {
            match self {
                Backing::U8(_) => Precision::U8,
                Backing::U16(_) => Precision::U16,
                Backing::F32(_) => Precision::F32,
            }
        }
    }
    impl From<Vec<u8>> for Backing {
        fn from(v: Vec<u8>) -> Self {
            Backing::U8(v)
        }
    }
    impl From<Vec<u16>> for Backing {
        fn from(v: Vec<u16>) -> Self {
            Backing::U16(v)
        }
    }
    impl From<Vec<f32>> for Backing {
        fn from(v: Vec<f32>) -> Self {
            Backing::F32(v)
        }
    }

    pub(crate) struct AlignedBuffer {
        pub(crate) buffer: Backing,
        size: Size,
        channels: Channels,
    }
    impl AlignedBuffer {
        pub fn new(buffer: impl Into<Backing>, size: Size, channels: Channels) -> Self {
            let buffer = buffer.into();

            debug_assert!(
                buffer.as_bytes().len()
                    == ColorFormat::new(channels, buffer.precision())
                        .buffer_size(size)
                        .expect("Invalid size"),
                "Buffer too small for aligned buffer"
            );

            Self {
                buffer,
                size,
                channels,
            }
        }

        pub fn from_view(view: AlignedView<'_>) -> Self {
            let ColorFormat {
                channels,
                precision,
            } = view.color();
            let bytes = view.view();

            let backing = match precision {
                Precision::U8 => Backing::U8(bytes.to_vec()),
                Precision::U16 => Backing::U16(cast::from_bytes(bytes).unwrap().to_vec()),
                Precision::F32 => Backing::F32(cast::from_bytes(bytes).unwrap().to_vec()),
            };

            Self::new(backing, view.size(), channels)
        }

        pub fn color(&self) -> ColorFormat {
            ColorFormat::new(self.channels, self.buffer.precision())
        }

        pub fn as_view(&self) -> AlignedView<'_> {
            AlignedView {
                view: self.buffer.as_bytes(),
                size: self.size,
                color: self.color(),
            }
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
    let needs_premultiply =
        straight_alpha && src.color().channels == Channels::Rgba && filter != ResizeFilter::Nearest;

    let src_buffer = if needs_premultiply {
        let mut buf = AlignedBuffer::from_view(src);
        premultiply_rgba(&mut buf);
        Some(buf)
    } else {
        None
    };

    let src_view = src_buffer.as_ref().map(|b| b.as_view()).unwrap_or(src);

    let mut buffer = base_resize(src_view, new_size, filter);

    if needs_premultiply {
        unpremultiply_rgba(&mut buffer);
    }

    buffer
}

pub(crate) fn base_resize(src: AlignedView, new_size: Size, filter: ResizeFilter) -> AlignedBuffer {
    let ColorFormat {
        channels,
        precision,
    } = src.color();

    let src_size = to_image_size(src.size());
    let dst_size = to_image_size(new_size);
    let resample = to_resampling_function(filter);

    match precision {
        Precision::U8 => {
            let resize_fn = get_resize_fn_u8(channels);
            let src_ref = src.view();
            let buffer: Vec<u8> = resize_fn(src_ref, src_size, dst_size, resample).unwrap();
            AlignedBuffer::new(buffer, new_size, channels)
        }
        Precision::U16 => {
            let resize_fn = get_resize_fn_u16(channels);
            let src_ref = cast::from_bytes(src.view()).unwrap();
            let buffer: Vec<u16> = resize_fn(src_ref, src_size, dst_size, resample).unwrap();
            AlignedBuffer::new(buffer, new_size, channels)
        }
        Precision::F32 => {
            let resize_fn = get_resize_fn_f32(channels);
            let src_ref = cast::from_bytes(src.view()).unwrap();
            let buffer: Vec<f32> = resize_fn(src_ref, src_size, dst_size, resample).unwrap();
            AlignedBuffer::new(buffer, new_size, channels)
        }
    }
}

pub(crate) fn premultiply_rgba(buffer: &mut AlignedBuffer) {
    debug_assert_eq!(buffer.color().channels, Channels::Rgba);
    match &mut buffer.buffer {
        Backing::U8(v) => pic_scale_safe::premultiply_rgba8(v.as_mut_slice()),
        Backing::U16(v) => pic_scale_safe::premultiply_rgba16(v.as_mut_slice(), 16),
        Backing::F32(v) => pic_scale_safe::premultiply_rgba_f32(v.as_mut_slice()),
    }
}
pub(crate) fn unpremultiply_rgba(buffer: &mut AlignedBuffer) {
    debug_assert_eq!(buffer.color().channels, Channels::Rgba);
    match &mut buffer.buffer {
        Backing::U8(v) => pic_scale_safe::unpremultiply_rgba8(v.as_mut_slice()),
        Backing::U16(v) => pic_scale_safe::unpremultiply_rgba16(v.as_mut_slice(), 16),
        Backing::F32(v) => pic_scale_safe::unpremultiply_rgba_f32(v.as_mut_slice()),
    }
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

type ResizeFn<T> = fn(&[T], ImageSize, ImageSize, ResamplingFunction) -> Result<Vec<T>, String>;
fn get_resize_fn_u8(channels: Channels) -> ResizeFn<u8> {
    match channels {
        Channels::Alpha | Channels::Grayscale => pic_scale_safe::resize_plane8,
        Channels::Rgb => pic_scale_safe::resize_rgb8,
        Channels::Rgba => pic_scale_safe::resize_rgba8,
    }
}
fn get_resize_fn_u16(channels: Channels) -> ResizeFn<u16> {
    match channels {
        Channels::Alpha | Channels::Grayscale => |src, src_size, dst_size, resampling_function| {
            pic_scale_safe::resize_plane16(src, src_size, dst_size, 16, resampling_function)
        },
        Channels::Rgb => |src, src_size, dst_size, resampling_function| {
            pic_scale_safe::resize_rgb16(src, src_size, dst_size, 16, resampling_function)
        },
        Channels::Rgba => |src, src_size, dst_size, resampling_function| {
            pic_scale_safe::resize_rgba16(src, src_size, dst_size, 16, resampling_function)
        },
    }
}
fn get_resize_fn_f32(channels: Channels) -> ResizeFn<f32> {
    match channels {
        Channels::Alpha | Channels::Grayscale => pic_scale_safe::resize_plane_f32,
        Channels::Rgb => pic_scale_safe::resize_rgb_f32,
        Channels::Rgba => pic_scale_safe::resize_rgba_f32,
    }
}

fn to_image_size(size: Size) -> ImageSize {
    ImageSize::new(size.width as usize, size.height as usize)
}
fn to_resampling_function(filter: ResizeFilter) -> ResamplingFunction {
    match filter {
        ResizeFilter::Nearest => ResamplingFunction::Nearest,
        ResizeFilter::Box => ResamplingFunction::Area,
        ResizeFilter::Triangle => ResamplingFunction::Bilinear,
        ResizeFilter::Mitchell => ResamplingFunction::MitchellNetravalli,
        ResizeFilter::Lanczos3 => ResamplingFunction::Lanczos3,
    }
}
