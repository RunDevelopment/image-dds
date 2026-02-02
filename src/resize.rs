use std::borrow::Cow;

use crate::{
    cast, util, Channels, ColorFormat, ImageView, Precision, ResizeFilter, Size, WithPrecision,
};

pub(crate) use aligned_types::*;
use pic_scale_safe::{ImageSize, ResamplingFunction};

mod aligned_types {
    use crate::{cast, resize::is_aligned, ColorFormat, ImageView, Size};

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
    }

    pub(crate) struct AlignedBuffer {
        buffer: Backing,
        size: Size,
        color: ColorFormat,
    }
    impl AlignedBuffer {
        pub fn new(buffer: Backing, size: Size, color: ColorFormat) -> Self {
            debug_assert!(
                buffer.as_bytes().len() >= color.buffer_size(size).expect("Invalid size"),
                "Buffer too small for aligned buffer"
            );

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
    let precision = src.color().precision;

    match precision {
        Precision::U8 => {
            let typed_view: TypedView<u8> =
                TypedView::new(src.view(), src.size(), src.color().channels);
            resize_typed(typed_view, new_size, straight_alpha, filter)
        }
        Precision::U16 => {
            let typed_view: TypedView<u16> = TypedView::new(
                cast::from_bytes(src.view()).unwrap(),
                src.size(),
                src.color().channels,
            );
            resize_typed(typed_view, new_size, straight_alpha, filter)
        }
        Precision::F32 => {
            let typed_view: TypedView<f32> = TypedView::new(
                cast::from_bytes(src.view()).unwrap(),
                src.size(),
                src.color().channels,
            );
            resize_typed(typed_view, new_size, straight_alpha, filter)
        }
    }
}
fn resize_typed<P: Copy + WithResizeFn + ToBacking + WithPrecision + PremultipliedAlpha>(
    src: TypedView<P>,
    new_size: Size,
    straight_alpha: bool,
    filter: ResizeFilter,
) -> AlignedBuffer {
    let resize_fn = P::get_resize_fn(src.channels);

    let needs_premultiply =
        straight_alpha && src.channels == Channels::Rgba && filter != ResizeFilter::Nearest;

    let src_buffer = if needs_premultiply {
        let mut buf = src.source.to_vec();
        P::premultiply_rgba(buf.as_mut_slice());
        Cow::Owned(buf)
    } else {
        Cow::Borrowed(src.source)
    };

    let mut buffer = resize_fn(
        src_buffer.as_ref(),
        to_image_size(src.size),
        to_image_size(new_size),
        to_resampling_function(filter),
    )
    .unwrap();

    if needs_premultiply {
        P::unpremultiply_rgba(buffer.as_mut_slice());
    }

    let backing = P::to_backing(buffer);

    AlignedBuffer::new(
        backing,
        new_size,
        ColorFormat::new(src.channels, P::PRECISION),
    )
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

struct TypedView<'a, P> {
    source: &'a [P],
    size: Size,
    channels: Channels,
}
impl<'a, P> TypedView<'a, P> {
    pub fn new(source: &'a [P], size: Size, channels: Channels) -> Self {
        debug_assert_eq!(
            source.len(),
            (size.width as usize)
                .checked_mul(size.height as usize)
                .and_then(|v| v.checked_mul(channels.count() as usize))
                .expect("size too big for typed image")
        );

        Self {
            source,
            size,
            channels,
        }
    }
}

type ResizeFn<T> = fn(&[T], ImageSize, ImageSize, ResamplingFunction) -> Result<Vec<T>, String>;
trait WithResizeFn: Sized {
    fn get_resize_fn(channels: Channels) -> ResizeFn<Self>;
}
impl WithResizeFn for u8 {
    fn get_resize_fn(channels: Channels) -> ResizeFn<Self> {
        match channels {
            Channels::Alpha | Channels::Grayscale => pic_scale_safe::resize_plane8,
            Channels::Rgb => pic_scale_safe::resize_rgb8,
            Channels::Rgba => pic_scale_safe::resize_rgba8,
        }
    }
}
impl WithResizeFn for u16 {
    fn get_resize_fn(channels: Channels) -> ResizeFn<Self> {
        match channels {
            Channels::Alpha | Channels::Grayscale => {
                |source, source_size, destination_size, resampling_function| {
                    pic_scale_safe::resize_plane16(
                        source,
                        source_size,
                        destination_size,
                        16,
                        resampling_function,
                    )
                }
            }
            Channels::Rgb => |source, source_size, destination_size, resampling_function| {
                pic_scale_safe::resize_rgb16(
                    source,
                    source_size,
                    destination_size,
                    16,
                    resampling_function,
                )
            },
            Channels::Rgba => |source, source_size, destination_size, resampling_function| {
                pic_scale_safe::resize_rgba16(
                    source,
                    source_size,
                    destination_size,
                    16,
                    resampling_function,
                )
            },
        }
    }
}
impl WithResizeFn for f32 {
    fn get_resize_fn(channels: Channels) -> ResizeFn<Self> {
        match channels {
            Channels::Alpha | Channels::Grayscale => pic_scale_safe::resize_plane_f32,
            Channels::Rgb => pic_scale_safe::resize_rgb_f32,
            Channels::Rgba => pic_scale_safe::resize_rgba_f32,
        }
    }
}

trait ToBacking: Sized {
    fn to_backing(vec: Vec<Self>) -> Backing;
}
impl ToBacking for u8 {
    fn to_backing(vec: Vec<Self>) -> Backing {
        Backing::U8(vec)
    }
}
impl ToBacking for u16 {
    fn to_backing(vec: Vec<Self>) -> Backing {
        Backing::U16(vec)
    }
}
impl ToBacking for f32 {
    fn to_backing(vec: Vec<Self>) -> Backing {
        Backing::F32(vec)
    }
}

trait PremultipliedAlpha: Sized {
    fn premultiply_rgba(slice: &mut [Self]);
    fn unpremultiply_rgba(slice: &mut [Self]);
}
impl PremultipliedAlpha for u8 {
    fn premultiply_rgba(slice: &mut [Self]) {
        pic_scale_safe::premultiply_rgba8(slice);
    }
    fn unpremultiply_rgba(slice: &mut [Self]) {
        pic_scale_safe::unpremultiply_rgba8(slice);
    }
}
impl PremultipliedAlpha for u16 {
    fn premultiply_rgba(slice: &mut [Self]) {
        pic_scale_safe::premultiply_rgba16(slice, 16);
    }
    fn unpremultiply_rgba(slice: &mut [Self]) {
        pic_scale_safe::unpremultiply_rgba16(slice, 16);
    }
}
impl PremultipliedAlpha for f32 {
    fn premultiply_rgba(slice: &mut [Self]) {
        pic_scale_safe::premultiply_rgba_f32(slice);
    }
    fn unpremultiply_rgba(slice: &mut [Self]) {
        pic_scale_safe::unpremultiply_rgba_f32(slice);
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
