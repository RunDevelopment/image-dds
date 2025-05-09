#![forbid(unsafe_code)]

mod cast;
mod color;
mod decode;
mod decoder;
mod detect;
mod encode;
mod encoder;
mod error;
mod format;
pub mod header;
mod iter;
mod layout;
mod pixel;
mod progress;
mod resize;
mod split;
mod util;

pub use color::*;
pub use decode::{decode, decode_rect, DecodeOptions};
pub use decoder::*;
pub use encode::{
    encode, CompressionQuality, Dithering, EncodeOptions, EncodingSupport, ErrorMetric,
};
pub use encoder::*;
pub use error::*;
pub use format::*;
pub use layout::*;
pub use pixel::*;
pub use progress::*;
pub use split::*;

/// A borrowed slice of image data.
#[derive(Clone, Copy)]
pub struct ImageView<'a> {
    data: &'a [u8],
    size: Size,
    color: ColorFormat,
}
impl<'a> ImageView<'a> {
    /// Creates a new image view from the given data, size, and color format.
    ///
    /// The data must be the correct size for the given size and color format.
    /// If `data.len() != size.pixels() * color.bytes_per_pixel()`,
    /// then `None` is returned.
    pub fn new(data: &'a [u8], size: Size, color: ColorFormat) -> Option<Self> {
        if data.len() as u64 != size.pixels().saturating_mul(color.bytes_per_pixel() as u64) {
            return None;
        }
        Some(Self { data, size, color })
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    pub fn size(&self) -> Size {
        self.size
    }
    pub fn width(&self) -> u32 {
        self.size.width
    }
    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn color(&self) -> ColorFormat {
        self.color
    }
    pub fn row_pitch(&self) -> usize {
        self.size.width as usize * self.color.bytes_per_pixel() as usize
    }
}

/// A borrowed mutable slice of image data.
pub struct ImageViewMut<'a> {
    data: &'a mut [u8],
    size: Size,
    color: ColorFormat,
}
impl<'a> ImageViewMut<'a> {
    /// Creates a new image view from the given data, size, and color format.
    ///
    /// The data must be the correct size for the given size and color format.
    /// If `data.len() != size.pixels() * color.bytes_per_pixel()`,
    /// then `None` is returned.
    pub fn new(data: &'a mut [u8], size: Size, color: ColorFormat) -> Option<Self> {
        if data.len() as u64 != size.pixels().saturating_mul(color.bytes_per_pixel() as u64) {
            return None;
        }
        Some(Self { data, size, color })
    }

    pub fn data(&mut self) -> &mut [u8] {
        self.data
    }

    pub fn size(&self) -> Size {
        self.size
    }
    pub fn width(&self) -> u32 {
        self.size.width
    }
    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn color(&self) -> ColorFormat {
        self.color
    }
    pub fn row_pitch(&self) -> usize {
        self.size.width as usize * self.color.bytes_per_pixel() as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}
impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
    /// Whether the area of this size is zero.
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
    /// The number of pixels in this size.
    pub const fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Returns the size of the mipmap at the given level.
    ///
    /// Level 0 is the original size. All returned sizes are at least 1x1.
    ///
    /// ```
    /// # use dds::Size;
    /// let size = Size::new(256, 100);
    /// assert_eq!(size.get_mipmap(0), size);
    /// assert_eq!(size.get_mipmap(1), Size::new(128, 50));
    /// assert_eq!(size.get_mipmap(2), Size::new(64, 25));
    /// assert_eq!(size.get_mipmap(3), Size::new(32, 12));
    /// assert_eq!(size.get_mipmap(4), Size::new(16, 6));
    /// assert_eq!(size.get_mipmap(5), Size::new(8, 3));
    /// assert_eq!(size.get_mipmap(6), Size::new(4, 1));
    /// assert_eq!(size.get_mipmap(7), Size::new(2, 1));
    /// assert_eq!(size.get_mipmap(8), Size::new(1, 1));
    /// assert_eq!(size.get_mipmap(9), Size::new(1, 1));
    /// // From now on, it's always 1x1
    /// assert_eq!(size.get_mipmap(255), Size::new(1, 1));
    /// ```
    pub const fn get_mipmap(&self, level: u8) -> Self {
        Self {
            width: util::get_mipmap_size(self.width, level).get(),
            height: util::get_mipmap_size(self.height, level).get(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
impl Rect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub const fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    /// Returns `true` if this rectangle is completely within the bounds of the
    /// given size.
    ///
    /// This means that `self.x + self.width <= size.width` and
    /// `self.y + self.height <= size.height`.
    pub(crate) fn is_within_bounds(&self, size: Size) -> bool {
        // use u64 to prevent overflow
        let end_x = self.x as u64 + self.width as u64;
        let end_y = self.y as u64 + self.height as u64;
        end_x <= size.width as u64 && end_y <= size.height as u64
    }
}
