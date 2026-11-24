//! Floating-point frame geometry shared by framing producers (effects)
//! and consumers (the crop stage and resize kernels).

/// Axis-aligned rectangle in pixel coordinates of some frame, with
/// sub-pixel precision so a slowly moving crop does not step by whole
/// pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width (positive).
    pub width: f32,
    /// Height (positive).
    pub height: f32,
}

impl Region {
    /// The whole `width × height` frame.
    #[must_use]
    pub fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
        }
    }

    /// Largest rectangle with aspect ratio `aspect` (width / height)
    /// centred inside a `width × height` frame.
    #[must_use]
    pub fn centered_with_aspect(width: u32, height: u32, aspect: f32) -> Self {
        let (fw, fh) = (width as f32, height as f32);
        let (w, h) = if fw / fh > aspect {
            (fh * aspect, fh)
        } else {
            (fw, fw / aspect)
        };
        Self {
            x: (fw - w) * 0.5,
            y: (fh - h) * 0.5,
            width: w,
            height: h,
        }
    }

    /// Horizontal centre.
    #[must_use]
    pub fn center_x(&self) -> f32 {
        self.x + self.width * 0.5
    }

    /// Vertical centre.
    #[must_use]
    pub fn center_y(&self) -> f32 {
        self.y + self.height * 0.5
    }

    /// Interpolate every edge from `self` (`t = 0`) to `other` (`t = 1`).
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let mix = |a: f32, b: f32| a + (b - a) * t;
        Self {
            x: mix(self.x, other.x),
            y: mix(self.y, other.y),
            width: mix(self.width, other.width),
            height: mix(self.height, other.height),
        }
    }

    /// Whether every edge of `self` is within `tolerance` pixels of the
    /// corresponding edge of `other`.
    #[must_use]
    pub fn approx_eq(&self, other: &Self, tolerance: f32) -> bool {
        (self.x - other.x).abs() <= tolerance
            && (self.y - other.y).abs() <= tolerance
            && (self.width - other.width).abs() <= tolerance
            && (self.height - other.height).abs() <= tolerance
    }

    /// Map a point given in `self`'s local coordinates, where `self` is
    /// shown as a `local_w × local_h` frame, back to the coordinates
    /// `self` is expressed in.
    #[must_use]
    pub fn map_from_local(&self, x: f32, y: f32, local_w: u32, local_h: u32) -> (f32, f32) {
        (
            self.x + x * self.width / local_w as f32,
            self.y + y * self.height / local_h as f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_with_aspect_crops_the_longer_axis() {
        // 4:3 frame, 16:9 view → full width, letterbox-free band.
        let r = Region::centered_with_aspect(1280, 960, 16.0 / 9.0);
        assert!((r.width - 1280.0).abs() < 1e-3);
        assert!((r.height - 720.0).abs() < 1e-3);
        assert!((r.y - 120.0).abs() < 1e-3);
        // Same aspect → whole frame.
        let r = Region::centered_with_aspect(1280, 720, 16.0 / 9.0);
        assert!(r.approx_eq(&Region::full(1280, 720), 1e-3));
    }

    #[test]
    fn lerp_hits_both_ends() {
        let a = Region::full(100, 100);
        let b = Region {
            x: 10.0,
            y: 20.0,
            width: 50.0,
            height: 40.0,
        };
        assert_eq!(a.lerp(&b, 0.0), a);
        assert_eq!(a.lerp(&b, 1.0), b);
    }

    #[test]
    fn map_from_local_scales_and_offsets() {
        let r = Region {
            x: 100.0,
            y: 50.0,
            width: 640.0,
            height: 360.0,
        };
        let (x, y) = r.map_from_local(400.0, 224.0, 800, 448);
        assert!((x - 420.0).abs() < 1e-3);
        assert!((y - 230.0).abs() < 1e-3);
    }
}
