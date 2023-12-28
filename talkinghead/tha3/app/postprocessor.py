"""Smoke and mirrors. Glitch artistry. Pixel-space postprocessing effects.

These effects work in linear intensity space, before gamma correction.
"""

import math
from typing import List, Optional

import torch
import torchvision

# TODO:
#   - make configurable (ask Cohee1207 about the preferable way)
#   - create the base meshgrid for image coordinates only once (not once per effect per frame, as now)

class Postprocessor:
    def __init__(self, device: torch.device):
        self.device = device
        self.frame_no = 0

    def configure(self):
        # TODO: take in a dict of config: {effect0: {key0: value0, ...}, ...}
        pass

    def render_into(self, image):
        """Apply postprocess chain, modifying `image`."""
        # This also documents the correct ordering of the filters.

        # physical input signal
        self.apply_bloom(image)  # Fake HDR; only makes sense with dark-ish backgrounds!

        # video camera
        self.apply_chromatic_aberration(image)
        self.apply_vignetting(image)

        # scifi hologram
        self.apply_translucency(image)
        self.apply_alphanoise(image, magnitude=0.1, sigma=0.0)

        # # lo-fi analog video
        # apply_analog_lowres(output_image)
        # apply_alphanoise(output_image, magnitude=0.2, sigma=2.0)
        # apply_analog_badhsync(output_image)
        # # apply_analog_vhsglitches(output_image)
        # apply_analog_vhstracking(output_image)

        # CRT TV output
        # apply_desaturate(output_image)
        self.apply_banding(image)
        self.apply_scanlines(image)

        # All done!
        self.frame_complete()

    def frame_complete(self):
        """Mark frame as complete, incrementing the frame counter."""
        self.frame_no += 1

    # --------------------------------------------------------------------------------

    def apply_bloom(self, image: torch.tensor, luma_threshold: float = 0.8, hdr_exposure: float = 0.7) -> None:
        """Bloom effect (fake HDR). Popular in early 2000s anime.

        Bright parts of the image bleed light into their surroundings, creating enhanced perceived contrast.
        Only makes sense when the character is rendered on a dark-ish background.

        `luma_threshold`: How bright is bright. 0.0 is full black, 1.0 is full white.
        `hdr_exposure`: Controls the overall brightness of the output. Like in photography,
                        higher exposure means brighter image (saturating toward white).
        """
        # There are online tutorials for how to create this effect, see e.g.:
        #   https://learnopengl.com/Advanced-Lighting/Bloom

        # Find the bright parts.
        Y = 0.2126 * image[0, :, :] + 0.7152 * image[1, :, :] + 0.0722 * image[2, :, :]  # HDTV luminance (ITU-R Rec. 709)
        mask = torch.ge(Y, luma_threshold)  # [h, w]

        # Make a copy of the image with just the bright parts.
        mask = torch.unsqueeze(mask, 0)  # -> [1, h, w]
        brights = image * mask  # [c, h, w]

        # Blur the bright parts. Two-pass blur to save compute, since we need a very large blur kernel.
        # It seems that in Torch, one large 1D blur is faster than looping with a smaller one.
        #
        # Although everything else in Torch takes (height, width), kernel size is given as (size_x, size_y);
        # see `gaussian_blur_image` in https://pytorch.org/vision/main/_modules/torchvision/transforms/v2/functional/_misc.html
        # for a hint (the part where it computes the padding).
        brights = torchvision.transforms.GaussianBlur((21, 1), sigma=7.0)(brights)  # blur along x
        brights = torchvision.transforms.GaussianBlur((1, 21), sigma=7.0)(brights)  # blur along y

        # Additively blend the images. Note we are working in linear intensity space, and we will now go over 1.0 intensity.
        image.add_(brights)

        # We now have a fake HDR image. Tonemap it back to LDR.
        image[:3, :, :] = 1.0 - torch.exp(-image[:3, :, :] * hdr_exposure)  # RGB: tonemap
        image[3, :, :] = torch.maximum(image[3, :, :], brights[3, :, :])  # alpha: max-combine
        torch.clamp_(image, min=0.0, max=1.0)

    def apply_scanlines(self, image: torch.tensor, field: int = 0, dynamic: bool = True) -> None:
        """CRT TV like scanlines.

        `field`: Which CRT field is dimmed at the first frame. 0 = top, 1 = bottom.
        `dynamic`: If `True`, the dimmed field will alternate each frame (top, bottom, top, bottom, ...)
                   for a more authentic CRT look (like Phosphor deinterlacer in VLC).
        """
        if dynamic:
            start = (field + self.frame_no) % 2
        else:
            start = field
        # We should ideally modify just the Y channel in YUV space, but modifying the alpha instead looks alright.
        image[3, start::2, :].mul_(0.5)

    def apply_alphanoise(self, image: torch.tensor, magnitude: float = 0.1, sigma: float = 0.0) -> None:
        """Dynamic noise to alpha channel. A cheap alternative to luma noise.

        `magnitude`: How much noise to apply. 0 is off, 1 is as much noise as possible.

        `sigma`: If nonzero, apply a Gaussian blur to the noise, thus reducing its spatial frequency
                 (i.e. making larger and smoother "noise blobs").

                 The blur kernel size is fixed to 5, so `sigma = 1.0` is the largest that will be
                 somewhat accurate. Nevertheless, `sigma = 2.0` looks acceptable, too, producing
                 square blobs.

        Suggested settings:
            Scifi hologram:   magnitude=0.1, sigma=0.0
            Analog VHS tape:  magnitude=0.2, sigma=2.0
        """
        c, h, w = image.shape
        noise_image = torch.rand(h, w, device=self.device, dtype=image.dtype)
        if sigma > 0.0:
            noise_image = noise_image.unsqueeze(0)  # [h, w] -> [c, h, w] (where c=1)
            noise_image = torchvision.transforms.GaussianBlur((5, 5), sigma=sigma)(noise_image)
            noise_image = noise_image.squeeze(0)  # -> [h, w]
        base_magnitude = 1.0 - magnitude
        image[3, :, :].mul_(base_magnitude + magnitude * noise_image)

    def apply_translucency(self, image: torch.tensor, alpha: float = 0.9) -> None:
        """A simple translucency filter for a hologram look.

        Multiplicatively adjusts the alpha channel.
        """
        image[3, :, :].mul_(alpha)

    def apply_banding(self, image: torch.tensor, strength: float = 0.4, density: float = 2.0, speed: float = 16.0) -> None:
        """Bad analog video signal, with traveling brighter and darker bands.

        This simulates a CRT display as it looks when filmed on video without syncing.

        `strength`: maximum brightness factor
        `density`: how many banding cycles per full image height
        `speed`: band movement, in pixels per frame
        """
        c, h, w = image.shape
        yy = torch.linspace(0, math.pi, h, dtype=image.dtype, device=self.device)

        # Animation
        cycle_pos = (self.frame_no / h) * speed
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part
        cycle_pos = 1.0 - cycle_pos  # -> motion from top toward bottom

        band_effect = torch.sin(density * yy + cycle_pos * math.pi)**2  # [h]
        band_effect = torch.unsqueeze(band_effect, 0)  # -> [1, h] = [c, h]
        band_effect = torch.unsqueeze(band_effect, 2)  # -> [1, h, 1] = [c, h, w]
        image[:3, :, :].mul_(1.0 + strength * band_effect)
        torch.clamp_(image, min=0.0, max=1.0)

    def apply_analog_lowres(self, image: torch.tensor, kernel_size: int = 5, sigma: float = 0.75) -> None:
        """Low-resolution analog video signal, simulated by blurring.

        `kernel_size`: size of the Gaussian blur kernel, in pixels.
        `sigma`: standard deviation of the Gaussian blur kernel, in pixels.

        Ideally, `kernel_size` should be `2 * (3 * sigma) + 1`, so that the kernel
        reaches its "3 sigma" (99.7% mass) point where the finitely sized kernel
        cuts the tail. "2 sigma" (95% mass) is also acceptable, to save some compute.

        The default settings create a slight blur without destroying much detail.
        """
        image[:, :, :] = torchvision.transforms.GaussianBlur((kernel_size, kernel_size), sigma=sigma)(image)

    def apply_analog_badhsync(self, image: torch.tensor, speed: float = 8.0,
                              amplitude1: float = 0.001, density1: float = 4.0,
                              amplitude2: Optional[float] = 0.001, density2: Optional[float] = 13.0,
                              amplitude3: Optional[float] = 0.001, density3: Optional[float] = 27.0) -> None:
        """Analog video signal with fluctuating hsync.

        We superpose three waves with different densities (1 / cycle length)
        to make the pattern look more irregular.

        E.g. density of 2.0 means that two full waves fit into the image height.

        Amplitudes are given in units where the height and width of the image
        are both 2.0.
        """
        c, h, w = image.shape

        # Seems the deformation geometry must be float32 no matter the image data type.
        yy = torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=self.device)
        xx = torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=self.device)
        meshy, meshx = torch.meshgrid((yy, xx), indexing="ij")

        # Animation
        cycle_pos = (self.frame_no / h) * speed
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part
        cycle_pos = 1.0 - cycle_pos  # -> motion from top toward bottom
        cycle_pos *= 2.0  # full cycle = 2 units

        # Deformation
        meshx = meshx + amplitude1 * torch.sin((density1 * (meshy + cycle_pos)) * math.pi)
        if amplitude2 and density2:
            meshx = meshx + amplitude2 * torch.sin((density2 * (meshy + cycle_pos)) * math.pi)
        if amplitude3 and density3:
            meshx = meshx + amplitude3 * torch.sin((density3 * (meshy + cycle_pos)) * math.pi)

        grid = torch.stack((meshx, meshy), 2)
        grid = grid.unsqueeze(0)  # batch of one
        image_batch = image.unsqueeze(0)  # batch of one -> [1, c, h, w]
        warped = torch.nn.functional.grid_sample(image_batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.squeeze(0)  # [1, c, h, w] -> [c, h, w]
        image[:, :, :] = warped

    def _vhs_noise(self, image: torch.tensor, height: int) -> torch.tensor:
        """Generate a horizontal band of noise that looks as if it came from a blank VHS tape.

        `height`: desired height of noise band, in pixels.

        Output is a tensor of shape `[1, height, w]`, where `w` is the postprocessor's input image width.
        """
        c, h, w = image.shape
        # This looks best if we randomize the alpha channel, too.
        noise_image = torch.rand(height, w, device=self.device, dtype=image.dtype).unsqueeze(0)  # [1, h, w]
        # Real VHS noise has horizontal runs of the same color, and the transitions between black and white are smooth.
        noise_image = torchvision.transforms.GaussianBlur((5, 1), sigma=2.0)(noise_image)
        return noise_image

    def apply_analog_vhstracking(self, image: torch.tensor, base_offset: float = 0.03, max_dynamic_offset: float = 0.01, speed: float = 2.5) -> None:
        """1980s VHS tape with bad tracking.

        Image floats up and down, and a band of black and white noise appears at the bottom.

        Units like in `apply_analog_badhsync`.
        """
        c, h, w = image.shape

        # Seems the deformation geometry must be float32 no matter the image data type.
        yy = torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=self.device)
        xx = torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=self.device)
        meshy, meshx = torch.meshgrid((yy, xx), indexing="ij")

        # Animation
        cycle_pos = (self.frame_no / h) * speed
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part
        cycle_pos *= 2.0  # full cycle = 2 units

        # Deformation - move image up/down
        yoffs = max_dynamic_offset * math.sin(cycle_pos * math.pi)
        meshy = meshy + yoffs

        grid = torch.stack((meshx, meshy), 2)
        grid = grid.unsqueeze(0)  # batch of one
        image_batch = image.unsqueeze(0)  # batch of one -> [1, c, h, w]
        warped = torch.nn.functional.grid_sample(image_batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.squeeze(0)  # [1, c, h, w] -> [c, h, w]
        image[:, :, :] = warped

        # Noise from bad VHS tracking at bottom
        yoffs_pixels = int((yoffs / 2.0) * 512.0)
        base_offset_pixels = int((base_offset / 2.0) * 512.0)
        noise_pixels = yoffs_pixels + base_offset_pixels
        if noise_pixels > 0:
            image[:, -noise_pixels:, :] = self._vhs_noise(image, height=noise_pixels)
            # # Fade out toward left/right, since the character does not take up the full width.
            # # Works, but fails at reaching the iconic VHS look.
            # x = torch.linspace(0, math.pi, IMAGE_HEIGHT, dtype=image.dtype, device=self.device)
            # fade = torch.sin(x)**2  # [w]
            # fade = fade.unsqueeze(0)  # [1, w]
            # image[3, -noise_pixels:, :] = fade

    def apply_analog_vhsglitches(self, image: torch.tensor, strength: float = 0.1, unboost: float = 4.0,
                                 max_glitches: int = 3, min_glitch_height: int = 3, max_glitch_height: int = 6) -> None:
        """Damaged 1980s VHS video tape, with transient (per-frame) glitching lines.

        This leaves the alpha channel alone, so the effect only affects parts that already show something.
        This is an artistic interpretation that makes the effect less distracting when used with RGBA data.

        `strength`: How much to blend in noise.
        `unboost`: Use this to adjust the probability profile for the appearance of glitches.
                   The higher `unboost` is, the less probable it is for glitches to appear at all,
                   and there will be fewer of them (in the same video frame) when they do appear.
        `max_glitches`: Maximum number of glitches in the video frame.
        `min_glitch_height`, `max_glitch_height`: in pixels. The height is randomized separately for each glitch.
        """
        c, h, w = image.shape
        n_glitches = torch.rand(1, device="cpu")**unboost  # higher probability of having none or few glitching lines
        n_glitches = int(max_glitches * n_glitches[0])
        if not n_glitches:
            return
        glitch_start_lines = torch.rand(n_glitches, device="cpu")
        glitch_start_lines = [int((h - (max_glitch_height - 1)) * x) for x in glitch_start_lines]
        for line in glitch_start_lines:
            glitch_height = torch.rand(1, device="cpu")
            glitch_height = int(min_glitch_height + (max_glitch_height - min_glitch_height) * glitch_height[0])
            noise_image = self._vhs_noise(image, height=glitch_height)
            # Apply glitch to RGB only, so fully transparent parts stay transparent (important to make the effect less distracting).
            image[:3, line:(line + glitch_height), :] = (1.0 - strength) * image[:3, line:(line + glitch_height), :] + strength * noise_image

    def apply_chromatic_aberration(self, image: torch.tensor, transverse_sigma: float = 0.5, axial_scale: float = 0.005) -> None:
        """Simulate the two types of chromatic aberration in a camera lens.

        Like everything else here, this is of course made of smoke and mirrors. We simulate the axial effect
        (index of refraction varying w.r.t. wavelength) by geometrically scaling the RGB channels individually,
        and the transverse effect (focal distance varying w.r.t. wavelength) by a gaussian blur.

        Note that in a real lens:
          - Axial CA is typical at long focal lengths (e.g. tele/zoom lens)
          - Axial CA increases at high F-stops (low depth of field, i.e. sharp focus at all distances)
          - Transverse CA is typical at short focal lengths (e.g. macro lens)

        However, in an RGB postproc effect, it is useful to apply both together, to help hide the clear-cut red/blue bands
        resulting from the different geometric scalings of just three wavelengths (instead of a continuous spectrum, like
        a scene lit with natural light would have).

        See:
            https://en.wikipedia.org/wiki/Chromatic_aberration
        """
        c, h, w = image.shape

        # Seems the deformation geometry must be float32 no matter the image data type.
        yy = torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=self.device)
        xx = torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=self.device)
        meshy, meshx = torch.meshgrid((yy, xx), indexing="ij")

        # Axial: Shrink R (deflected less), pass G through (lens reference wavelength), enlarge B (deflected more).
        grid_R = torch.stack((meshx * (1.0 + axial_scale), meshy * (1.0 + axial_scale)), 2)
        grid_R = grid_R.unsqueeze(0)
        grid_B = torch.stack((meshx * (1.0 - axial_scale), meshy * (1.0 - axial_scale)), 2)
        grid_B = grid_B.unsqueeze(0)

        image_batch_R = image[0, :, :].unsqueeze(0).unsqueeze(0)  # [h, w] -> [c, h, w] -> [n, c, h, w]
        warped_R = torch.nn.functional.grid_sample(image_batch_R, grid_R, mode="bilinear", padding_mode="border", align_corners=False)
        warped_R = warped_R.squeeze(0)  # [1, c, h, w] -> [c, h, w]
        image_batch_B = image[2, :, :].unsqueeze(0).unsqueeze(0)
        warped_B = torch.nn.functional.grid_sample(image_batch_B, grid_B, mode="bilinear", padding_mode="border", align_corners=False)
        warped_B = warped_B.squeeze(0)  # [1, c, h, w] -> [c, h, w]

        # Transverse (blur to simulate wrong focal distance for R and B)
        warped_R[:, :, :] = torchvision.transforms.GaussianBlur((5, 5), sigma=transverse_sigma)(warped_R)
        warped_B[:, :, :] = torchvision.transforms.GaussianBlur((5, 5), sigma=transverse_sigma)(warped_B)

        # Alpha channel: treat similarly to each of R,G,B and average the three resulting alpha channels
        image_batch_A = image[3, :, :].unsqueeze(0).unsqueeze(0)
        warped_A1 = torch.nn.functional.grid_sample(image_batch_A, grid_R, mode="bilinear", padding_mode="border", align_corners=False)
        warped_A1[:, :, :] = torchvision.transforms.GaussianBlur((5, 5), sigma=transverse_sigma)(warped_A1)
        warped_A2 = torch.nn.functional.grid_sample(image_batch_A, grid_B, mode="bilinear", padding_mode="border", align_corners=False)
        warped_A2[:, :, :] = torchvision.transforms.GaussianBlur((5, 5), sigma=transverse_sigma)(warped_A2)
        averaged_alpha = (warped_A1 + image[3, :, :] + warped_A2) / 3.0

        image[0, :, :] = warped_R
        # image[1, :, :] passed through as-is
        image[2, :, :] = warped_B
        image[3, :, :] = averaged_alpha

    def apply_vignetting(self, image: torch.tensor, strength: float = 0.42) -> None:
        """Simulate vignetting (less light hitting the corners of a film frame or CCD sensor).

        The profile used here is [cos(strength * d * pi)]**2, where `d` is the distance
        from the center, scaled such that `d = 1.0` is reached at the corners.
        Thus, at the midpoints of the frame edges, `d = 1 / sqrt(2) ~ 0.707`.
        """
        c, h, w = image.shape

        # Seems the deformation geometry must be float32 no matter the image data type.
        yy = torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=self.device)
        xx = torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=self.device)
        meshy, meshx = torch.meshgrid((yy, xx), indexing="ij")

        euclidean_distance_from_center = (meshy**2 + meshx**2)**0.5 / 2**0.5  # [h, w]

        brightness = torch.cos(strength * euclidean_distance_from_center * math.pi)**2  # [h, w]
        brightness = torch.unsqueeze(brightness, 0)  # -> [1, h, w]
        image[:3, :, :] *= brightness

    def _rgb_to_hue(rgb: List[float]) -> float:
        """Convert an RGB color to an HSL hue, for use as `bandpass_hue` in `apply_desaturate`.

        This uses a cartesian-to-polar approximation of the HSL representation,
        which is fine for hue detection, but should not be taken as an authoritative
        H component of an accurate RGB->HSL conversion.
        """
        R, G, B = rgb
        alpha = 0.5 * (2.0 * R - G - B)
        beta = 3.0**0.5 / 2.0 * (G - B)
        hue = math.atan2(beta, alpha) / (2.0 * math.pi)  # note atan2(0, 0) := 0
        return hue

    # This filter is adapted from an old GLSL code I made for Panda3D 1.8 back in 2014.
    def apply_desaturate(self, image: torch.tensor,
                         strength: float = 1.0,
                         tint_rgb: List[float] = [1.0, 1.0, 1.0],
                         bandpass_reference_rgb: List[float] = [1.0, 0.0, 0.0], bandpass_q: float = 0.0) -> None:
        """Desaturation with bells and whistles.

        Does not touch the alpha channel.

        `strength`: Overall blending strength of the filter (0 is off, 1 is fully applied).

        `tint_rgb`: Color to multiplicatively tint the image with. Applied after desaturation.

                    Some example tint values:
                        Green monochrome computer monitor: [0.5, 1.0, 0.5]
                        Amber monochrome computer monitor: [1.0, 0.5, 0.2]
                        Sepia effect:                      [0.8039, 0.6588, 0.5098]
                        No tint (off; default):            [1.0, 1.0, 1.0]

        `bandpass_reference_rgb`: Reference color for hue to let through the bandpass.
                                  Use this to let e.g. red things bypass the desaturation.
                                  The hue is extracted automatically from the given color.

        `bandpass_q`: Hue bandpass band half-width, in (0, 1]. Hues farther away from `bandpass_hue`
                      than `bandpass_q` will be fully desaturated. The opposite colors on the color
                      circle are defined as having the largest possible hue difference, 1.0.

                      The shape of the filter is a quadratic spike centered on the reference hue,
                      and smoothly decaying to zero at `bandpass_q` away from the center.

                      The special value 0 (default) switches the hue bandpass code off,
                      saving some compute.
        """
        R = image[0, :, :]
        G = image[1, :, :]
        B = image[2, :, :]
        if bandpass_q > 0.0:  # hue bandpass enabled?
            # Calculate hue of each pixel, using a cartesian-to-polar approximation of the HSL representation.
            # An approximation is fine here, because we only use this for a hue detector.
            # This is faster and requires less branching than the exact hexagonal representation.
            desat_alpha = 0.5 * (2.0 * R - G - B)
            desat_beta = 3.0**0.5 / 2.0 * (G - B)
            desat_hue = torch.atan2(desat_beta, desat_alpha) / (2.0 * math.pi)  # note atan2(0, 0) := 0
            desat_hue = desat_hue + torch.where(torch.lt(desat_hue, 0.0), 0.5, 0.0)  # convert from `[-0.5, 0.5)` to `[0, 1)`
            # -> [h, w]

            # Determine whether to keep this pixel or desaturate (and by how much).
            #
            # Calculate distance of each pixel from reference hue, accounting for wrap-around.
            bandpass_hue = self._rgb_to_hue(bandpass_reference_rgb)
            desat_temp1 = torch.abs(desat_hue - bandpass_hue)
            desat_temp2 = torch.abs((desat_hue + 1.0) - bandpass_hue)
            desat_temp3 = torch.abs(desat_hue - (bandpass_hue + 1.0))
            desat_hue_distance = 2.0 * torch.minimum(torch.minimum(desat_temp1, desat_temp2),
                                                     desat_temp3)  # [0, 0.5] -> [0, 1]
            # -> [h, w]

            # - Pixels with their hue at least `bandpass_q` away from `bandpass_hue` are fully desaturated.
            # - As distance falls below `bandpass_q`, a blend starts very gradually.
            # - As the hue difference approaches zero, the pixel is fully passed through.
            # - The 1.0 - ... together with the square makes a sharp spike at the reference hue.
            desat_diff2 = (1.0 - torch.clamp(desat_hue_distance / bandpass_q, max=1.0))**2
            strength_field = strength * (1.0 - desat_diff2)  # [h, w]
        else:
            strength_field = strength  # just a scalar!

        # Desaturate, then apply tint
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B  # HDTV luminance (ITU-R Rec. 709)  -> [h, w]
        Y = Y.unsqueeze(0)  # -> [1, h, w]
        tint_color = torch.tensor(tint_rgb, device=self.device, dtype=image.dtype).unsqueeze(1).unsqueeze(2)  # [c, 1, 1]
        tinted_desat_image = Y * tint_color  # -> [c, h, w]

        # Final blend
        image[:3, :, :] = (1.0 - strength_field) * image[:3, :, :] + strength_field * tinted_desat_image
