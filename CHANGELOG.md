# Changelog

## [1.0.1]

### Added

- Add DA3 camera-token-based cross-view attention, multi-view inputs, and the
  optional image-conditioned GSDPT Gaussian splat head (enabled by default for
  `vit_g`).
- Add shared checkpoint-remapping utilities and broaden tests for model
  inference, wrappers, and Gaussian outputs.
- Add project architecture and model documentation, plus automated GitHub
  Pages documentation deployment.

### Changed

- Align DINOv2, DepthPro, and other model implementations around shared
  conventions and clearer module boundaries; improve DINOv2 input validation
  and type safety.
- Correct the citation author metadata and update project links to the current
  repository.
- Make pretrained dependencies optional, remove an unused dependency, and
  defer sparse 3D backend detection until it is needed.
- Update package documentation to match the current codebase.

### Fixed

- Improve DepthPro correctness and consistency, including inference behavior
  and the `VisionModule` forward path.
- Consolidate DA3/CDM checkpoint-key remapping and improve handling of optional
  model attributes.
- Replace non-ASCII characters in docstrings for broader tooling compatibility.
