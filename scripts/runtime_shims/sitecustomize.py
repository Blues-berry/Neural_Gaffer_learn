try:
    import kornia.geometry.transform as _kornia_transform
    from kornia.geometry.transform.pyramid import build_pyramid as _build_pyramid

    if not hasattr(_kornia_transform, "build_laplacian_pyramid"):
        def build_laplacian_pyramid(input, max_level, border_type="reflect", align_corners=False):
            # Compatibility shim for newer diffusers importing a symbol that is
            # absent in the installed kornia version. The official Neural Gaffer
            # demo path does not use frequency-decoupled guidance, so a Gaussian
            # pyramid fallback is sufficient to satisfy the import safely.
            return _build_pyramid(
                input,
                max_level=max_level,
                border_type=border_type,
                align_corners=align_corners,
            )

        _kornia_transform.build_laplacian_pyramid = build_laplacian_pyramid
except Exception:
    pass
