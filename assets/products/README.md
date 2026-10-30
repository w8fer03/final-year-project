# Product photos

No verified product photos have been supplied. Requests to the Baseus/Shopee sites were blocked in the current environment. The catalog therefore contains empty image paths, and the landing page displays the honest **Product photo not available** state. It does not fetch third-party substitutes or hotlink remote images.

To add a photo:

1. Obtain an actual photo for the exact named product from its official Baseus or Shopee product page using permitted access. Confirm the product and any variant shown match the catalog entry. Retain the original source and any required attribution; check permission to use the image.
2. Save the JPEG, PNG or WebP file in this directory, for example `baseus-h17.jpg` for the H17 entry. Subdirectories beneath this directory are supported. Do not add unrelated stock photos, generated product images or guessed filenames representing another product.
3. Set that entry's `image_path` in `catalog.toml` to the local relative filename and `source_url` to the real HTTPS product page from which the photo was obtained. Leave the source blank if it is not known; do not invent a link.
4. Refresh the dashboard. Images are read locally; no image requests or model downloads are performed. A source link is exposed only when a real local image has passed validation and the source is a valid HTTPS URL.

The stable slugs are `baseus-h17`, `baseus-c17`, `baseus-cz11-hz11`, `baseus-wm01` and `baseus-h19`. Public media metadata contains product names and categories, not underlying Shopee listing identifiers.

Invalid TOML, wrong product names, missing files, paths escaping this directory, SVGs and corrupt images resolve to the same empty image state. `product_media(name).warnings` provides owner diagnostics without exposing raw listing identifiers. The loader checks actual image content with Pillow, including complete decoding; changing a file extension alone does not make an unsupported file acceptable. It never edits the original files.

`ProductMedia` exposes `path`, `alt`, `source_url`, `category`, `slug`, `warnings`, `mime_type` and an optional `image_data_uri` property. HTML renderers must escape text and source URLs; Streamlit can display the validated local `path` directly.
