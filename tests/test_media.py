"""Local media validation tests. Tiny images here are test fixtures only."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from dashboard import media


class ProductMediaTests(unittest.TestCase):
    name = "Baseus H17 Encok 3.5mm Wired Earphone"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.catalog = self.root / "catalog.toml"
        self.patch_assets = patch.object(media, "ASSETS_DIR", self.root)
        self.patch_catalog = patch.object(media, "CATALOG_PATH", self.catalog)
        self.patch_assets.start()
        self.patch_catalog.start()
        self.addCleanup(self.patch_assets.stop)
        self.addCleanup(self.patch_catalog.stop)

    def write_catalog(self, path="", source="", name=None):
        self.catalog.write_text(
            '[products.baseus-h17]\n'
            f'name = "{name or self.name}"\n'
            f'image_path = "{path}"\n'
            f'source_url = "{source}"\n', encoding="utf-8",
        )

    def test_known_products_have_distinct_slugs_and_honest_missing_photos(self):
        self.write_catalog()
        photos = [media.product_media(name) for name in media.PRODUCT_MEDIA]
        self.assertEqual(len({photo.slug for photo in photos}), 5)
        self.assertTrue(all(photo.path is None for photo in photos))
        self.assertTrue(all(photo.source_url is None for photo in photos))
        self.assertTrue(all(photo.image_data_uri is None for photo in photos))

    def test_valid_photo_exposes_local_path_and_https_provenance(self):
        Image.new("RGB", (3, 3), color="white").save(self.root / "fixture.png")
        self.write_catalog("fixture.png", "https://www.baseus.com/products/verified-source")
        photo = media.product_media(self.name)
        self.assertEqual(photo.path, self.root / "fixture.png")
        self.assertEqual(photo.source_url, "https://www.baseus.com/products/verified-source")
        self.assertTrue(photo.image_data_uri.startswith("data:image/png;base64,"))
        self.assertEqual(photo.category, "Wired earphones")
        self.assertFalse(photo.warnings)

    def test_corrupt_and_missing_images_have_no_photo_or_source(self):
        (self.root / "corrupt.jpg").write_bytes(b"not a JPEG")
        for filename in ["corrupt.jpg", "missing.png"]:
            with self.subTest(filename=filename):
                self.write_catalog(filename, "https://www.baseus.com/")
                photo = media.product_media(self.name)
                self.assertIsNone(photo.path)
                self.assertIsNone(photo.source_url)
                self.assertTrue(photo.warnings)

    def test_source_link_requires_https_and_no_credentials(self):
        Image.new("RGB", (3, 3)).save(self.root / "fixture.jpg")
        for source in ["http://www.baseus.com/", "https://user:secret@www.baseus.com/", "javascript:alert(1)"]:
            with self.subTest(source=source):
                self.write_catalog("fixture.jpg", source)
                photo = media.product_media(self.name)
                self.assertIsNotNone(photo.path)
                self.assertIsNone(photo.source_url)
                self.assertTrue(photo.warnings)

    def test_escaping_path_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            external = Path(other) / "external.png"
            Image.new("RGB", (3, 3)).save(external)
            (self.root / "linked.png").symlink_to(external)
            for filename in [str(external), "linked.png", "../external.png"]:
                with self.subTest(filename=filename):
                    self.write_catalog(filename)
                    photo = media.product_media(self.name)
                    self.assertIsNone(photo.path)
                    self.assertTrue(photo.warnings)

    def test_bad_catalog_and_wrong_product_are_safe(self):
        self.catalog.write_text("not valid = [", encoding="utf-8")
        self.assertIsNone(media.product_media(self.name).path)
        self.write_catalog(name="Another product")
        self.assertTrue(media.product_media(self.name).warnings)
        photo = media.product_media("Unknown product")
        self.assertIsNone(photo.path)
        self.assertEqual(photo.category, "Earphones")

    def test_svg_is_not_accepted(self):
        (self.root / "fixture.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        self.write_catalog("fixture.svg")
        self.assertIsNone(media.product_media(self.name).path)


if __name__ == "__main__":
    unittest.main()
