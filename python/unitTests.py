import unittest


class TestSum(unittest.TestCase):

    def test_cv_install(self):
        try:
            import cv2
        except ImportError:
            self.fail("Failed to import cv2. Check OpenCV installation.")


if __name__ == '__main__':
    unittest.main()