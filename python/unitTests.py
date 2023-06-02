import unittest


class TestSum(unittest.TestCase):

    def test_cv_install(self):
        try:
            import cv2
        except ImportError:
            self.fail("Failed to import cv2. Check OpenCV installation.")

    def test_false(self):
        self.assertEqual(1, 0, "1 isn't equal to 0")
        
if __name__ == '__main__':
    unittest.main()