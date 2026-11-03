import unittest
from prepare_dataset import convert_box

class Boxes(unittest.TestCase):
    def test_coco_to_yolo(self):
        self.assertEqual(convert_box([10,20,20,40],100,200),[.2,.2,.2,.2])
    def test_clip(self):
        self.assertEqual(convert_box([-10,0,30,100],100,100),[.1,.5,.2,1])
    def test_invalid(self):
        for box in ([0,0,-1,4],[101,0,2,2],[float('nan'),0,1,1]):
            with self.assertRaises(ValueError):
                convert_box(box,100,100)

if __name__=='__main__':
    unittest.main()
