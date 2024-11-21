from process.processor import Processor
from utils.calculate import get_CNFL, get_CNFD, get_CNBD
from utils.common import show_image, save_image
from process.draw import draw_result_image


test_image_path = './assets/test.jpg'
segmenter_model_path = './models/nerve.onnx'
result_image_path = './assets/result.png'

p = Processor()
p.set_model_path(segmenter_model_path)
p.load_model()
p.load_image(test_image_path)
p.process()

print(f'CNFL: {get_CNFL(p)}\nCNFD: {get_CNFD(p)}\nCNBD: {get_CNBD(p)}')

result_image = draw_result_image(p.segments, p.nodes, p.image, 'none', 'bone')
show_image(result_image)
save_image(result_image, result_image_path)

