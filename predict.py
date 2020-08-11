from ssd import SSD
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

ssd = SSD()

while True:
    img = 'img/' + input('Input image filename:') + '.jpg'
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = ssd.detect_image(image)

        r_image.save('result/{}_result.jpg'.format(img.replace('img', '').replace('.jpg', '')))
        print('Detect result is saved.')
        r_image.close()
        break