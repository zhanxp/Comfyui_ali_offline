import cv2
import numpy as np
from PIL import Image, ImageOps
from .utils import array2image


def add_alpha_channel(img):
    # 为jpg图像添加alpha通道
    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new


def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[y1:y2, x1:x2, 3] / 255.0
    alpha_jpg = 1 - alpha_png
    # 开始叠加
    for c in range(0, 3):
        jpg_img[y1:y2, x1:x2, c] = (alpha_jpg * jpg_img[y1:y2, x1:x2, c]) + (
            alpha_png * png_img[y1:y2, x1:x2, c]
        )
    return jpg_img


def Resize_cv2(img_jpg, img_png):
    newImg = cv2.resize(img_jpg, (img_png.shape[1], img_png.shape[0]))
    return newImg


def put_png_to_jpg(img_jpg, img_png):
    img_jpg = Resize_cv2(img_jpg, img_png)  # 变换大小跟透明图一样
    res_img = merge_img(img_jpg, img_png, 0, img_png.shape[0], 0, img_png.shape[1])
    # cv2.imwrite("output6.jpg", res_img)
    return res_img


def png_add_bg(img_png, r=255, g=255, b=255):
    w, h = img_png.size
    jpg = Image.new("RGB", size=(w, h), color=(r, g, b))
    image = put_png_to_jpg(np.array(jpg), np.array(img_png))
    return Image.fromarray(np.uint8(image)).convert("RGB")


# 白底黑色 mask 用于重绘
def png_to_mask(img, reverse=False, alpha=255):
    ac = 255
    bc = 0
    if reverse == True:
        ac = 0
        bc = 255
    image = array2image(img)
    # 创建一个新的RGBA图像，背景设置为白色
    new_image = Image.new("RGBA", image.size, (ac, ac, ac, 255))

    # 遍历图像的每个像素
    for x in range(image.width):
        for y in range(image.height):
            # 获取像素的RGBA值
            r, g, b, a = image.getpixel((x, y))
            # 判断像素的透明度
            if a > 0:
                # 不透明部分设置为黑色
                new_image.putpixel((x, y), (bc, bc, bc, 255))
            else:
                # 透明部分设置为白色
                new_image.putpixel((x, y), (ac, ac, ac, alpha))
    return new_image


def png_to_mask1(image):
    black_image = np.zeros_like(image)
    black_image[:, :, 3] = image[:, :, 3]
    white_image = np.ones_like(image) * 255
    white_image[:, :, :3] = image[:, :, :3]
    return put_png_to_jpg(white_image, black_image)


def png_to_mask2(image):
    black_image = np.zeros_like(image) + 255
    black_image[:, :, 3] = image[:, :, 3]
    white_image = np.ones_like(image) * 255
    white_image[:, :, :3] = image[:, :, :3]
    mask_image = put_png_to_jpg(white_image, black_image)
    mask_image = array2image(mask_image)
    mask_image = mask_image.convert("L")
    mask_image = Image.eval(mask_image, lambda x: 255 - x)
    mask_image = Image.eval(mask_image, lambda x: 0 if x < 127 else 255)
    return np.array(mask_image)


# 使用mask对图片进行抠图，mask为true false二维数组
def apply_mask(image, mask):
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = image  # 复制原始图像的 RGB 通道
    result[mask, 3] = 255  # 将掩膜中保留区域的透明通道设置为完全不透明
    return result


# 使用mask图片进行抠图，mask为true false二维数组同时进行高斯模糊
def apply_mask_width_blur(image, mask_svg, blur):
    blur = int(blur)
    blur = blur if blur % 2 == 1 else blur + 1

    # 创建一个新的 mask，用于添加新通道
    mask = np.zeros((mask_svg.shape[0], mask_svg.shape[1], 4), dtype=np.uint8)
    mask[:] = [255, 255, 255, 0]  # 将 mask 的初始透明度通道设为 0

    # 将不透明像素设为白色
    mask[mask_svg] = [255, 255, 255, 255]

    # 创建具有透明通道的图像
    image_with_alpha = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    image_with_alpha[:, :, :3] = image
    image_with_alpha[:, :, 3] = 255

    # 对 mask 进行高斯模糊
    if blur != 0:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    # 将模糊后的透明度通道应用到图像上
    image_with_alpha[:, :, 3] = mask[:, :, 3]

    return image_with_alpha


if __name__ == "__main__":
    put_png_to_jpg("example2.jpg", "output2.png")
