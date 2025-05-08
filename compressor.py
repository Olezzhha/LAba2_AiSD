import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DC_HUFFMAN_TABLE = {
    0: '00',
    1: '010',
    2: '011',
    3: '100',
    4: '101',
    5: '110',
    6: '1110',
    7: '11110',
    8: '111110',
    9: '1111110',
    10: '11111110',
    11: '111111110',
}
DC_HUFFMAN_DECODE = {v: k for k, v in DC_HUFFMAN_TABLE.items()}
AC_HUFFMAN_TABLE = {
    (0, 0): '1010',      # EOB
    (0, 1): '00', (0, 2): '01', (0, 3): '100', (0, 4): '1011',
    (0, 5): '11010', (0, 6): '1111000', (0, 7): '11111000', (0, 8): '1111110110',
    (0, 9): '1111111110000010', (0,10): '1111111110000011',

    (1, 1): '1100', (1, 2): '11011', (1, 3): '1111001', (1, 4): '111110110',
    (1, 5): '11111110110', (1, 6): '1111111110000100', (1, 7): '1111111110000101',
    (1, 8): '1111111110000110', (1, 9): '1111111110000111', (1,10): '1111111110001000',

    (2, 1): '11100', (2, 2): '11111001', (2, 3): '1111110111',
    (2, 4): '111111110100', (2, 5): '1111111110001001', (2, 6): '1111111110001010',
    (2, 7): '1111111110001011', (2, 8): '1111111110001100', (2, 9): '1111111110001101',
    (2,10): '1111111110001110',

    (3, 1): '111010', (3, 2): '111110111', (3, 3): '111111110101',
    (3, 4): '1111111110001111', (3, 5): '1111111110010000', (3, 6): '1111111110010001',
    (3, 7): '1111111110010010', (3, 8): '1111111110010011', (3, 9): '1111111110010100',
    (3,10): '1111111110010101',

    (4, 1): '111011', (4, 2): '1111111000', (4, 3): '1111111110010110',
    (4, 4): '1111111110010111', (4, 5): '1111111110011000', (4, 6): '1111111110011001',
    (4, 7): '1111111110011010', (4, 8): '1111111110011011', (4, 9): '1111111110011100',
    (4,10): '1111111110011101',

    (5, 1): '1111010', (5, 2): '11111110111', (5, 3): '1111111110011110',
    (5, 4): '1111111110011111', (5, 5): '1111111110100000', (5, 6): '1111111110100001',
    (5, 7): '1111111110100010', (5, 8): '1111111110100011', (5, 9): '1111111110100100',
    (5,10): '1111111110100101',

    (6, 1): '1111011', (6, 2): '111111110110', (6, 3): '1111111110100110',
    (6, 4): '1111111110100111', (6, 5): '1111111110101000', (6, 6): '1111111110101001',
    (6, 7): '1111111110101010', (6, 8): '1111111110101011', (6, 9): '1111111110101100',
    (6,10): '1111111110101101',

    (7, 1): '11111010', (7, 2): '111111110111', (7, 3): '1111111110101110',
    (7, 4): '1111111110101111', (7, 5): '1111111110110000', (7, 6): '1111111110110001',
    (7, 7): '1111111110110010', (7, 8): '1111111110110011', (7, 9): '1111111110110100',
    (7,10): '1111111110110101',

    (8, 1): '111111000', (8, 2): '111111111000000', (8, 3): '1111111110110110',
    (8, 4): '1111111110110111', (8, 5): '1111111110111000', (8, 6): '1111111110111001',
    (8, 7): '1111111110111010', (8, 8): '1111111110111011', (8, 9): '1111111110111100',
    (8,10): '1111111110111101',

    (9, 1): '111111001', (9, 2): '1111111110111110', (9, 3): '1111111110111111',
    (9, 4): '1111111111000000', (9, 5): '1111111111000001', (9, 6): '1111111111000010',
    (9, 7): '1111111111000011', (9, 8): '1111111111000100', (9, 9): '1111111111000101',
    (9,10): '1111111111000110',

    (10,1): '111111010', (10,2): '1111111111000111', (10,3): '1111111111001000',
    (10,4): '1111111111001001', (10,5): '1111111111001010', (10,6): '1111111111001011',
    (10,7): '1111111111001100', (10,8): '1111111111001101', (10,9): '1111111111001110',
    (10,10): '1111111111001111',

    (11,1): '1111111001', (11,2): '1111111111010000', (11,3): '1111111111010001',
    (11,4): '1111111111010010', (11,5): '1111111111010011', (11,6): '1111111111010100',
    (11,7): '1111111111010101', (11,8): '1111111111010110', (11,9): '1111111111010111',
    (11,10): '1111111111011000',

    (12,1): '1111111010', (12,2): '1111111111011001', (12,3): '1111111111011010',
    (12,4): '1111111111011011', (12,5): '1111111111011100', (12,6): '1111111111011101',
    (12,7): '1111111111011110', (12,8): '1111111111011111', (12,9): '1111111111100000',
    (12,10): '1111111111100001',

    (13,1): '11111111000', (13,2): '1111111111100010', (13,3): '1111111111100011',
    (13,4): '1111111111100100', (13,5): '1111111111100101', (13,6): '1111111111100110',
    (13,7): '1111111111100111', (13,8): '1111111111101000', (13,9): '1111111111101001',
    (13,10): '1111111111101010',

    (14,1): '1111111111101011', (14,2): '1111111111101100', (14,3): '1111111111101101',
    (14,4): '1111111111101110', (14,5): '1111111111101111', (14,6): '1111111111110000',
    (14,7): '1111111111110001', (14,8): '1111111111110010', (14,9): '1111111111110011',
    (14,10): '1111111111110100',

    (15,1): '1111111111110101', (15,2): '1111111111110110', (15,3): '1111111111110111',
    (15,4): '1111111111111000', (15,5): '1111111111111001', (15,6): '1111111111111010',
    (15,7): '1111111111111011', (15,8): '1111111111111100', (15,9): '1111111111111101',
    (15,10): '1111111111111110',

    (15, 0): '11111111001',  # ZRL (16 нулей подряд)
}
AC_HUFFMAN_DECODE = {v: k for k, v in AC_HUFFMAN_TABLE.items()}

def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128
    return np.stack((Y, Cb, Cr), axis=-1)
def ycbcr_to_rgb(img):
    Y, Cb, Cr = img[..., 0], img[..., 1], img[..., 2]
    Cb -= 128
    Cr -= 128
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    return np.clip(np.stack((R, G, B), axis=-1), 0, 255).astype(np.uint8)

def downsample_420(channel):
    H, W = channel.shape
    H = H - (H % 2)
    W = W - (W % 2)
    channel = channel[:H, :W]

    return (channel[0::2, 0::2] +
            channel[1::2, 0::2] +
            channel[0::2, 1::2] +
            channel[1::2, 1::2]) / 4
def upsample_420(channel, shape): return np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)[:shape[0], :shape[1]]

def create_dct_matrix(N):
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            D[k, n] = alpha * np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
    return D
def dct2(block, D): return D @ block @ D.T
def idct2(block, D): return D.T @ block @ D

def get_quant_matrix(N, quality, chrominance=False):
    Q_chrom_base = np.array([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]
    ])
    Q_base_luminance = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])
    Q_base = Q_chrom_base if chrominance else Q_base_luminance
    if quality <= 0: quality = 1
    if N != 8: return np.ones((N, N)) * (quality if quality > 0 else 1)
    scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
    return np.clip((Q_base * scale + 50) // 100, 1, 255)
def quantize(block, Q): return np.round(block / Q)
def dequantize(block, Q): return block * Q

def zigzag(block):
    h, w = block.shape
    result = []
    for s in range(h + w - 1):
        for y in range(s + 1):
            x = s - y
            if y < h and x < w:
                result.append(block[y, x] if s % 2 == 0 else block[x, y])
    return np.array(result)
def inverse_zigzag(arr, N):
    block = np.zeros((N, N))
    i = 0
    for s in range(2 * N - 1):
        for y in range(s + 1):
            x = s - y
            if y < N and x < N:
                if s % 2 == 0:
                    block[y, x] = arr[i]
                else:
                    block[x, y] = arr[i]
                i += 1
    return block

def differential_encode_dc(dc_values):
    diffs = [dc_values[0]]
    for i in range(1, len(dc_values)):
        diffs.append(dc_values[i] - dc_values[i - 1])
    return np.array(diffs)
def differential_decode_dc(diffs):
    values = [diffs[0]]
    for i in range(1, len(diffs)):
        values.append(values[-1] + diffs[i])
    return np.array(values)

def pad_image(img, block_size):
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
def process_blocks(channel, block_size, func):
    padded = pad_image(channel, block_size)
    h, w = padded.shape
    result = np.zeros_like(padded)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            result[i:i+block_size, j:j+block_size] = func(block)
    return result[:channel.shape[0], :channel.shape[1]]

def get_category(value):
    if value == 0:
        return 0
    abs_val = abs(int(value))
    return int(np.floor(np.log2(abs_val))) + 1
def encode_value_with_inversion(value, category):
    if category == 0:
        return ''
    abs_val = abs(int(value))
    bin_str = format(abs_val, f'0{category}b')
    if value >= 0:
        return bin_str
    inverted = ''.join('1' if b == '0' else '0' for b in bin_str)
    return inverted
def variable_length_encode(zz_blocks):
    encoded_blocks = []
    for zz in zz_blocks:
        dc = zz[0]
        dc_cat = get_category(dc)
        dc_huff = DC_HUFFMAN_TABLE[dc_cat]
        dc_bin = encode_value_with_inversion(dc, dc_cat)
        dc_code = (dc_huff, dc_bin)

        ac_codes = []
        zero_count = 0
        for val in zz[1:]:
            if val == 0:
                zero_count += 1
            else:
                while zero_count > 15:
                    ac_huff = AC_HUFFMAN_TABLE[(15, 0)]
                    ac_codes.append((ac_huff, ''))  # ZRL
                    zero_count -= 16
                cat = get_category(val)
                val_bin = encode_value_with_inversion(val, cat)
                ac_huff = AC_HUFFMAN_TABLE.get((zero_count, cat), '')
                ac_codes.append((ac_huff, val_bin))
                zero_count = 0
        if zero_count > 0:
            ac_codes.append((AC_HUFFMAN_TABLE[(0, 0)], ''))  # EOB

        encoded_blocks.append((dc_code, ac_codes))
    return encoded_blocks
def decode_inverted_bits(bit_str, category):
    if not bit_str:
        return 0
    if bit_str[0] == '1':
        return int(bit_str, 2)
    inverted = ''.join('1' if b == '0' else '0' for b in bit_str)
    return -int(inverted, 2)
def variable_length_decode(encoded_blocks, block_size):
    zz_blocks = []
    for dc_code, ac_codes in encoded_blocks:
        dc_huff, dc_bin = dc_code
        dc_cat = DC_HUFFMAN_DECODE[dc_huff]
        dc_val = decode_inverted_bits(dc_bin, dc_cat)
        zz = [dc_val]

        ac = []
        for ac_huff, val_bin in ac_codes:
            run_size = AC_HUFFMAN_DECODE[ac_huff]
            run, size = run_size
            if (run, size) == (0, 0):  # EOB
                ac.extend([0] * (block_size * block_size - 1 - len(ac)))
                break
            elif (run, size) == (15, 0):  # ZRL
                ac.extend([0] * 16)
            else:
                val = decode_inverted_bits(val_bin, size)
                ac.extend([0] * run)
                ac.append(val)
        while len(ac) < block_size * block_size - 1:
            ac.append(0)
        zz.extend(ac)
        zz_blocks.append(np.array(zz))
    return zz_blocks

def compress(img, quality=50, block_size=8):
    D = create_dct_matrix(block_size)
    Q_Y = get_quant_matrix(block_size, quality, chrominance=False)
    Q_C = get_quant_matrix(block_size, quality, chrominance=True)

    img_np = np.array(img)
    ycbcr = rgb_to_ycbcr(img_np)
    Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    Cb_d = downsample_420(Cb)
    Cr_d = downsample_420(Cr)

    def encode_channel(channel, Q):
        padded = pad_image(channel, block_size)
        h, w = padded.shape
        dc_values = []
        ac_blocks = []

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = padded[i:i + block_size, j:j + block_size]
                dct = dct2(block, D)
                q = quantize(dct, Q)
                zz = zigzag(q)
                dc_values.append(zz[0])
                ac_blocks.append(zz[1:])

        dc_diffs = differential_encode_dc(dc_values)
        encoded_ac = variable_length_encode(ac_blocks)
        return dc_diffs, encoded_ac, padded.shape

    y_dc, y_ac, shape_Y = encode_channel(Y, Q_Y)
    cb_dc, cb_ac, shape_Cb = encode_channel(Cb_d, Q_C)
    cr_dc, cr_ac, shape_Cr = encode_channel(Cr_d, Q_C)

    dc_encoded = {
        'y': y_dc,
        'cb': cb_dc,
        'cr': cr_dc
    }

    return {
        'dc': dc_encoded,
        'ac_y': y_ac,
        'ac_cb': cb_ac,
        'ac_cr': cr_ac,
        'size': img_np.shape[:2],
        'quality': quality,
        'shapes': {
            'y': shape_Y,
            'cb': shape_Cb,
            'cr': shape_Cr
        }
    }
def decompress(data, block_size=8):
    quality = data['quality']
    D = create_dct_matrix(block_size)
    Q_Y = get_quant_matrix(block_size, quality, chrominance=False)
    Q_C = get_quant_matrix(block_size, quality, chrominance=True)

    def decode_channel(dc_diffs, ac_blocks, shape, Q):
        dc_values = differential_decode_dc(dc_diffs)

        ac_zz = variable_length_decode(ac_blocks, block_size)

        zz_blocks = []
        for dc, ac in zip(dc_values, ac_zz):
            zz = np.concatenate(([dc], ac))
            zz_blocks.append(zz)

        blocks = []
        for zz in zz_blocks:
            q = inverse_zigzag(zz, block_size)
            idct = idct2(dequantize(q, Q), D)
            blocks.append(idct)

        h, w = shape
        rec = np.zeros((h, w))
        idx = 0
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                rec[i:i + block_size, j:j + block_size] = blocks[idx]
                idx += 1
        return rec[:shape[0], :shape[1]]

    dc_data = data['dc']
    Y_rec = decode_channel(dc_data['y'], data['ac_y'], data['shapes']['y'], Q_Y).clip(0, 255)
    Cb_rec = upsample_420(decode_channel(dc_data['cb'], data['ac_cb'], data['shapes']['cb'], Q_C), Y_rec.shape)
    Cr_rec = upsample_420(decode_channel(dc_data['cr'], data['ac_cr'], data['shapes']['cr'], Q_C), Y_rec.shape)

    final_ycbcr = np.stack((Y_rec, Cb_rec, Cr_rec), axis=-1)
    return ycbcr_to_rgb(final_ycbcr).astype(np.uint8)


img = Image.open("Lenna1.png").convert("RGB")
compressed = compress(img, quality=0)

restored_img = decompress(compressed)
Image.fromarray(restored_img).show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Restored")
plt.imshow(restored_img)
plt.axis("off")
plt.tight_layout()
plt.show()

Image.fromarray(restored_img).save("restored_Lenna1.png")



# import pickle
# import zlib
# def save_compressed_data(data, filename):
#     serialized = pickle.dumps(data)
#     compressed = zlib.compress(serialized)
#     with open(filename, 'wb') as f:
#         f.write(compressed)
#
# def load_compressed_data(filename):
#     with open(filename, 'rb') as f:
#         compressed = f.read()
#     serialized = zlib.decompress(compressed)
#     return pickle.loads(serialized)
#
# img = Image.open("Lenna1.png").convert("RGB")
# compressed = compress(img, quality=100)
# save_compressed_data(compressed, "compressed_lenna.jpc")
# loaded_data = load_compressed_data("compressed_lenna.jpc")
# restored_img = decompress(loaded_data)
# Image.fromarray(restored_img).show()

#ывести все сжатые картинки в папку
# import os
# output_dir = r"C:\Users\root\Desktop\четвертый семестр\алгосы\ready_im"
# os.makedirs(output_dir, exist_ok=True)
#
# qualities = [0, 20, 40, 60, 80, 100]
#
# img = Image.open("im2.png").convert("RGB")
#
# for quality in qualities:
#     print(f"Обработка с качеством {quality}...")
#
#     compressed = compress(img, quality=quality)
#     restored_img = decompress(compressed)
#
#     output_path = os.path.join(output_dir, f"restored_im_q{quality}.png")
#     Image.fromarray(restored_img).save(output_path)
#
#     print(f"Изображение сохранено: {output_path}")
#
# print("Все изображения успешно обработаны и сохранены.")


