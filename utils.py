from io import StringIO
import PIL.Image
import base64

classes = {'abraham_grampa_simpson': 0, 'agnes_skinner': 1, 'apu_nahasapeemapetilon': 2, 
    'barney_gumble': 3, 'bart_simpson': 4, 'bumblebee_man': 5, 'carl_carlson': 6, 
    'charles_montgomery_burns': 7, 'chief_wiggum': 8, 'cletus_spuckler': 9, 
    'comic_book_guy': 10, 'disco_stu': 11, 'edna_krabappel': 12, 'fat_tony': 13, 
    'gil': 14, 'groundskeeper_willie': 15, 'hans_moleman': 16, 'helen_lovejoy': 17, 
    'homer_simpson': 18, 'jasper_beardly': 19, 'jimbo_jones': 20, 'kent_brockman': 21, 
    'krusty_the_clown': 22, 'lenny_leonard': 23, 'lionel_hutz': 24, 'lisa_simpson': 25, 
    'maggie_simpson': 26, 'marge_simpson': 27, 'martin_prince': 28, 'mayor_quimby': 29, 
    'milhouse_van_houten': 30, 'miss_hoover': 31, 'moe_szyslak': 32, 'ned_flanders': 33, 
    'nelson_muntz': 34, 'otto_mann': 35, 'patty_bouvier': 36, 'principal_skinner': 37, 
    'professor_john_frink': 38, 'rainier_wolfcastle': 39, 'ralph_wiggum': 40, 'selma_bouvier': 41, 
    'sideshow_bob': 42, 'sideshow_mel': 43, 'snake_jailbird': 44, 'troy_mcclure': 45, 'waylon_smithers': 46}



def decode_img(img_base64):
  decode_str = base64.b64decode(img_base64)
  file_like = StringIO.StringIO(decode_str)
  img = PIL.Image.open(file_like)
  # rgb_img[c, r] is the pixel values.
  rgb_img = img.convert("RGB")
  return rgb_img


