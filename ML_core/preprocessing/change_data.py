import utils
import DateCategories

# for photo_id in DateCategories.ultraviolet_photos:
#     utils.change_segments(photo_id,
#                           'ultraviolet',
#                           DateCategories.colors_for_ultraviolet)
#
#
for photo_id in DateCategories.daylight_photos:
    utils.change_segments(photo_id,
                          'daylight',
                           DateCategories.colors_for_daylight)

# tools.show_colormap()

