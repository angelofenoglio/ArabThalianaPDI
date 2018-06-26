# # Esta parte muestra todas las labels en distintos colores
# # no esta funcional
# label_hue = numpy.uint8(179*labels/numpy.max(labels))
# blank_ch = 255*numpy.ones_like(label_hue)
# labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
# # cvt to BGR for display
# labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
# # set bg label to black
# labeled_img[label_hue==0] = 0
#
# cv2.imshow('labeled.png', labeled_img)

# # Esto es codigo que dibuja todas las caracteristicas obtenidas de la raíz
# # no esta funcional
# def draw_roots(roots, ):
#     blank = 255 * numpy.ones_like(roots)
#     label_hue = numpy.uint8(179 * (roots + intersect * 5))
#     for index in range(len(interPoints)):
#         label_hue[interPoints[index][0], interPoints[index][1]] = 120
#     for index in range(len(endPoints)):
#         label_hue[endPoints[index][0], endPoints[index][1]] = 50
#     withintersect = cv2.merge([label_hue, blank, blank])
#     withintersect = cv2.cvtColor(withintersect, cv2.COLOR_HSV2BGR)
#     withintersect[roots == 0] = 0
#     return