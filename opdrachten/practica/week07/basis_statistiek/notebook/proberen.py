array = np.array(img) #hier maken we een numpy array van onze variabele img (waar plaatje aarde in staat)
type(array)
scaleFactor = 1 #hier zetten we de scale factor (nu dus 1 want dat is onze basis)
scaleArray = ndimage.zoom(array, (scaleFactor,scaleFactor, 1)) #hier schalen we onze array op de x en y as(niet de Z as dat is de 1 en die blijft 1)
imgScaled = Image.fromarray(scaleArray)
imgScaled.show()

"""
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Conv2D(16, 3, padding= 'same', activation='relu'),
#    tf.keras.layers.MaxPooling2D(2, 2),
#    tf.keras.layers.Conv2D(16, 3, padding= 'same', activation='softmax'),
#    tf.keras.layers.MaxPooling2D(2, 2),
#    tf.keras.layers.Conv2D(16, 3, padding= 'same', activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(10) #, activation='relu')
#    ])
"""



"""
invoerlaag=tf.keras.layers.Flatten(input_shape = [64, 64, 3])
[invoerlaag]
hiddenlayer = tf.keras.layers.Dense(16, activation ='relu'),
outputlayer =tf.keras.layers.Dense(10,)
#model = tf.keras.models.Sequential([invoerlaag, hiddenlayer, hiddenlayer, outputlayer ])
"""