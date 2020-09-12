import keras 
import keras.backend as K
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Conv1D, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Dense
from keras.layers import UpSampling2D, LeakyReLU, Lambda, Add, Activation, BatchNormalization, add


def residual_block(inx, filters, kernel, activation, pooling):
    x = Conv2D(filters, (1, kernel), padding='same')(inx)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters, (1, kernel), padding='same')(x)
    x = BatchNormalization()(x)
    x = add([x, inx])
    x_k = Activation(activation=activation)(x)
    x = Conv2D(filters, (1, pooling), strides=(1, pooling))(x_k)
    return x, x_k

def residual_block_up(inx, x_k, filters, kernel, activation, pooling):
    x = Conv2D(filters, (1, kernel), padding='same')(inx)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters, (1, kernel), padding='same')(x)
    x = BatchNormalization()(x)
    x = add([x, x_k, inx])
    x = Activation(activation=activation)(x)
    x = UpSampling2D((1, pooling))(x)
    return x


def generator(gen_shape):
    gen_inputs = Input(shape=gen_shape)
    x = Conv2D(32, (1, 5), padding='same')(gen_inputs)
    x, x_k_1 = residual_block(x, 32, 5, 'relu', 2)
    x, x_k_2 = residual_block(x, 32, 5, 'relu', 2)
    x, x_k_3 = residual_block(x, 32, 5, 'relu', 2)
    x, x_k_4 = residual_block(x, 32, 5, 'relu', 2)
    x = Conv2D(64, (1, 5), padding='same')(x)
    x, x_k_5 = residual_block(x, 64, 5, 'relu', 2)
    x, x_k_6 = residual_block(x, 64, 5, 'relu', 2)
    x, x_k_7 = residual_block(x, 64, 5, 'relu', 2)
    y, x_k_8 = residual_block(x, 64, 5, 'relu', 2)
    x = Conv2D(64, (1, 5), padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = x

    decoder = encoded
    x = Conv2D(64, (1, 5), padding='same')(decoder)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    m1 = x

    x = Conv2D(128, (1, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    m2 = x 

    x = Conv2D(256, (1, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(0.25)(x)

    x = Conv2D(256, (1, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2D(128, (1, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([m2,x])

    x = Conv2D(128, (1, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (1, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([m1,x])

    x = Conv2D(64, (1, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block_up(x, y, 64, 5, 'relu', 2)
    x = residual_block_up(x, x_k_8, 64, 5, 'relu', 2)
    x = residual_block_up(x, x_k_7, 64, 5, 'relu', 2)
    x = residual_block_up(x, x_k_6, 64, 5, 'relu', 2)
    x = residual_block_up(x, x_k_5, 64, 5, 'relu', 2)
    x = Conv2D(32, (1, 5), padding='same')(x)
    x = residual_block_up(x, x_k_4, 32, 5, 'relu', 2)
    x = residual_block_up(x, x_k_3, 32, 5, 'relu', 2)
    x = residual_block_up(x, x_k_2, 32, 5, 'relu', 2)
    x = residual_block_up(x, x_k_1, 32, 5, 'relu', 1)
    _, x = residual_block(x, 32, 5, 'relu', 1)
    gen_outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='gen_output')(x)
    gen_outputs_lambda = Lambda(lambda x: 1-x, name='gen_output_2')(gen_outputs)
    return gen_inputs, encoded, gen_outputs, gen_outputs_lambda

def discriminator(dis_shape):
    dis_inputs = Input(shape=dis_shape)
    x = Conv2D(64, (1, 5), strides=(1,2), padding='same')(dis_inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(256, (1, 5), kernel_initializer='he_normal', strides=(1,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.25)(x)
    dis_outputs = Dense(1, kernel_initializer='he_normal', name='dis_output')(x)
    return dis_inputs, dis_outputs

def classifier(main_input_shape,class_inputs_shapes):
    class_in = Input(shape=main_input_shape)
    class_inputs = []
    for shape in class_inputs_shapes:
        class_inputs.append(Input(shape=shape))
    z = Conv2D(32, (1, 5), padding='same')(class_in)
    z = Add()([z, class_inputs[0]])
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 32, 5, 'relu', 2)
    z = Add()([z, class_inputs[1]])
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 32, 5, 'relu', 2)
    z = Add()([z, class_inputs[2]])
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 32, 5, 'relu', 2)
    z = Add()([z, class_inputs[3]])
    z = Dropout(0.25)(z)

    z = Conv2D(64, (1, 5), padding='same')(z)

    z, _ = residual_block(z, 64, 5, 'relu', 2)
    z = Add()([z, class_inputs[4]])
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 64, 5, 'relu', 2)
    z = Add()([z, class_inputs[5]])
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 64, 5, 'relu', 2)
    z = Add()([z, class_inputs[6]])
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 64, 5, 'relu', 2)
    z = Add()([z, class_inputs[7]])
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 64, 5, 'relu', 2)
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 64, 5, 'relu', 2)
    z = Dropout(0.25)(z)

    z, _ = residual_block(z, 64, 5, 'relu', 2)
    z = Dropout(0.25)(z)

    z = Conv2D(128, (1, 5), kernel_initializer='he_normal', padding='same')(z)
    z = BatchNormalization()(z)
    z = LeakyReLU(0.2)(z)
    z = Dropout(0.25)(z)

    z = Conv2D(256, (1, 5), kernel_initializer='he_normal', padding='same')(z)
    z = BatchNormalization()(z)
    z = LeakyReLU(0.2)(z)
    z = Dropout(0.25)(z)


    z = GlobalMaxPooling2D()(z)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.4)(z)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.4)(z)

    class_out = Dense(1, activation='sigmoid')(z)
    return class_in, class_inputs, class_out