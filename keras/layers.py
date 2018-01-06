def dense_layer(x, h, bn=True, act='relu', d=0):
    k_init = 'he_uniform' if act == 'relu' else 'glorot_uniform'
    out = Dropout(d)(x)
    out = Dense(h, kernel_initializer=k_init)(out)
    if bn: out = BatchNormalization()(out)
    out = Activation(act)(out)
    return out

def conv_layer(x, f, k=3, s=1, p='same', act='relu', bn=True, se=True, se_ratio=16, d=0):
    k_init = 'he_uniform' if act == 'relu' else 'glorot_uniform'
    out = Dropout(d)(x)
    out = Conv2D(f, k, strides=s, padding=p, use_bias=True, kernel_initializer=k_init)(x)
    if bn: out = BatchNormalization()(out)
    if act is not None: out = Activation(act)(out)

    # squeeze and excite
    if se:
        out_se = GlobalAvgPool2D()(out)
        r = f // se_ratio if (f // se_ratio) > 0 else 1
        out_se = Reshape((1, f))(out_se)
        out_se = Dense(r, use_bias=False, kernel_initializer='he_uniform',
                       activation='relu')(out_se)
        out_se = Dense(f, use_bias=False, activation='sigmoid')(out_se)
        out = Multiply()([out, out_se])
    
    return out