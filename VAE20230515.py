import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import utils
import requests
import datetime

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_layers = [
            tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),#128
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),#64
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),#32
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),#16*16
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=-1)
        return mean, logvar


# 解碼器
class Decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder_layers = [
            tf.keras.layers.Dense(units=16 * 16 * 128, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(16, 16, 128)),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding="SAME",#32
                                            activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding="SAME",#64
                                            activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME",#128
                                            activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME",#256,256,32
                                            activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=(1, 1), padding="SAME",activation='sigmoid')#256,256,3

        ]

    def call(self, inputs):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(x)
        return x
#損失函數
def compute_loss(encoder, decoder, x, real_data):#x=sketch_data->input
    mean, logvar = encoder(x)
    eps = tf.random.normal(shape=mean.shape)
    z = eps * tf.exp(logvar * 0.5) + mean
    x_recon = decoder(z)
    recon_loss = tf.reduce_sum(tf.square(real_data - x_recon), axis=[1, 2, 3])
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
    loss = tf.reduce_mean(recon_loss + kl_loss)
    return loss

# 梯度下降優化器
def train_step(encoder, decoder, x, real_data, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(encoder, decoder, x, real_data)
    gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return loss

def generate_data(decoder, encoder, n_samples, input_):
    # z = tf.random.normal(shape=(n_samples, latent_dim))
    start = np.random.randint(0,input_.shape[0])
    x = input_[start:start+n_samples]
    mean, logvar = encoder(x)
    eps = tf.random.normal(shape=mean.shape)
    z = eps * tf.exp(logvar * 0.5) + mean
    x_generated = decoder(z)
    return x_generated

# 顯示生成的數據
def picture(x_generated,epoch):
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    axs = axs.flatten()
    for i in range(16):
        axs[i].imshow(x_generated[i])
        axs[i].axis("off")
    plt.savefig("./images/%d.png" % epoch)
    plt.close()

class Code_returner:
    def __init__(self,token=''):
        assert token!='','no token'
        self.headers = {
                        "Authorization": "Bearer "+token,
                        #"Content-Type":"application/x-www-form-urlencoded"
                        }
    def send_message(self,message:str):
        params = {"message": f'[INFO] {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:\n {message}'}
        requests.post('https://notify-api.line.me/api/notify', headers=self.headers, params=params)
    def send_image(self,img_path:str,message=None):
        img=open(img_path,'rb')
        files = {"imageFile": img}
        data = {'message': f'[INFO] {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:\n {message}'}
        requests.post('https://notify-api.line.me/api/notify', headers=self.headers, files=files, data=data)

if __name__=='__main__':
    # 可改參數---------------
    num_epochs = 10000
    batch_size = 64
    latent_dim = 16

    # 可改參數---------------

    # 載入資料集
    d = utils.DataLoader()
    sketchdata, realdata = d.load_data(metrics='[0,1]')  # metrics是可以改成'[-1,1]'，代表圖片標準化後的值區間
    test_sketchdata, test_realdata = d.load_testing_data(metrics='[0,1]')
    num_batches = sketchdata.shape[0] // batch_size
    # 建立VAE
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # 訓練VAE
    loss_lst = np.zeros(shape=(num_batches))
    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            x_batch = sketchdata[start_idx:end_idx]
            real_batch = realdata[start_idx:end_idx]
            loss = train_step(encoder, decoder, x_batch, real_batch, optimizer)
            loss_lst[batch_idx] = loss
        print("Epoch: {}, Loss: {}".format(epoch+1, loss))
        if epoch%1000 == 0  :
            # predict
            try:
                x_generated = generate_data(decoder, encoder, 16, sketchdata)
                picture(x_generated,epoch)
            except Exception as e:
                print(epoch)
                print(e)
    try:
        x_generated = generate_data(decoder, encoder, 16, test_sketchdata)
        picture(x_generated,123456789)
    except Exception as e:
        print(e)

    elf = Code_returner(
        token='lT8ZnYtG6gVYlMWttzydJwKzZclGxJbdwWfLzLdw8ns')  # lT8ZnYtG6gVYlMWttzydJwKzZclGxJbdwWfLzLdw8ns
    elf.send_message(message='Finish')
    elf.send_image(img_path='./images/9999.png')
    print('Connect to Line successfully')