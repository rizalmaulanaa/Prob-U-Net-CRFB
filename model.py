# modified code from https://github.com/SimonKohl/probabilistic_unet/blob/master/model/probabilistic_unet.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Mean
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, MaxPool2D, Concatenate, Conv2D

from utils import *

class Prob_Unet_crfb(Model):
    def __init__(self, num_classes, activation, latent_dim=6,
                 resolution_lvl=5, block_type='conv_relu', concat_lvl=[0],
                 img_shape=(None, None, 1), seg_shape=(None, None, 1),
                 num_filters=(32, 64, 128, 256, 512), downsample_signal=(2,2,2,2,2)):
        super(Prob_Unet_crfb, self).__init__()
        self.latent_dim = latent_dim
        self.activation = activation
        self.block_type = block_type
        self.concat_lvl = concat_lvl
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.resolution_lvl = resolution_lvl
        self.downsample_signal = downsample_signal

        self.prior = self.latent_space_net(img_shape, None)
        self.posterior = self.latent_space_net(img_shape, seg_shape)
        self.det_unet = self.unet(img_shape)

    def latent_space_net(self, img_shape, seg_shape):
        if seg_shape is not None:
            # Posterior inputs
            inputs = [Input(shape=img_shape), Input(shape=seg_shape)]
            input_ = Concatenate(name='input_con') (inputs)
            name = 'prob_unet_posterior'
        else:
            # Prior input
            inputs = Input(shape=img_shape)
            input_ = inputs
            name = 'prob_unet_prior'

        # Encoder blocks
        for i in range(self.resolution_lvl):
            if i == 0:
                x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='encoder_latent') (input_)
            else:
                x = MaxPool2D(pool_size=self.downsample_signal[i],
                              name='encoder_latent_stage0-{}_pool'.format(i)) (x)
                x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='encoder_latent') (x)

        # Z sample
        z, mu, sigma = z_mu_sigma(self.latent_dim, 0, self.resolution_lvl+1) (x)
        return Model(inputs, [z, mu, sigma], name=name)

    def unet(self, img_shape):
        z_sample = Input(shape=(None, None, self.latent_dim))
        inputs = Input(shape=img_shape)
        skip_connections = [None] * self.resolution_lvl

        # Define block
        if self.block_type == 'inception_block':
            block = inception_block
        elif self.block_type == 'res_block':
            block = res_block
        else:
            block = conv_block

        # Encoder blocks
        for i in range(self.resolution_lvl):
            if i == 0:
                x = block(self.num_filters[i], 0, i, amount=2, type_block='encoder') (inputs)
                x_u = block(self.num_filters[i], 1, i, amount=2, type_block='encoder') (inputs)
                skip_connections[i] = x

                # CRFB
                x = MaxPool2D(pool_size=self.downsample_signal[i],
                              name='encoder_stage0-{}_pool'.format(i)) (x)
                x_u = UpSampling2D(size=self.downsample_signal[i],
                                   name='encoder_stage1-{}_up'.format(i)) (x_u)
                x, x_u = CRFB(self.num_filters[i], 0, i, self.downsample_signal[i],
                              type_block='CRFB') (x, x_u)

            else:
                if i > 1:
                    x = MaxPool2D(pool_size=self.downsample_signal[i],
                                name='encoder_stage0-{}_pool'.format(i)) (x)
                x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='encoder') (x)

                skip_connections[i] = x
        skip_connections = skip_connections[:-1]

        # Decoder blocks
        for i in reversed(range(self.resolution_lvl-1)):
            x = UpSampling2D(size=self.downsample_signal[i],
                            name='decoder_stage0-{}_up'.format(i)) (x)

            x = Concatenate(name='decoder_stage0-{}_con'.format(i)) ([x, skip_connections[i]])
            x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='decoder') (x)

            if i == 1:
                temp_x = UpSampling2D(size=self.downsample_signal[i]**2,
                                      name='decoder_stage{}-1_up'.format(i)) (x)
                temp_x = conv_block(self.num_filters[0], i, 1, amount=2, type_block='decoder') (temp_x)
                x_u = Add(name='decoder_stage{}-1_add'.format(i)) ([temp_x,x_u])

        x_u = conv_block(self.num_filters[0], self.resolution_lvl-1, 1, amount=2,
                         type_block='decoder') (x_u)
        x_u = MaxPool2D(pool_size=self.downsample_signal[0],
                        name='decoder_stage{}-1_pool'.format(self.resolution_lvl-1)) (x_u)
        x = Add(name='final_add') ([x_u,x])

        # Concatenate U-Net and Z sample
        if 0 in self.concat_lvl:
            lvl_div = np.power(2, self.resolution_lvl-1)
            broadcast_z = tf.tile(z_sample, (1, lvl_div, lvl_div, 1))
            x = Concatenate(name='final_con') ([x, broadcast_z])

        x = conv_block(self.num_filters[0], 0, self.resolution_lvl-1, amount=2, type_block='final') (x)
        x = Conv2D(self.num_classes, kernel_size=1, padding='same',
                   activation=self.activation, name='final_conv') (x)
        return Model([inputs, z_sample], x, name='prob_unet_det')

    def kl_score(self, mu0, sigma0, mu1, sigma1):
        # Calculate kl loss
        sigma0_f = K.square(K.flatten(sigma0))
        sigma1_f = K.square(K.flatten(sigma1))
        logsigma0 = K.log(sigma0_f + 1e-10)
        logsigma1 = K.log(sigma1_f + 1e-10)
        mu0_f = K.flatten(mu0)
        mu1_f = K.flatten(mu1)

        return tf.reduce_mean(
            0.5*tf.reduce_sum(tf.divide(sigma0_f + tf.square(mu1_f - mu0_f), sigma1_f + 1e-10)
            + logsigma1 - logsigma0 - 1, axis=-1))

    def compile(self, prior_opt, posterior_opt, unet_opt, loss, metric, beta=1):
        super(Prob_Unet_crfb, self).compile()
        self.posterior_opt = posterior_opt
        self.prior_opt = prior_opt
        self.unet_opt = unet_opt
        self.beta = beta

        self.metric = metric
        self.compiled_loss = loss
        self.metric_tracker = Mean(name='metric')
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.total_loss_tracker = Mean(name='total_loss')
        self.compiled_loss_tracker = Mean(name='compiled_loss')

    @property
    def metrics(self):
        return [
            self.compiled_loss_tracker,
            self.kl_loss_tracker,
            self.total_loss_tracker,
            self.metric_tracker
        ]

    def train_step(self, data):
        img, seg = data
        with tf.GradientTape(persistent=True) as tape:
            #Prior and Posterior
            _, mu_prior, sigma_prior = self.prior(img, training=True)
            z_posterior, mu_posterior, sigma_posterior = self.posterior([img, seg], training=True)

            #U-Net
            reconstruction = self.det_unet([img, z_posterior], training=True)

            #Calculate losses and metric
            kl_loss = self.kl_score(mu_posterior, sigma_posterior, mu_prior, sigma_prior)
            reconstruction_loss = self.compiled_loss(seg, reconstruction)
            total_loss = reconstruction_loss + self.beta * kl_loss
            dsc_score = self.metric(seg, reconstruction)

        # Update weights
        grad_prior = tape.gradient(kl_loss, self.prior.trainable_weights)
        self.prior_opt.apply_gradients(zip(grad_prior, self.prior.trainable_weights))

        grad_posterior = tape.gradient(kl_loss, self.posterior.trainable_weights)
        self.posterior_opt.apply_gradients(zip(grad_posterior, self.posterior.trainable_weights))

        grad_unet = tape.gradient(reconstruction_loss, self.det_unet.trainable_weights)
        self.unet_opt.apply_gradients(zip(grad_unet, self.det_unet.trainable_weights))

        self.metric_tracker.update_state(dsc_score)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.compiled_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.compiled_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "dice_coef": self.metric_tracker.result()
        }

    def test_step(self, data):
        img, seg = data
        z_prior, mu_prior, sigma_prior = self.prior(img, training=False)
        _, mu_posterior, sigma_posterior = self.posterior([img, seg], training=False)
        reconstruction = self.det_unet([img, z_prior], training=False)

        kl_loss = self.kl_score(mu_posterior, sigma_posterior, mu_prior, sigma_prior)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(self.compiled_loss(seg, reconstruction)))
        total_loss = reconstruction_loss + self.beta * kl_loss
        dsc_score = tf.reduce_mean(tf.reduce_sum(self.metric(seg, reconstruction)))

        self.metric_tracker.update_state(dsc_score)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.compiled_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.compiled_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "dice_coef": self.metric_tracker.result()
        }
