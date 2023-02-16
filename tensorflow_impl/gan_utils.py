import keras 
import numpy as np
from tqdm import tqdm
import keras.backend as K
from functools import partial
from keras.models import Model
from keras.layers import Input
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
from keras.layers.merge import _Merge
from sklearn.metrics import roc_curve, auc, roc_auc_score
from strudel_utils import imap_unordered_bar, load_data, process_transit, p_epsilon_chart


class IWGAN_TrainingScheme(object):

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    @staticmethod
    def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    @staticmethod
    class RandomWeightedAverage(_Merge):
        def _merge_function(self, inputs):
            weights = K.random_uniform((32, 1, 1, 1))
            return (weights * inputs[0]) + ((1 - weights) * inputs[1])

    @staticmethod
    def compile_discriminator(generator, discriminator, optimizer):
        gp_weight = 10
        for layer in generator.layers: layer.trainable = False
        generator.trainable = False
        inp = Input(tuple(generator.layers[0].input_shape[1:]))
        in_gen = generator(inp)
        in_real = Input(tuple(discriminator.layers[0].input_shape[1:]))
        discriminator_output_from_generator = discriminator(in_gen)
        discriminator_output_from_real_samples = discriminator(in_real)
        averaged_samples = IWGAN_TrainingScheme.RandomWeightedAverage()([in_real, in_gen])
        averaged_samples_out = discriminator(averaged_samples)
        partial_gp_loss = partial(IWGAN_TrainingScheme.gradient_penalty_loss, averaged_samples=averaged_samples,
                                  gradient_penalty_weight=gp_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'
        in_gen.trainable = False

        # ----- Discriminator -----
        discriminator_model = Model(inputs=[in_real, inp], outputs=[discriminator_output_from_real_samples,
                                                                    discriminator_output_from_generator,
                                                                    averaged_samples_out])
        discriminator_model.layers[1].trainable = False
        discriminator_model.compile(optimizer=optimizer, loss=[IWGAN_TrainingScheme.wasserstein_loss,
                                                            IWGAN_TrainingScheme.wasserstein_loss, partial_gp_loss])
        # ----- Discriminator -----

        for layer in generator.layers: layer.trainable = True
        generator.trainable = True
        plot_model(discriminator_model, show_shapes=True, to_file='DCGAN_model_dis.png')
        return IWGAN_TrainingScheme.wasserstein_loss, optimizer, discriminator_model

    @staticmethod
    def compile_generator(generator, discriminator, gen_loss, dis_loss, optimizer, gen_metrics, dis_metrics, gen_dis_loss_ratio):
        for layer in discriminator.layers: layer.trainable = False
        discriminator.trainable = False
        inp = Input(tuple(generator.layers[0].input_shape[1:]))
        gen_out = generator(inp)

        # ----- Generator -----
        model = Model(inp, [gen_out,discriminator(generator(inp))])
        model.layers[2].trainable = False
        model.compile(loss={'discriminator': dis_loss, 'generator': gen_loss},
                      optimizer=optimizer, metrics={'discriminator':dis_metrics, 'generator':gen_metrics},
                      loss_weights={'discriminator': 1-gen_dis_loss_ratio, 'generator':gen_dis_loss_ratio})
        # ----- Generator -----

        for layer in discriminator.layers: layer.trainable = True
        discriminator.trainable = True
        return model

    @staticmethod
    def train_discriminator(dis, x_true, y_true, batch_size):
        # Discriminator Training
        loss = dis.train_on_batch([y_true, x_true],  # inp_real, x
                                         [np.ones((batch_size, 1), dtype=np.float32),  # discriminator_output_from_real_samples
                                          -np.ones((batch_size, 1), dtype=np.float32),  # discriminator_output_from_generator
                                          np.zeros((batch_size, 1), dtype=np.float32)])  # averaged_samples_out

        return loss, dis

    @staticmethod
    def train_generator(gen, x_true, y_true, batch_size):

        # Generator Training
        loss = gen.train_on_batch(x_true,                                                              # x_true
                                    [y_true, np.ones((len(x_true), 1), dtype=np.float32)])            # y_true , 1
                                 

        return loss, gen


class GAN(object):
    def __init__(self, gen_inputs, gen_outputs, encoded, optimizer, gen_loss, dis_inputs, dis_outputs, class_inputs, class_outputs, training_scheme='IWGAN', gen_dis_loss_ratio=0.5, dis_metrics=(), gen_metrics=()):
        
        self._training_scheme = None
        if training_scheme == 'IWGAN': self._training_scheme = IWGAN_TrainingScheme
        # if training_scheme == 'DCGAN': self._training_scheme = DCGAN_TrainingScheme
        # if training_scheme == 'RANDOMGANNAME': self._training_scheme = RANDOMGANNAME_TrainingScheme
        # ...
        # ..

        # TODO: ----- Unrelated, Move out -----
        self.__encoder = Model(gen_inputs, encoded, name='encoder')
        #self.__classifier = Model(class_inputs, class_outputs, 'classifier')
        #self.__classifier = OUTOFGAN_stack_compile(self.__encoder, self.__classifier, 'binary_crossentropy', Adam(), ['accuracy'], None, 'classifier', reverse_freeze=True)
        self.__generator = Model(gen_inputs, gen_outputs, name='generator')
        #self.__generator.summary()
        self.__discriminator = Model(dis_inputs, dis_outputs, name='discriminator')
        #self.__discriminator.summary()
        # TODO: ----- Unrelated, Move out -----

        # TODO: Extract the gen optimizer
        # TODO: Extract the losses from the keras Models
        # TODO: Extract the metrics from the keras Models
        dis_loss, dis_optimizer, self.__dis = self._training_scheme.compile_discriminator(self.__generator, self.__discriminator, optimizer)
        self.__gen_dis = self._training_scheme.compile_generator(self.__generator, self.__discriminator, gen_loss, dis_loss, dis_optimizer, gen_metrics, dis_metrics, gen_dis_loss_ratio)
    
    def encoder(self): return self.__encoder
    def generator(self): return self.__generator
    def discriminator(self): return self.__discriminator
    def gen_dis(self): return self.__gen_dis
    def dis(self): return self.__dis
    
    def fit(self, x, y, gen_overtraining_multiplier=1, verbose=True, epochs=10, batch_size=64, true_idxs=None, pregen_epochs=None, callbacks=None):
        self.verbose = verbose
        if true_idxs is not None:x_true, y_true = x[true_idxs], y[true_idxs]
        else: x_true, y_true = x, y
        for callback in callbacks:
            callback.on_train_begin()

        if pregen_epochs is not None: self.__generator.fit(x, y, epochs=pregen_epochs, batch_size=batch_size, validation_split=0.2, verbose=1, shuffle=True )
        for i in tqdm(range(epochs)):

            for callback in callbacks:
                callback.on_epoch_begin(i)
            for j in range(gen_overtraining_multiplier): 
                idxs = np.random.randint(0, x_true.shape[0], size=batch_size)
                loss, _ = self._training_scheme.train_discriminator(self.__dis, x_true[idxs], y_true[idxs], batch_size)
            if self.verbose: print('Discriminator Loss:', loss)

            # Generator Training
            idxs = np.random.randint(0, x_true.shape[0], size=batch_size)
            gen_dis_loss, _ = self._training_scheme.train_generator(self.__gen_dis, x_true[idxs], y_true[idxs], batch_size)
            if self.verbose: print('Generator Loss:', gen_dis_loss)
            for callback in callbacks:
                callback.on_epoch_end(i)
        
        for callback in callbacks:
            callback.on_train_end()

        pass


class classifier_training(keras.callbacks.Callback):
    def __init__(self, classifier_model, training_ratio, batch_size):
        self.classifier_model = classifier_model
        self.train_ratio = training_ratio
        self.x, _ ,self.y = load_data()
        self.batch_size = batch_size
    
    def on_epoch_end(self, epoch, logs=None):
        loss = None
        for i in range(self.train_ratio):
            idxs = np.random.randint(0, self.x.shape[0], size=self.batch_size)
            loss = self.classifier_model.train_on_batch(self.x[idxs], self.y[idxs])
        #print 'class loss: ', loss

class log_results(keras.callbacks.Callback):
    def __init__(self, wgan_model=None, classifier_model=None, logging_frequency=0, log_path=''):
        self.logging_frequency = logging_frequency
        self.log_path = log_path
        self.model = wgan_model
        self.classifier_model = classifier_model
        self.x, self.y, has_transit = load_data()
        self.x = self.x[has_transit]
        self.y = self.y[has_transit]
        import random
        self.idex = random.randint(0,self.x.shape[0]-1) 
        self.last_best_classifier, self.last_best_generator, self.last_best_combined= 0, 0, 0
    def on_epoch_end(self, epoch, logs=None):
        if self.logging_frequency != 0 and epoch % self.logging_frequency == 0:
            time = np.linspace(0, 28.625, 20610)
            plt.figure(1, figsize=(10,10))
            plt.subplot(211)
            plt.scatter(time,  self.y[self.idex:self.idex+1][0, 0, :20610, 0], s=0.5)
            plt.title("Simulation Output, " + str(epoch))

            final_res = self.model.gen_dis().predict_on_batch(self.x[self.idex:self.idex+1])
            plt.subplot(212)
            plt.scatter(time, final_res[0][0, 0, :20610, 0], s=0.5)
            plt.title("Neural Net Output,"+ str(final_res[1][0,0]))
            plt.tight_layout()
            plt.savefig('img_'+str(epoch)+'_test.png')
            plt.show()
            self.get_score(epoch, self.model != None, self.classifier_model != None)
    
    
    def get_score(self, i_in, gen_test = False, class_test = False):
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        
        at_2perc = 1
        if gen_test:
            periods = np.load('./total_params_sim_test_true.npy')[:,1]
            transits_ref = np.load('./total_transit_sim_test_true.npy')
            x_test = np.expand_dims(np.load('./total_x_sim_test_true.npy'),axis=1)
            x_test = np.pad(x_test, ((0,0), (0,0) ,(0, 30+ 96), (0,0)), 'constant', constant_values=(0, 0))
            print('X loaded')
            transits = self.model.gen_dis().predict(x_test, verbose=1)[0][:, 0, :20610, :]
            print("Finished predicting data")
            transits_ref = transits_ref[:,:,0]
            y_norm_max = np.max(transits_ref, axis=1, keepdims=True)
            y_norm_min = np.min(transits_ref, axis=1, keepdims=True)
            transits_ref = (transits_ref - y_norm_min) / (y_norm_max - y_norm_min)

            print("Finished loading data")
            periods = np.power(10, periods)
            period_pred = []
            model_preds = []
            np.warnings.filterwarnings('ignore')
            for i in tqdm(range(10000)): model_preds.append([transits[i, :, 0], periods[i], 1000, transits_ref[i, :]])
            model_preds = np.asarray(imap_unordered_bar(process_transit, model_preds, 5))
            auc_p, percentages, epsilon_range = p_epsilon_chart(model_preds[:, 0], model_preds[:, 1])
            at_1perc = percentages[np.argmin(np.abs(epsilon_range - 0.01))]
            at_2perc = percentages[np.argmin(np.abs(epsilon_range - 0.02))]
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(epsilon_range, percentages, label='Keras (area = {:.5f})'.format(auc_p))
            plt.xlabel('epsilon')
            plt.ylabel('period detection rate')
            plt.title('Period ROC curve')
            plt.legend(loc='best')
            plt.ylim(0, 1)
            plt.xlim(0, 0.1)
            plt.savefig('img_PAUC_'+str(i_in)+'_test.png')
            plt.show()

            if(at_2perc > self.last_best_generator): 
                self.model.gen_dis().save('best_generator_test.h5')
                self.classifier_model.save('best_classifier_test_generator_based.h5')
                self.last_best_generator = at_2perc

            if not class_test:
                open(self.log_path,'a').write( 'i: %d, width: 1000, PAUC: %.5f, Percentage at 0.01: %.5f, Percentage at 0.02: %.5f\n' % (i_in,auc_p, at_1perc, at_2perc))
                print('i: %d, width: 1000, PAUC: %.5f, Percentage at 0.01: %.5f, Percentage at 0.02: %.5f\n' % (i_in,auc_p, at_1perc, at_2perc))

        if class_test:
            y_test = np.load('./total_params_sim_test.npy')[:,1] > 0
            x_test = np.expand_dims(np.load('./total_x_sim_test.npy'),axis=1)
            x_test = np.pad(x_test, ((0,0), (0,0) ,(0, 30+ 96), (0,0)), 'constant', constant_values=(0, 0))
            y_pred = self.classifier_model.predict(x_test, verbose=1)[:,0]
            fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            if(tpr[find_nearest(fpr, 0.01)] > self.last_best_classifier): 
                self.model.gen_dis().save('best_generator_test_classifier_based.h5')
                self.classifier_model.save('best_classifier_test.h5')
                self.last_best_classifier = tpr[find_nearest(fpr, 0.01)]
            
            if(tpr[find_nearest(fpr, 0.01)] * at_2perc > self.last_best_combined):
                self.model.gen_dis().save('best_generator_combined_test.h5')
                self.classifier_model.save('best_classifier_combined_test.h5')
                self.last_best_combined = tpr[find_nearest(fpr, 0.01)] * at_2perc              
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label='Keras (area = {:.5f})'.format(roc_auc))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('Classifier ROC curve')
            plt.legend(loc='best')
            plt.ylim(0, 1)
            plt.xlim(0, 0.02)
            plt.savefig('img_ClassAUC_'+str(i_in)+'_test.png')
            plt.show()

            if not gen_test:
                open(self.log_path,'a').write( 'i: %d, Class Percentage at 0.001: %.5f, Class Percentage at 0.01: %.5f, Class AUC: %.5f\n' % (i_in, tpr[find_nearest(fpr, 0.001)], tpr[find_nearest(fpr, 0.01)], roc_auc))
                print('i: %d, Class Percentage at 0.001: %.5f, Class Percentage at 0.01: %.5f, Class AUC: %.5f\n' % (i_in, tpr[find_nearest(fpr, 0.001)], tpr[find_nearest(fpr, 0.01)], roc_auc))
                
        if class_test and gen_test:
            open(self.log_path,'a').write( 'i: %d, width: 1000, PAUC: %.5f, Percentage at 0.01: %.5f, Percentage at 0.02: %.5f, Class Percentage at 0.001: %.5f, Class Percentage at 0.01: %.5f, Class AUC: %.5f\n' % (i_in,auc_p, at_1perc, at_2perc, tpr[find_nearest(fpr, 0.001)], tpr[find_nearest(fpr, 0.01)], roc_auc))
            print('i: %d, width: 1000, PAUC: %.5f, Percentage at 0.01: %.5f, Percentage at 0.02: %.5f, Class Percentage at 0.001: %.5f, Class Percentage at 0.01: %.5f, Class AUC: %.5f\n' % (i_in,auc_p, at_1perc, at_2perc, tpr[find_nearest(fpr, 0.001)], tpr[find_nearest(fpr, 0.01)], roc_auc))


def compile_classifier(classifier, generator, loss, optimizer, metrics, loss_weights, name, reverse_freeze=False):
    for layer in generator.layers: layer.trainable = False
    generator.trainable = False
    inp = Input(tuple(generator.layers[0].input_shape[1:]))
    gen_out = [layer.output for layer in generator.layers[0::8]][1:]
    gen_multi_out = Model(generator.inputs, gen_out)
    classifier_out = classifier(gen_multi_out(inp)+[inp])
    model = Model(inp, classifier_out)
    if reverse_freeze: model.layers[1].trainable = False
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, loss_weights=loss_weights)
    model.summary()
    plot_model(model, show_shapes=True, to_file='DCGAN_model_classifier.png')
    for layer in generator.layers: layer.trainable = True
    generator.trainable = True
    return model