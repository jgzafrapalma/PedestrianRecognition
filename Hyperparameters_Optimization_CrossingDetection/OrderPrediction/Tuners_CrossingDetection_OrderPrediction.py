import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
rootdir = os.path.dirname(parentparentdir)
sys.path.append(os.path.join(rootdir, 'Downstream_Tasks', 'CrossingDetection', 'OrderPrediction'))

import DataGenerators_CrossingDetection_OrderPrediction

import kerastuner



class TunerBayesianCrossingDetectionOrderPrediction(kerastuner.tuners.BayesianOptimization):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_classes, n_channels,
                    verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPrediction(validation_ids_instances, **params)

        super(TunerBayesianCrossingDetectionOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks, shuffle=False)

class TunerHyperBandCrossingDetectionOrderPrediction(kerastuner.tuners.Hyperband):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_classes, n_channels,
                    verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPrediction(validation_ids_instances, **params)

        super(TunerHyperBandCrossingDetectionOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks, shuffle=False)

class TunerRandomCrossingDetectionOrderPrediction(kerastuner.tuners.RandomSearch):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_classes, n_channels,
                    verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_CrossingDetection_OrderPrediction.DataGeneratorCrossingDetectionOrderPrediction(validation_ids_instances, **params)

        super(TunerRandomCrossingDetectionOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks, shuffle=False)