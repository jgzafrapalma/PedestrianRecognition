import os, sys

#Falta importar del DataGenerator
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.append(os.path.join(rootdir, 'Pretext_Tasks', 'OrderPrediction'))

import DataGenerators_OrderPrediction

import kerastuner



class TunerBayesianOrderPrediction(kerastuner.tuners.BayesianOptimization):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_classes, n_channels,
                    verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64, 128], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
            'n_epochs': epochs
        }

        train_generator = DataGenerators_OrderPrediction.DataGeneratorOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_OrderPrediction.DataGeneratorOrderPrediction(validation_ids_instances, **params)

        super(TunerBayesianOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)


class TunerRandomOrderPrediction(kerastuner.tuners.RandomSearch):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_classes,
                    n_channels, verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64, 128], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
            'n_epochs': epochs
        }

        train_generator = DataGenerators_OrderPrediction.DataGeneratorOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_OrderPrediction.DataGeneratorOrderPrediction(validation_ids_instances, **params)

        super(TunerRandomOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerHyperBandOrderPrediction(kerastuner.tuners.Hyperband):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_classes,
                  n_channels, verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64, 128], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
            'n_epochs': epochs
        }

        train_generator = DataGenerators_OrderPrediction.DataGeneratorOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_OrderPrediction.DataGeneratorOrderPrediction(validation_ids_instances, **params)

        super(TunerHyperBandOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)