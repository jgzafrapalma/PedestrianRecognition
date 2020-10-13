import kerastuner
import DataGenerators_Downstream_Tasks



########################################################################################################################
#############################################  CROSSING DETECTION ######################################################
#############################################       SHUFFLE       ######################################################
########################################################################################################################



class TunerBayesianFINALCrossingDetectionShuffle(kerastuner.tuners.BayesianOptimization):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, n_classes, n_channels,
                    verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'n_frames': n_frames,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionShuffe(train_ids_instances, **params)

        validation_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionShuffe(validation_ids_instances, **params)

        super(TunerBayesianFINALCrossingDetectionShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerHyperBandFINALCrossingDetectionShuffle(kerastuner.tuners.Hyperband):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, n_classes, n_channels,
                    verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'n_frames': n_frames,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionShuffe(train_ids_instances, **params)

        validation_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionShuffe(validation_ids_instances, **params)

        super(TunerHyperBandFINALCrossingDetectionShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerRandomFINALCrossingDetectionShuffle(kerastuner.tuners.RandomSearch):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_frames, n_classes, n_channels,
                    verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64], default=32),
            'n_clases': n_classes,
            'n_channels': n_channels,
            'n_frames': n_frames,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionShuffe(train_ids_instances, **params)

        validation_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionShuffe(validation_ids_instances, **params)

        super(TunerRandomFINALCrossingDetectionShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)



########################################################################################################################
#######################################   CROSSING DETECTION    ########################################################
#######################################    ORDER PREDICTION     ########################################################
########################################################################################################################



class TunerBayesianFINALCrossingDetectionOrderPrediction(kerastuner.tuners.BayesianOptimization):
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
            'n_epochs': epochs
        }

        train_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionOrderPrediction(validation_ids_instances, **params)

        super(TunerBayesianFINALCrossingDetectionOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerHyperBandFINALCrossingDetectionOrderPrediction(kerastuner.tuners.Hyperband):
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
            'n_epochs': epochs
        }

        train_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionOrderPrediction(validation_ids_instances, **params)

        super(TunerHyperBandFINALCrossingDetectionOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerRandomFINALCrossingDetectionOrderPrediction(kerastuner.tuners.RandomSearch):
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
            'n_epochs': epochs
        }

        train_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_Downstream_Tasks.DataGeneratorFINALCrossingDetectionOrderPrediction(validation_ids_instances, **params)

        super(TunerRandomFINALCrossingDetectionOrderPrediction, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)