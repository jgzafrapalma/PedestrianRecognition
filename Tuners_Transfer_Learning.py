import kerastuner
import DataGenerators_Pretext_Tasks


class TunerBayesianFINALCrossingDetection(kerastuner.tuners.BayesianOptimization):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_frames,
                    verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64, 128], default=32),
            'n_clases': 2,
            'n_channels': 3,
            'n_frames': n_frames,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorFINALCrossingDetection(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorFINALCrossingDetection(validation_ids_instances, **params)

        super(TunerBayesianFINALCrossingDetection, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerHyperBandFINALCrossingDetection(kerastuner.tuners.Hyperband):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_frames,
                  verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64, 128], default=32),
            'n_clases': 2,
            'n_channels': 3,
            'n_frames': n_frames,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorFINALCrossingDetection(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorFINALCrossingDetection(validation_ids_instances, **params)

        super(TunerHyperBandFINALCrossingDetection, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerRandomFINALCrossingDetection(kerastuner.tuners.RandomSearch):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_frames,
                  verbose, epochs, callbacks):

        params = {
            'dim': dim,
            'path_instances': path_instances,
            'batch_size': trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64, 128], default=32),
            'n_clases': 2,
            'n_channels': 3,
            'n_frames': n_frames,
            'normalized': trial.hyperparameters.Boolean('normalized', default=True),
            'shuffle': trial.hyperparameters.Boolean('shuffle', default=True),
        }

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorFINALCrossingDetection(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorFINALCrossingDetection(validation_ids_instances, **params)

        super(TunerRandomFINALCrossingDetection, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)