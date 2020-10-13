import kerastuner
import DataGenerators_Pretext_Tasks



########################################################################################################################
############################################  PRETEXT TASK SHUFFLE  ####################################################
########################################################################################################################



class TunerBayesianShuffle(kerastuner.tuners.BayesianOptimization):
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
            'step_swaps': trial.hyperparameters.Int('step_swaps', min_value=1, max_value=int(epochs / n_frames), default=5, step=1)
        }

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorShuffle(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorShuffle(validation_ids_instances, **params)

        super(TunerBayesianShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)


class TunerRandomShuffle(kerastuner.tuners.RandomSearch):
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
            'step_swaps': trial.hyperparameters.Int('step_swaps', min_value=1, max_value=int(epochs / n_frames), default=5, step=1)
        }

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorShuffle(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorShuffle(validation_ids_instances, **params)

        super(TunerRandomShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerHyperBandShuffle(kerastuner.tuners.Hyperband):
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
            'step_swaps': trial.hyperparameters.Int('step_swaps', min_value=1, max_value=int(epochs / n_frames), default=5, step=1)
        }

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorShuffle(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorShuffle(validation_ids_instances, **params)

        super(TunerHyperBandShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)



########################################################################################################################
########################################  PRETEXT TASK ORDER PREDICTION  ###############################################
########################################################################################################################



class TunerBayesianOrderPrediction(kerastuner.tuners.BayesianOptimization):
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

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorOrderPrediction(validation_ids_instances, **params)

        super(TunerBayesianShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)


class TunerRandomOrderPrediction(kerastuner.tuners.RandomSearch):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_classes,
                    n_channels, verbose, epochs, callbacks):

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

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorOrderPrediction(validation_ids_instances, **params)

        super(TunerRandomShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)

class TunerHyperBandShuffle(kerastuner.tuners.Hyperband):
    def run_trial(self, trial, train_ids_instances, validation_ids_instances, dim, path_instances, n_classes,
                  n_channels, verbose, epochs, callbacks):

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

        train_generator = DataGenerators_Pretext_Tasks.DataGeneratorOrderPrediction(train_ids_instances, **params)

        validation_generator = DataGenerators_Pretext_Tasks.DataGeneratorOrderPrediction(validation_ids_instances, **params)

        super(TunerHyperBandShuffle, self).run_trial(trial, train_generator, validation_data=validation_generator, verbose=verbose, epochs=epochs, callbacks=callbacks)