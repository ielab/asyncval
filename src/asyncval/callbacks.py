from transformers.integrations import TensorBoardCallback, WandbCallback
import logging
import os
logger = logging.getLogger(__name__)


class AsyncvalWandbCallback(WandbCallback):
    def setup(self, args, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.ai/integrations/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to log model as artifact at the end of training. Use along with
                `TrainingArguments.load_best_model_at_end` to upload best model.
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True

        logger.info(
            'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
        )
        combined_dict = {**args.to_sanitized_dict()}
        init_args = {}
        run_name = args.run_name

        if self._wandb.run is None:
            self._wandb.init(
                project=os.getenv("WANDB_PROJECT", "asyncval"),
                name=run_name,
                **init_args,
            )
        # add config parameters (run may have been created manually)
        self._wandb.config.update(combined_dict, allow_val_change=True)

    def log(self, args, logs, step):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args)
        self._wandb.log(logs)


class AsyncvalTensorBoardCallback(TensorBoardCallback):
    def log(self, args, logs, step):
        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()


class CallbackHandler:
    """ Internal class that just calls the list of callbacks in order. """

    def __init__(self, callbacks):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        self.callbacks.append(cb)

    def log(self, args, logs, step):
        return self.call_event("log", args, logs, step)

    def call_event(self, event, args, logs, step):
        for callback in self.callbacks:
            getattr(callback, event)(
                args,
                logs,
                step
            )


INTEGRATION_TO_CALLBACK = {
    "tensorboard": AsyncvalTensorBoardCallback,
    "wandb": AsyncvalWandbCallback,
}


def get_reporting_integration_callbacks(report_to):
    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )
    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]