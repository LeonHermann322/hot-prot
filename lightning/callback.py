from typing import Any
import lightning as l
import lightning.pytorch as pl
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from util.telegram import TelegramBot


class TelegramCallback(l.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.telegram = TelegramBot()
        self.batch_message = self.telegram.send_telegram("batach..")["result"][
            "message_id"
        ]

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.telegram.send_telegram(f"starting {stage}")

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        self.telegram.send_telegram(f"ending {stage}")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        val_loss = trainer.callback_metrics.get("val_loss")
        self.telegram.edit_text_message(
            self.batch_message,
            f"Epoch {trainer.current_epoch}/{trainer.max_epochs} : [{batch_idx}/{trainer.num_training_batches}] loss: {outputs['loss']} val_loss: {val_loss}",
        )

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        self.telegram.send_telegram(f"exception {exception}")

   