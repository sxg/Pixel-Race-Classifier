import torch
from torch import nn
import lightning.pytorch as pl
from torchvision import models
from torchvision.utils import make_grid
from torchmetrics import Accuracy, AUROC, ROC, Recall, Precision, F1Score
import matplotlib.pyplot as plt


class PixelModule(pl.LightningModule):
    def __init__(self, tasks, criterion):
        super().__init__()

        self.tasks = tasks
        self.save_hyperparameters()

        # Set up the model, loss function, and other metrics
        self.model = models.densenet121(weights="DEFAULT")
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, len(self.tasks)),
            nn.Sigmoid(),
        )
        self.criterion = criterion
        self.accuracy_fn = Accuracy(
            task="multiclass", num_classes=len(self.tasks), average=None
        )
        self.auroc_fn = AUROC(
            task="multiclass", num_classes=len(self.tasks), average=None
        )
        self.recall_fn = Recall(
            task="multiclass", num_classes=len(self.tasks), average=None
        )
        self.precision_fn = Precision(
            task="multiclass", num_classes=len(self.tasks), average=None
        )
        self.f1score_fn = F1Score(
            task="multiclass", num_classes=len(self.tasks), average=None
        )
        self.val_output = None
        self.val_labels = None

        # Flags for logging
        self.train_logged_images = False
        self.val_logged_images = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self(imgs)
        loss = self.criterion(output, labels)
        self.log("train/loss", loss)

        # Log sample data to TensorBoard
        if not self.train_logged_images:
            self.train_logged_images = True

            img_grid = make_grid(imgs, nrow=4)
            self.logger.experiment.add_image(
                "train/imgs", img_grid, self.current_epoch
            )
            self.logger.experiment.add_text(
                "train/labels",
                PixelModule._format_labels(labels, self.tasks),
                self.current_epoch,
            )
            self.logger.experiment.add_text(
                "train/preds",
                PixelModule._format_labels(output, self.tasks),
                self.current_epoch,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        labels_red = torch.argmax(labels, dim=1)
        output = self(imgs)

        if self.val_output is None:
            self.val_output = output
        else:
            self.val_output = torch.cat([self.val_output, output])

        if self.val_labels is None:
            self.val_labels = labels
        else:
            self.val_labels = torch.cat([self.val_labels, labels])

        loss = self.criterion(output, labels)
        self.log("val/loss", loss)

        # Calculate metrics for each task
        acc = self.accuracy_fn(output, labels_red)
        auroc = self.auroc_fn(output, labels_red.int())
        recall = self.recall_fn(output, labels_red.int())
        precision = self.precision_fn(output, labels_red.int())
        f1score = self.f1score_fn(output, labels_red.int())
        for i, task in enumerate(self.tasks):
            self.log(f"val/acc_{task}", acc[i])
            self.log(f"val/auroc_{task}", auroc[i])
            self.log(f"val/recall_{task}", recall[i])
            self.log(f"val/precision_{task}", precision[i])
            self.log(f"val/f1score_{task}", f1score[i])

        # Log sample data to TensorBoard
        if not self.val_logged_images:
            self.val_logged_images = True

            img_grid = make_grid(imgs, nrow=4)
            self.logger.experiment.add_image(
                "val/imgs", img_grid, self.current_epoch
            )
            self.logger.experiment.add_text(
                "val/labels",
                PixelModule._format_labels(labels, self.tasks),
                self.current_epoch,
            )
            self.logger.experiment.add_text(
                "val/preds",
                PixelModule._format_labels(output, self.tasks),
                self.current_epoch,
            )

        return {"val/loss": loss, "val/acc": acc, "val/auroc": auroc}

    def on_training_epoch_end(self):
        self.train_logged_images = False

    def on_validation_epoch_end(self):
        self.val_logged_images = False

        roc = ROC(task="multiclass", num_classes=len(self.tasks))
        val_labels_red = torch.argmax(self.val_labels, dim=1)
        fpr, tpr, _ = roc(self.val_output.float(), val_labels_red.int())
        for i in range(self.val_output.shape[1]):
            plt.figure()
            plt.plot(fpr[i].cpu(), tpr[i].cpu(), label="ROC Curve")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.plot([0, 1], [0, 1])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{self.tasks[i]} ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig(f"./Results/{self.tasks[i]} ROC Curve.png")
            plt.close()

        self.val_output = None
        self.val_labels = None

    def _format_labels(labels, tasks):
        string = ", ".join(tasks) + "  \n"
        for row_i in range(labels.shape[0]):
            str_labels = []
            for col_i in range(labels.shape[1]):
                str_labels += str(labels[row_i, col_i].item())
            string += ", ".join(str_labels) + "  \n"
        return string
